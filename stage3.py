import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from config import STAGE3_CKPT, STAGE3_MODEL, MAPPING_PATH, HIERARCHY_PATH
from utils import mapping

with open(HIERARCHY_PATH, encoding='utf-8') as f:
    _hierarchy = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_FEATURES = len(mapping['feature'])
FEAT_ID2IDX  = {int(k): i for i, k in enumerate(mapping['feature'].keys())}
FEAT_IDX2ID  = {v: k for k, v in FEAT_ID2IDX.items()}
A_ID2IDX     = {int(k): i for i, k in enumerate(mapping['a_progress'].keys())}
B_ID2IDX     = {int(k): i for i, k in enumerate(mapping['b_progress'].keys())}

# 전체 valid (A, B) 쌍 구축
_feat_name2id = {v: int(k) for k, v in mapping['feature'].items()}
_a_name2id    = {v: int(k) for k, v in mapping['a_progress'].items()}
_b_name2id    = {v: int(k) for k, v in mapping['b_progress'].items()}

_ab_set = set()
for obj_name, places in _hierarchy.items():
    for place_name, features in places.items():
        for feat_name, pairs in features.items():
            for pair_key in pairs:
                a_name, b_name = pair_key.split('|')
                a_id = _a_name2id.get(a_name)
                b_id = _b_name2id.get(b_name)
                if a_id is not None and b_id is not None:
                    _ab_set.add((a_id, b_id))

AB_PAIRS = sorted(_ab_set)
AB2IDX   = {p: i for i, p in enumerate(AB_PAIRS)}
NUM_AB   = len(AB_PAIRS)


class Stage3Classifier(nn.Module):
    def __init__(self, model_name, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        for p in self.encoder.parameters():
            p.requires_grad = False
        d = self.encoder.config.hidden_size

        def mlp(n):
            return nn.Sequential(
                nn.Linear(d, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, n))

        self.feat_head = mlp(NUM_FEATURES)
        self.ab_head   = mlp(NUM_AB)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        vec  = (out.last_hidden_state.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return {'feat_logits': self.feat_head(vec), 'ab_logits': self.ab_head(vec)}


_model     = None
_tokenizer = None


def _valid_features(obj_name, place_name):
    try:
        return list(_hierarchy[obj_name][place_name].keys())
    except KeyError:
        return []


def _valid_pairs(obj_name, place_name, feat_name):
    try:
        return list(_hierarchy[obj_name][place_name][feat_name].keys())
    except KeyError:
        return []


def load_model():
    global _model, _tokenizer
    if _model:
        return _model, _tokenizer

    tok = AutoTokenizer.from_pretrained(STAGE3_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = Stage3Classifier(STAGE3_MODEL).to(device)
    ckpt = torch.load(STAGE3_CKPT, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()

    _model, _tokenizer = m, tok
    return m, tok


def infer(description, obj_name, place_names):
    model, tokenizer = load_model()

    enc   = tokenizer(description, max_length=256, padding='max_length',
                      truncation=True, return_tensors='pt')
    iids  = enc['input_ids'].to(device)
    amask = enc['attention_mask'].to(device)

    # Step 1: feature 마스킹
    feat_mask = torch.full((1, NUM_FEATURES), float('-inf'), device=device)
    for place_name in place_names:
        for feat_name in _valid_features(obj_name, place_name):
            fid = _feat_name2id.get(feat_name)
            if fid is not None and fid in FEAT_ID2IDX:
                feat_mask[0, FEAT_ID2IDX[fid]] = 0.0
    if (feat_mask == 0.0).sum() == 0:
        feat_mask[:] = 0.0

    with torch.no_grad():
        out = model(iids, amask)
        pred_feat_idx  = (out['feat_logits'] + feat_mask).argmax(1).item()

    pred_feat_id   = FEAT_IDX2ID.get(pred_feat_idx, -1)
    pred_feat_name = mapping['feature'].get(str(pred_feat_id), '?')

    # Step 2: (A, B) 쌍 마스킹
    ab_mask = torch.full((1, NUM_AB), float('-inf'), device=device)
    for place_name in place_names:
        for pair_key in _valid_pairs(obj_name, place_name, pred_feat_name):
            a_name, b_name = pair_key.split('|')
            a_id = _a_name2id.get(a_name)
            b_id = _b_name2id.get(b_name)
            if a_id is not None and b_id is not None:
                idx = AB2IDX.get((a_id, b_id), -1)
                if idx >= 0:
                    ab_mask[0, idx] = 0.0
    if (ab_mask == 0.0).sum() == 0:
        ab_mask[:] = 0.0

    with torch.no_grad():
        pred_ab_idx = (out['ab_logits'] + ab_mask).argmax(1).item()

    pred_a_id, pred_b_id = AB_PAIRS[pred_ab_idx]
    pred_a = mapping['a_progress'].get(str(pred_a_id), '?')
    pred_b = mapping['b_progress'].get(str(pred_b_id), '?')

    return {'pred_feature': pred_feat_name, 'pred_a': pred_a, 'pred_b': pred_b}