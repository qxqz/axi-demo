import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from config import STAGE1_CKPT, STAGE1_MODEL
from utils import mapping, extract_frames

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return (attn @ v).mean(dim=1)


class CLIPAccidentClassifier(nn.Module):
    def __init__(self, clip_model_name, num_objects=4, num_places=15, embed_dim=768):
        super().__init__()
        self.clip_model        = CLIPModel.from_pretrained(clip_model_name)
        self.temporal_attention = TemporalAttention(embed_dim)
        self.object_classifier  = nn.Linear(embed_dim, num_objects)
        self.place_classifier   = nn.Linear(embed_dim, num_places)

    def forward(self, pixel_values):
        B, N, C, H, W = pixel_values.shape
        pv    = pixel_values.view(B * N, C, H, W)
        feats = self.clip_model.vision_model(pixel_values=pv).pooler_output
        feats = feats.view(B, N, -1)
        vec   = self.temporal_attention(feats)
        return {
            'object_logits': self.object_classifier(vec),
            'place_logits':  self.place_classifier(vec),
        }


_model     = None
_processor = None


def load_model():
    global _model, _processor

    if _model:
        return _model, _processor

    processor = CLIPProcessor.from_pretrained(STAGE1_MODEL)
    ckpt      = torch.load(STAGE1_CKPT, map_location=device)

    model = CLIPAccidentClassifier(STAGE1_MODEL).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _model, _processor = model, processor
    return model, processor


def infer(video_path):
    model, processor = load_model()
    frames = extract_frames(video_path, n=4)

    inputs       = processor(images=frames, return_tensors='pt', padding=True)
    pixel_values = inputs['pixel_values'].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(pixel_values)

    obj_probs   = torch.softmax(out['object_logits'], dim=1)[0]
    place_probs = torch.softmax(out['place_logits'],  dim=1)[0]
    pred_obj    = obj_probs.argmax().item()
    top3_place  = place_probs.topk(3).indices.tolist()

    return {
        'pred_object':      pred_obj,
        'pred_obj_name':    mapping['object'][str(pred_obj)],
        'top3_place_ids':   top3_place,
        'top3_place_names': [mapping['place'][str(p)] for p in top3_place],
        'object_probs':     {mapping['object'][str(i)]: round(p.item() * 100, 1)
                             for i, p in enumerate(obj_probs)},
    }