import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from config import STAGE2_MODEL
from utils import mapping

_model     = None
_processor = None


def load_model():
    global _model, _processor
    if _model:
        return _model, _processor

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
    )
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        STAGE2_MODEL, quantization_config=quant, device_map='cuda:0', low_cpu_mem_usage=True)
    p = AutoProcessor.from_pretrained(
        STAGE2_MODEL, min_pixels=16*28*28, max_pixels=256*28*28)
    m.eval()
    _model, _processor = m, p
    return m, p


def _load_slowfast(video_path, slow_n=2, fast_n=4):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale  = (64 * 28 * 28 / (W * H)) ** 0.5
    fast_W = max(28, int(W * scale // 28) * 28)
    fast_H = max(28, int(H * scale // 28) * 28)

    def read(indices, resize=None):
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if resize:
                    img = img.resize(resize, Image.LANCZOS)
                frames.append(img)
        return frames

    slow = read(np.linspace(0, total-1, slow_n, dtype=int))
    fast = read(np.linspace(0, total-1, fast_n, dtype=int), (fast_W, fast_H))
    cap.release()
    return slow, fast


def infer(video_path, obj_name, place_name):
    model, processor = load_model()
    slow, fast = _load_slowfast(video_path)

    prompt = (f"사고 유형: {obj_name} / 장소: {place_name}\n\n"
              f"이 교통사고 영상을 분석하여 아래 형식으로 답하세요:\n"
              f"특징: (사고 특징)\nA행동: (A의 행동)\nB행동: (B의 행동)\n"
              f"설명: (2~3문장 사고 상황 설명)")

    messages = [{"role": "user", "content":
        [{"type": "image", "image": f} for f in slow] +
        [{"type": "image", "image": f} for f in fast] +
        [{"type": "text",  "text": prompt}]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs,
                       videos=video_inputs, return_tensors='pt').to('cuda')

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    description = processor.batch_decode(
        generated, skip_special_tokens=True)[0].strip()

    return {'description': description, 'source': 'Qwen2.5-VL-7B-Instruct (4bit)'}
