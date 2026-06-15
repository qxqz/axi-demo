from pathlib import Path

BASE_DIR       = Path(__file__).parent
WEIGHTS_DIR    = BASE_DIR / 'weights'

STAGE1_CKPT    = WEIGHTS_DIR / 'stage1_best.pt'
STAGE3_CKPT    = WEIGHTS_DIR / 'stage3_best.pt'
HIERARCHY_PATH = WEIGHTS_DIR / 'hierarchy.json'
MAPPING_PATH   = WEIGHTS_DIR / 'mapping.json'

STAGE1_MODEL   = 'openai/clip-vit-large-patch14'
STAGE2_MODEL   = 'Qwen/Qwen2.5-VL-7B-Instruct'
STAGE3_MODEL   = 'Qwen/Qwen2.5-0.5B-Instruct'

UPLOAD_DIR     = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
