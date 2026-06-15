# 교통사고 과실비율 분석 시스템

블랙박스 영상을 입력받아 사고 유형, 특징, A/B 행동을 분류하고 과실비율을 출력하는 end-to-end 파이프라인.

## 요구사항

- Python 3.10+
- CUDA GPU (VRAM 11GB 이상 권장)

## 설치

```bash
pip install -r requirements.txt
```

## 모델 다운로드

HuggingFace에서 모델을 다운로드.

```bash
huggingface-cli download openai/clip-vit-large-patch14
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

## 가중치

`weights/` 폴더
업로드 중입니다. 업로드 후 링크 올리겠습니다.

```
weights/
├── stage1_best.pt
├── stage3_best.pt
├── hierarchy.json
└── mapping.json
```

## 실행

```bash
python app.py
```

`http://localhost:8000`

## 파이프라인

1. Stage 1 (CLIP ViT-L/14): 사고 객체(4종), 장소 Top-3(15종) 분류
2. Stage 2 (Qwen2.5-VL-7B, 4bit): SlowFast 방식으로 사고 상황 자연어 묘사
3. Stage 3 (Qwen2.5-0.5B): 계층적 마스킹 기반 특징(59종) 및 A/B 행동 쌍(255종) 분류
4. hierarchy.json 매핑으로 최종 과실비율 산출
