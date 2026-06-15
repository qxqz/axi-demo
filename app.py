import shutil, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import stage1, stage2, stage3
from utils import mapping, get_negligence
from config import UPLOAD_DIR

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(400, "mp4 파일만 지원합니다.")

    video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        s1 = stage1.infer(video_path)
        s2 = stage2.infer(
            video_path,
            s1['pred_obj_name'],
            s1['top3_place_names'][0],
        )
        s3 = stage3.infer(
            s2['description'],
            s1['pred_obj_name'],
            s1['top3_place_names'],
        )

        neg_a, neg_b = None, None
        for place_name in s1['top3_place_names']:
            neg_a, neg_b = get_negligence(
                s1['pred_obj_name'], s1['top3_place_ids'],
                s3['pred_feature'], s3['pred_a'], s3['pred_b'])
            if neg_a is not None:
                break

        return {
            'stage1':     s1,
            'stage2':     s2,
            'stage3':     s3,
            'negligence': {'a': neg_a, 'b': neg_b},
        }
    finally:
        video_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    print("Loading models...")
    stage1.load_model()
    stage3.load_model()
    print("Ready.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
