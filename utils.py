import json
import numpy as np
import cv2
from PIL import Image
from config import HIERARCHY_PATH, MAPPING_PATH

with open(MAPPING_PATH, encoding='utf-8') as f:
    mapping = json.load(f)
with open(HIERARCHY_PATH, encoding='utf-8') as f:
    hierarchy = json.load(f)


def extract_frames(video_path, n=4):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def get_negligence(obj_name, place_ids, feat, a, b):
    pair_key = f"{a}|{b}"
    for place_id in place_ids:
        place_name = mapping['place'].get(str(place_id), '')
        try:
            entry = hierarchy[obj_name][place_name][feat][pair_key]
            return entry['negligence_A'], entry['negligence_B']
        except KeyError:
            continue
    return None, None
