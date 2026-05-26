"""
A/IA Region Splitting and Classification Pipeline for MyoCAS

This script performs the prediction step for the first-stage active/inactive
region classification in MyoCAS.

Input time-lapse microscopy videos are optionally divided into spatial
subregions, such as 12 regions (3 × 4) or 48 regions (6 × 8). Each subregion
video is then processed using the trained A/IA classifier.

For each subregion video, grayscale frames are sampled, a median reference
image is calculated, and temporal motion features are extracted from the
absolute pixel-wise differences between each frame and the median reference
image. The trained DNN classifier then predicts whether each subregion is
active or inactive.

Classes:
    A  : active region
    IA : inactive region

Main processing steps:
    1. Optionally divide each input video into spatial subregions.
    2. Load each subregion video as sampled grayscale frames.
    3. Calculate a median reference image.
    4. Extract six temporal motion features.
    5. Apply the trained A/IA classifier.
    6. Save prediction results and sort videos into active/inactive folders.

Expected input:
    - Original time-lapse microscopy videos.
    - A trained A/IA classifier checkpoint containing both the model weights
      and fitted StandardScaler.

Expected output:
    - Divided subregion videos.
    - Active/inactive prediction folders.
    - prediction_result.csv containing file-level A/IA predictions.

Note:
    This script is intended for prediction. Model training should be performed
    using train_A_IA_classifier.py.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import shutil
from natsort import natsorted
from tqdm import tqdm
# ============================================================
# SETTINGS
# ============================================================

DIVIDED_MODE = "on"   # Whether to divide videos before prediction

BASIC_PATH=r"path/to/input_folder"
# Base directory containing Train, Val, Test, and Predict folders.
# Replace this example path with your own project directory.


PRED_DIR  = BASIC_PATH

BEST_MODEL = r"weights/MyoCAS_A_IA_classification.pth"
# Path to the trained model used for prediction.
# Replace this with the checkpoint file to be used.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_EXT = ['mp4','mov','wmv','mkv','mpeg','flv','webm','avi']

FRAME_STEP = 2
BG_THR = 50

N_WORKERS = max(1, int(multiprocessing.cpu_count() * 2/3))

ROWS = 3
COLS = 4  # 3 × 4 = 12 regions; 6 × 8 = 48 regions
# ============================================================
# VIDEO DIVISION (12 OR 48 SUBREGIONS)
# ============================================================
# ==========================
# single video divide worker
# ==========================
def divide_one_video(args):

    video_file, divided_dir, num_rows, num_cols = args

    path = os.path.join(PRED_DIR, video_file)
    cap = cv2.VideoCapture(path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    part_w, part_h = w // num_cols, h // num_rows

    writers = []
    name = os.path.splitext(video_file)[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for r in range(num_rows):
        for c in range(num_cols):

            out_path = os.path.join(
                divided_dir,
                f"{name}_({r*num_cols+c+1}).avi"
            )

            writers.append(
                cv2.VideoWriter(out_path, fourcc, fps, (part_w, part_h))
            )

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        for r in range(num_rows):
            for c in range(num_cols):

                x1, y1 = c*part_w, r*part_h
                x2, y2 = (c+1)*part_w, (r+1)*part_h

                writers[r*num_cols+c].write(frame[y1:y2, x1:x2])

    cap.release()

    for w in writers:
        w.release()

    return video_file


# ==========================
# multiprocessing divide
# ==========================
def divide_videos():

    print("\n===== DIVIDE MULTIPROCESS MODE =====")

    num_rows, num_cols = ROWS, COLS
    divided_dir = os.path.join(PRED_DIR, "Divided")
    os.makedirs(divided_dir, exist_ok=True)

    videos = [f for f in os.listdir(PRED_DIR)
              if f.split('.')[-1].lower() in VIDEO_EXT]

    tasks = [(v, divided_dir, num_rows, num_cols) for v in videos]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

        for _ in tqdm(ex.map(divide_one_video, tasks),
                      total=len(tasks),
                      desc="Dividing Videos"):
            pass

    print("Video division completed.")

    return divided_dir


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def load_video_frames(path):

    cap = cv2.VideoCapture(path)
    frames = []
    base_shape = None
    idx = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if idx % FRAME_STEP != 0:
            idx += 1
            continue

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if base_shape is None:
            base_shape = frame.shape
        elif frame.shape != base_shape:
            idx += 1
            continue

        frames.append(frame)
        idx += 1

    cap.release()

    if len(frames) < 3:
        return None

    return np.stack(frames)


def compute_motion_features(frames):

    ref=np.median(frames,axis=0).astype(np.uint8)
    valid_mask=ref>BG_THR

    if valid_mask.sum()<50:
        valid_mask=ref>np.percentile(ref,20)

    diff=np.abs(frames.astype(np.int16)-ref.astype(np.int16))
    temporal_curve=diff[:,valid_mask].mean(axis=1)

    temporal_std=float(temporal_curve.std())
    temporal_max=float(temporal_curve.max())
    temporal_min=float(temporal_curve.min())
    temporal_range=temporal_max-temporal_min

    p95=np.percentile(temporal_curve,95)
    p5=np.percentile(temporal_curve,5)
    temporal_range_p=float(p95-p5)


    return [
        temporal_std,
        temporal_max,
        temporal_min,
        temporal_range,
        temporal_range_p,
        temporal_curve.mean()
    ]



def extract_one(path):

    frames = load_video_frames(path)
    if frames is None:
        return None

    return path, compute_motion_features(frames)


# ============================================================
# MODEL
# ============================================================

class DNN(nn.Module):

    def __init__(self, in_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)



# ============================================================
# PREDICT PIPELINE
# ============================================================

def predict():

    folder = PRED_DIR

    if DIVIDED_MODE.lower() == "on":
        folder = divide_videos()

    videos = [os.path.join(folder,f)
              for f in natsorted(os.listdir(folder))
              if f.split('.')[-1].lower() in VIDEO_EXT]

    feats, paths = [], []

    print("\n===== FEATURE EXTRACT =====")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

        futures = [ex.submit(extract_one,p) for p in videos]

        for fut in tqdm(as_completed(futures), total=len(futures)):

            r = fut.result()
            if r is None:
                continue

            p,feat = r
            feats.append(feat)
            paths.append(p)

    feats = np.array(feats, np.float32)

    data = torch.load(BEST_MODEL, map_location=DEVICE)

    scaler = data["scaler"]
    X = scaler.transform(feats)
    X = torch.tensor(X).to(DEVICE)

    model = DNN(X.shape[1]).to(DEVICE)
    model.load_state_dict(data["model"])
    model.eval()

    with torch.no_grad():
        pred = torch.sigmoid(model(X)).cpu().numpy().flatten()

    active_dir   = os.path.join(PRED_DIR,"Active Objects")
    inactive_dir = os.path.join(PRED_DIR,"Inactive Objects")

    os.makedirs(active_dir,exist_ok=True)
    os.makedirs(inactive_dir,exist_ok=True)

    result=[]

    for p,v in natsorted(zip(paths,pred),
                         key=lambda x: os.path.basename(x[0])):

        label = "ACTIVE" if v>0.5 else "INACTIVE"
        fname = os.path.basename(p)

        dst = os.path.join(
            active_dir if label=="ACTIVE" else inactive_dir,
            fname
        )

        if not os.path.exists(dst):
            shutil.move(p,dst)

        print(fname,"→",label)
        result.append([fname,label])

    pd.DataFrame(result,
        columns=["file","prediction"]
    ).to_csv(os.path.join(PRED_DIR,"prediction_result.csv"),
             index=False)

    print("\n===== DONE =====")


# ============================================================
# AUTO RUN
# ============================================================

def auto_run():

    print("\nMyoCAS A/IA region splitting and classification started.\n")

    predict()

    print("\nPipeline finished.\n")

# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":
    auto_run()
