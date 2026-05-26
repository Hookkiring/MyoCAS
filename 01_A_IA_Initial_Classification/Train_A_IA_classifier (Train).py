"""
A/IA Initial Classification for MyoCAS

This script trains, tests, and applies a binary classifier for the first-stage
active/inactive region classification in MyoCAS.

The input videos are subregion-level time-lapse videos generated from the
original microscopy videos. Each video is loaded as a grayscale frame sequence.
A median reference image is calculated from the sampled frames, and temporal
motion features are extracted from the absolute pixel-wise differences between
each frame and the median reference image.

The extracted features are used to train a DNN-based binary classifier.

Classes:
    0: inactive region (IA)
    1: active region (A)

Main processing steps:
    1. Load time-lapse video frames.
    2. Calculate a median reference image.
    3. Generate a temporal motion curve from frame-wise intensity differences.
    4. Extract six temporal motion features.
    5. Train, test, or apply the A/IA classifier.

Expected training directory structure:
    BASIC_PATH/
        Train/
            Active/
            Inactive/
        Val/
            Active/
            Inactive/
        Test/
            Active/
            Inactive/

Outputs:
    - Trained model weights (.pth)
    - Feature cache CSV files
    - Prediction result CSV file
    - Active/inactive prediction folders when prediction mode is used

Note:
    Before public release, replace all local absolute paths with relative or
    example paths. This script is part of the initial activity classification
    step before optical flow-derived displacement analysis and A/IA/N
    classification.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.preprocessing import StandardScaler
import shutil
from natsort import natsorted
# ============================================================
# SETTINGS
# ============================================================

MODE = "train"  # Options: "train", "test", or "predict"

BASIC_PATH=r"path/to/input_folder"
# Base directory containing Train, Val, Test, and Predict folders.
# Replace this example path with your own project directory.

TRAIN_DIR=os.path.join(BASIC_PATH,"Train")
VAL_DIR  =os.path.join(BASIC_PATH,"Val")
TEST_DIR =os.path.join(BASIC_PATH,"Test")
PRED_DIR =os.path.join(BASIC_PATH)

WEIGHT_DIR=os.path.join(BASIC_PATH,"dnn_weights")

if MODE.lower() in ["train", "test"]:
    os.makedirs(WEIGHT_DIR, exist_ok=True)


BEST_MODEL = r"weights/MyoCAS_A_IA_classification.pth"
# Path to the trained model used for prediction.
# Replace this with the checkpoint file to be used.

# Feature cache file for each dataset split
FEATURE_CACHE={
    "train":os.path.join(BASIC_PATH,"train_feature_cache_features.csv"),
    "val":os.path.join(BASIC_PATH,"val_feature_cache_features.csv"),
    "test":os.path.join(BASIC_PATH,"test_feature_cache_features.csv"),
    "predict":os.path.join(BASIC_PATH,"predict_feature_cache_features.csv"),
}
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

VIDEO_EXT=['mp4','mov','wmv','mkv','mpeg','flv','webm','avi']

FRAME_STEP=2
BG_THR=50
FEATURE_DIM = 6

EPOCHS=100
BATCH_SIZE=32
LR=1e-3

N_WORKERS=max(1,multiprocessing.cpu_count()-2)

# ============================================================
# FEATURE CACHE
# ============================================================

def load_feature_cache(split):
    '''
    Load cached motion features for the specified dataset split.
    '''

    path=FEATURE_CACHE[split]

    if os.path.exists(path):
        df=pd.read_csv(path)
        cache={row["path"]:row.to_dict() for _,row in df.iterrows()}
    else:
        cache={}

    return cache


def save_feature_cache(cache,split):
    '''
    Save extracted motion features to a CSV cache file.
    '''

    path=FEATURE_CACHE[split]

    df=pd.DataFrame(list(cache.values()))
    df=df.replace([np.inf,-np.inf],np.nan).dropna()

    df.to_csv(path,index=False)


# ============================================================
# VIDEO LOAD
# ============================================================

def load_video_frames(path):
    '''
    Load a video file and return sampled grayscale frames.

    Frames are sampled according to FRAME_STEP. Frames with inconsistent image
    sizes are skipped to avoid shape mismatch during feature extraction.
    '''
    cap=cv2.VideoCapture(path)
    frames=[]
    base_shape=None
    idx=0

    while True:
        ret,frame=cap.read()
        if not ret: break

        if idx%FRAME_STEP!=0:
            idx+=1
            continue

        if frame.ndim==3:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if base_shape is None:
            base_shape=frame.shape
        elif frame.shape!=base_shape:
            idx+=1
            continue

        frames.append(frame)
        idx+=1

    cap.release()

    if len(frames)<3:
        return None

    return np.stack(frames)


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_motion_features(frames):
    '''
    Extract six temporal motion features from a grayscale video sequence.

    A median reference image is calculated from the sampled frames. A temporal
    motion curve is generated by measuring the mean absolute pixel-wise difference
    between each frame and the median reference image within the foreground mask.
    '''

    ref = np.median(frames, axis=0).astype(np.uint8)

    valid_mask = ref > BG_THR

    # Fallback foreground mask when too few pixels pass the default threshold
    if valid_mask.sum() < 50:
        valid_mask = ref > np.percentile(ref, 20)

    diff = np.abs(frames.astype(np.int16) - ref.astype(np.int16))
    temporal_curve = diff[:, valid_mask].mean(axis=1)

    temporal_std = float(temporal_curve.std())
    temporal_max = float(temporal_curve.max())
    temporal_min = float(temporal_curve.min())
    temporal_range = temporal_max - temporal_min

    p95 = np.percentile(temporal_curve, 95)
    p5 = np.percentile(temporal_curve, 5)
    temporal_range_p = float(p95 - p5)

    temporal_mean = float(temporal_curve.mean())

    return [
        temporal_std,
        temporal_max,
        temporal_min,
        temporal_range,
        temporal_range_p,
        temporal_mean
    ]

def extract_one(args):

    path,label=args

    frames=load_video_frames(path)
    if frames is None:
        return None

    feat=compute_motion_features(frames)
    return path,feat,label


# ============================================================
# DATASET LOAD + CACHE
# ============================================================

def load_dataset(folder,split):
    '''
    Load labeled videos from Active and Inactive folders and extract motion features.

    Folder names containing "active" are assigned label 1, and folder names
    containing "inactive" are assigned label 0.
    '''
    cache=load_feature_cache(split)

    tasks=[]
    X,y=[] ,[]

    for name in os.listdir(folder):

        sub=os.path.join(folder,name)
        if not os.path.isdir(sub):
            continue

        if "inactive" in name.lower():
            label=0
        elif "active" in name.lower():
            label=1
        else:
            continue

        for f in os.listdir(sub):

            if f.split('.')[-1].lower() not in VIDEO_EXT:
                continue

            path=os.path.join(sub,f)

            if path in cache:

                row=cache[path]
                feat=[row[f"f{i}"] for i in range(FEATURE_DIM)]

                X.append(feat)
                y.append(label)

            else:
                tasks.append((path,label))

    if tasks:

        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

            futures=[ex.submit(extract_one,t) for t in tasks]

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc=f"Feature extracting ({split})"):

                r=fut.result()
                if r is None:
                    continue

                path,feat,label=r

                X.append(feat)
                y.append(label)

                cache[path]={
                    "path":path,
                    "label":label,
                    **{f"f{i}":feat[i] for i in range(len(feat))}
                }

        save_feature_cache(cache,split)

    return np.array(X,np.float32),np.array(y,np.float32)
def load_predict_dataset(folder):

    split="predict"
    cache=load_feature_cache(split)

    X=[]
    paths=[]
    tasks=[]

    for f in os.listdir(folder):

        if f.split('.')[-1].lower() not in VIDEO_EXT:
            continue

        path=os.path.join(folder,f)

        if path in cache:

            row=cache[path]
            feat=[row[f"f{i}"] for i in range(FEATURE_DIM)]

            X.append(feat)
            paths.append(path)

        else:
            tasks.append(path)

    # Extract features for videos that are not already stored in the cache
    if tasks:

        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

            futures=[ex.submit(extract_one,(p,0)) for p in tasks]

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Feature extracting (predict)"):

                r=fut.result()
                if r is None:
                    continue

                path,feat,_=r

                X.append(feat)
                paths.append(path)

                cache[path]={
                    "path":path,
                    **{f"f{i}":feat[i] for i in range(len(feat))}
                }

        save_feature_cache(cache,split)

    return np.array(X,np.float32),paths

# ============================================================
# MODEL
# ============================================================

class DNN(nn.Module):
    '''
    Fully connected neural network for binary A/IA classification.

    The model receives six temporal motion features and outputs one logit for
    active/inactive classification.
    '''
    def __init__(self,in_dim):
        super().__init__()

        self.net=nn.Sequential(
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
# TRAIN
# ============================================================

def train():
    '''
    Train the A/IA classifier using the training and validation datasets.

    The extracted features are standardized with StandardScaler. The trained model,
    fitted scaler, and validation accuracy are saved together in each checkpoint.
    '''
    X_train,y_train=load_dataset(TRAIN_DIR,"train")
    X_val,y_val=load_dataset(VAL_DIR,"val")

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_val=scaler.transform(X_val)

    X_train=torch.tensor(X_train).to(DEVICE)
    y_train=torch.tensor(y_train).unsqueeze(1).to(DEVICE)
    X_val=torch.tensor(X_val).to(DEVICE)
    y_val=torch.tensor(y_val).unsqueeze(1).to(DEVICE)

    model=DNN(X_train.shape[1]).to(DEVICE)

    optimizer=torch.optim.Adam(model.parameters(),LR)
    loss_fn=nn.BCEWithLogitsLoss()

    best_acc=0

    for epoch in range(EPOCHS):

        model.train()

        idx=torch.randperm(len(X_train))

        for i in range(0,len(idx),BATCH_SIZE):

            b=idx[i:i+BATCH_SIZE]

            xb=X_train[b]
            yb=y_train[b]

            pred=model(xb)
            loss=loss_fn(pred,yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():

            pred=torch.sigmoid(model(X_val))
            pred=(pred>0.5).float()
            acc=(pred==y_val).float().mean().item()

        print(f"Epoch {epoch+1}/{EPOCHS} ACC={acc:.4f}")

        save_dict={"model":model.state_dict(),"scaler":scaler,"acc":acc}

        torch.save(save_dict,
                   os.path.join(WEIGHT_DIR,
                                f"epoch_{epoch:03d}_acc_{acc:.4f}.pth"))

        if acc>best_acc:
            best_acc=acc
            torch.save(save_dict,
                       os.path.join(WEIGHT_DIR,"best_model.pth"))

    print("\nBEST VAL ACC:",best_acc)


# ============================================================
# TEST
# ============================================================

def test():
    '''
    Evaluate all saved checkpoint files on the test dataset and report the best model.
    '''
    X_test,y_test=load_dataset(TEST_DIR,"test")

    best_acc=0
    best_file=None

    for f in sorted(os.listdir(WEIGHT_DIR)):

        if not f.endswith(".pth"):
            continue

        data=torch.load(os.path.join(WEIGHT_DIR,f))

        scaler=data["scaler"]
        X_scaled=scaler.transform(X_test)

        X_scaled=torch.tensor(X_scaled).to(DEVICE)
        y_test_t=torch.tensor(y_test).unsqueeze(1).to(DEVICE)

        model=DNN(X_scaled.shape[1]).to(DEVICE)
        model.load_state_dict(data["model"])
        model.eval()

        with torch.no_grad():

            pred=torch.sigmoid(model(X_scaled))
            pred=(pred>0.5).float()
            acc=(pred==y_test_t).float().mean().item()

        print(f"{f} → ACC {acc:.4f}")

        if acc>best_acc:
            best_acc=acc
            best_file=f

    print("\nBEST MODEL :",best_file)
    print("BEST ACC :",best_acc)
    
# ============================================================
# PREDICT
# ============================================================
def predict():
    '''
    Apply the trained A/IA classifier to unlabeled prediction videos.

    Prediction results are saved as a CSV file, and videos are sorted into active
    or inactive output folders.
    '''
    weight_path=BEST_MODEL

    if not os.path.exists(weight_path):
        print("BEST_MODEL file was not found.")
        return

    data=torch.load(weight_path, map_location=DEVICE)

    scaler=data["scaler"]

    model=DNN(FEATURE_DIM).to(DEVICE)
    model.load_state_dict(data["model"])
    model.eval()

    X_pred,paths=load_predict_dataset(PRED_DIR)

    if len(X_pred)==0:
        print("No prediction data found.")
        return

    X_scaled=scaler.transform(X_pred)
    X_scaled=torch.tensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        pred=torch.sigmoid(model(X_scaled))
        pred=(pred>0.5).cpu().numpy().flatten()

    print("\n===== PREDICTION RESULT =====")

    # Create output folders for prediction results
    active_dir=os.path.join(PRED_DIR,"Active Objects")
    inactive_dir=os.path.join(PRED_DIR,"Inactive Objects")

    os.makedirs(active_dir,exist_ok=True)
    os.makedirs(inactive_dir,exist_ok=True)

    result=[]

    # Apply natural sorting to prediction results
    sorted_items = natsorted(
        zip(paths, pred),
        key=lambda x: os.path.basename(x[0])
    )

    for p,v in sorted_items:

        label="ACTIVE" if v else "INACTIVE"
        fname=os.path.basename(p)

        print(fname,"→",label)

        # Move each video to the corresponding prediction folder
        if label=="ACTIVE":
            dst=os.path.join(active_dir,fname)
        else:
            dst=os.path.join(inactive_dir,fname)
        # This moves the original video file. Use copy instead if the original files should be preserved.
        if not os.path.exists(dst): 
            shutil.move(p,dst)
        
        result.append([fname,label])


    # Save prediction results as a CSV file
    pd.DataFrame(result,
                 columns=["file","prediction"]
                 ).to_csv(
        os.path.join(PRED_DIR,"prediction_result.csv"),
        index=False
    )
    print("Total files in prediction directory:", len(os.listdir(PRED_DIR)))
    print("Number of videos to classify:", len(paths))
    print("\nFile sorting completed and prediction_result.csv was saved.")


# ============================================================
# MAIN
# ============================================================
if __name__=="__main__":

    if MODE.lower()=="train":
        train()

    elif MODE.lower()=="test":
        test()

    elif MODE.lower()=="predict":
        predict()

