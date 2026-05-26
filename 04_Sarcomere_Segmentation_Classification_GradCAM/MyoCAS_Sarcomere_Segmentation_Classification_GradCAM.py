"""
MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py

Sarcomere mask-based maturation classification and Grad-CAM visualization
pipeline for MyoCAS.

This script first applies a trained sarcomere segmentation model
(DualNet: CustomHRNet + DeepLabV3) to input images and generates binary
sarcomere masks. The generated masks are then converted to three-channel
images and used as inputs for an EfficientNet-B4 classifier.

The classifier predicts sarcomere maturation states into three classes:
    0: Pre
    1: Nascent
    2: Mature

Main functions:
    1. Generate sarcomere segmentation masks using a trained DualNet checkpoint.
    2. Train an EfficientNet-B4 classifier using segmentation-mask inputs.
    3. Resume classifier training from saved checkpoints.
    4. Evaluate saved checkpoints on a test dataset.
    5. Export checkpoint accuracy results to Excel.
    6. Generate confusion matrix plots.
    7. Generate Grad-CAM and Grad-CAM++ visualizations.
    8. Classify all images in a selected folder into Pre/Nascent/Mature folders.

Important:
    The scientific logic and model flow should remain unchanged.
    The sarcomere segmentation mask is used as the classifier input.

KR:
MyoCAS의 sarcomere mask 기반 성숙도 분류 및 Grad-CAM visualization pipeline입니다.

이 script는 먼저 학습된 sarcomere segmentation model
(DualNet: CustomHRNet + DeepLabV3)을 입력 이미지에 적용하여 binary
sarcomere mask를 생성합니다. 생성된 mask는 3-channel image로 변환된 뒤
EfficientNet-B4 classifier의 입력으로 사용됩니다.

Classifier는 sarcomere maturation state를 세 class로 분류합니다.
    0: Pre
    1: Nascent
    2: Mature

주요 기능:
    1. 학습된 DualNet checkpoint를 이용한 sarcomere segmentation mask 생성
    2. Segmentation mask 입력 기반 EfficientNet-B4 classifier 학습
    3. 저장된 checkpoint에서 classifier 학습 재개
    4. Test dataset에서 checkpoint 평가
    5. Checkpoint별 accuracy 결과를 Excel로 저장
    6. Confusion matrix plot 생성
    7. Grad-CAM 및 Grad-CAM++ visualization 생성
    8. 선택한 folder 내 이미지를 Pre/Nascent/Mature folder로 분류

중요:
    과학적 로직과 model flow는 변경하지 않습니다.
    Sarcomere segmentation mask가 classifier 입력으로 사용됩니다.
"""

import os
from pathlib import Path
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from efficientnet_pytorch import EfficientNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import random
from tifffile import imread as tif_imread
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

# ====================== 재현성 보장 ======================
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ============================================================
# User settings
# ============================================================

# Training mode for the EfficientNet-B4 sarcomere maturation classifier.
# "on"  : train the classifier and save checkpoints.
# "off" : do not train.
# KR: EfficientNet-B4 기반 sarcomere maturation classifier 학습 여부입니다.
TRAIN_MODE = "on"

# Test mode for evaluating classifier checkpoints.
# "on"  : evaluate saved checkpoints on the test dataset.
# "off" : skip checkpoint evaluation.
# KR: 저장된 classifier checkpoint를 test dataset에서 평가할지 여부입니다.
TEST_MODE = "off"

# Confusion matrix export mode.
# "on"  : generate a confusion matrix using BEST_CKPT and the test dataset.
# "off" : skip confusion matrix generation.
# KR: BEST_CKPT와 test dataset을 이용해 confusion matrix를 생성할지 여부입니다.
MATRIX = "off"

# Grad-CAM visualization mode.
# "off"    : disable Grad-CAM.
# "single" : generate Grad-CAM/Grad-CAM++ for one selected image.
# "batch"  : generate Grad-CAM/Grad-CAM++ for all images in a folder.
# KR: Grad-CAM visualization mode입니다.
GRAD_CAM_MODE = "off"

# Folder-level classification mode.
# "on"  : classify all images in CLASSIFY_GRAD_INPUT_DIR into class folders.
# "off" : skip folder-level classification.
# KR: 지정한 folder 내 이미지를 Pre/Nascent/Mature로 분류할지 여부입니다.
CLASSIFICATION_MODE = "off"

# Continue training mode.
# "on"  : resume training from the latest epoch checkpoint if available.
# "off" : train from scratch.
# KR: 학습을 중간 checkpoint부터 이어서 진행할지 여부입니다.
KEEP_MODE = "on"

# EfficientNet-B4 classifier checkpoint used for classification and Grad-CAM.
# KR: classification 및 Grad-CAM에 사용할 EfficientNet-B4 classifier checkpoint입니다.
CLASSIFY_EPOCH = r"weights/efficient_net_sarcomere_best.pth"

# Input folder for classification or Grad-CAM batch mode.
# KR: classification 또는 Grad-CAM batch mode에서 사용할 입력 이미지 folder입니다.
CLASSIFY_GRAD_INPUT_DIR = Path(r"path/to/input_images")

# Root directory for classifier training, validation, testing, and outputs.
# Expected structure:
#   ROOT/
#       Train/
#           Pre/
#           Nascent/
#           Mature/
#       Val/
#           Pre/
#           Nascent/
#           Mature/
#       Test_re/
#           Pre/
#           Nascent/
#           Mature/
#
# KR:
# Classifier 학습/검증/테스트 및 output 저장을 위한 root directory입니다.
ROOT = Path(r"path/to/sarcomere_classification_dataset")

TRAIN_DIR = ROOT / "Train"
VAL_DIR = ROOT / "Val"
TEST_DIR = ROOT / "Test"

# Output folder name for Grad-CAM batch results.
# KR: Grad-CAM batch 결과를 저장할 folder 이름입니다.
GRAD_CAM_FOLDER_NAME = "GradCAM_Results"

# Class names for sarcomere maturation classification.
# KR: sarcomere maturation classification class 이름입니다.
CLASS_NAME = ["Pre", "Nascent", "Mature"]

# Checkpoint and TensorBoard log directories.
# KR: checkpoint 및 TensorBoard log 저장 경로입니다.
CHECKPOINT_DIR = ROOT / "checkpoints"
TB_LOG_DIR = ROOT / "tensorboard_logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
TB_LOG_DIR.mkdir(exist_ok=True)

# Best classifier checkpoint path.
# KR: 가장 좋은 classifier checkpoint 저장 경로입니다.
BEST_CKPT = CHECKPOINT_DIR / "best.pth"

# Training parameters.
# KR: classifier 학습 설정입니다.
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 50
NUM_CLASSES = 3
IMG_SIZE = 512

# Excel file name for checkpoint evaluation results.
# KR: checkpoint 평가 결과를 저장할 Excel file 이름입니다.
EXCEL_NAME = "Checkpoint_Evaluation_Results"

# Sarcomere segmentation checkpoint.
# This checkpoint is generated by the sarcomere segmentation training script.
# KR:
# Sarcomere segmentation checkpoint입니다.
# 이전 segmentation 학습 코드에서 얻은 DualNet checkpoint를 지정합니다.
BEST_SEG_CHECK_SAR = r"weights/HRNet_DeepLabV3_best_Sarcomere_segmentation.pth"

# Segmentation mask cache mode.
# True  : save generated segmentation masks and reuse them.
# False : do not save generated masks.
# KR: 생성된 segmentation mask를 저장하여 재사용할지 여부입니다.
SAVE_SEG_MASK = True

# Directory for cached sarcomere segmentation masks.
# KR: 생성된 sarcomere segmentation mask cache 저장 경로입니다.
SAVE_MASK_DIR = ROOT / "Saved_Segmentation_Masks"
SAVE_MASK_DIR.mkdir(exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ============================================================
# Sarcomere segmentation model definition
# ============================================================
# Defines the DualNet segmentation model used to generate sarcomere masks.
# KR: Sarcomere mask 생성을 위한 DualNet segmentation model을 정의합니다.
# ============================================================


if TRAIN_MODE == "on":
    CURRENT_MODE = "Train"
elif TEST_MODE == "on":
    CURRENT_MODE = "Test"
elif GRAD_CAM_MODE in ["single", "batch"]:
    CURRENT_MODE = "GradCAM"
elif CLASSIFICATION_MODE == "on":
    CURRENT_MODE = "Classification"
else:
    CURRENT_MODE = "Other"

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class CustomHRNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)

        self.stage2_branch1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage2_branch2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage3_branch1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.stage3_branch2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.stage3_branch3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv = nn.Conv2d(32 + 64 + 128, 256, 1)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers.append(block(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)

        b1 = self.stage2_branch1(x)
        b2 = self.stage2_branch2(x)

        b1_3 = self.stage3_branch1(b1)
        b2_3 = self.stage3_branch2(b2)
        b3_3 = self.stage3_branch3(b2)

        b2_3_up = F.interpolate(b2_3, size=b1_3.shape[-2:], mode='bilinear')
        b3_3_up = F.interpolate(b3_3, size=b1_3.shape[-2:], mode='bilinear')

        feat = torch.cat([b1_3, b2_3_up, b3_3_up], dim=1)
        feat = F.relu(self.fuse_conv(feat))

        feat = F.interpolate(feat, size=input_shape, mode='bilinear')
        return self.classifier(feat)


def convert_batchnorm_to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            num_groups = min(32, num_features)
            num_groups = max(1, num_groups)
            setattr(module, name, nn.GroupNorm(num_groups, num_features))
        else:
            convert_batchnorm_to_groupnorm(child)


class DualNet(nn.Module):
    def __init__(self, hrnet, deeplabv3, num_classes=2):
        super().__init__()
        self.hrnet = hrnet
        self.deeplabv3 = deeplabv3
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        hrnet_out = self.hrnet(x)
        deeplab_out = self.deeplabv3(x)['out']

        if hrnet_out.shape[-2:] != deeplab_out.shape[-2:]:
            deeplab_out = F.interpolate(deeplab_out, size=hrnet_out.shape[-2:], mode='bilinear')

        fusion = torch.cat([hrnet_out, deeplab_out], dim=1)
        out = self.fuse(fusion)
        return F.interpolate(out, size=input_shape, mode='bilinear')


from torchvision.models.segmentation import deeplabv3_resnet101

# 🔧 세그멘테이션용 transform (inference)
transform_infer_seg = A.Compose([
    A.Normalize(mean=3.0, std=3.0),
    ToTensorV2()
])


def build_sarco_model(ckpt_path, device):
    hrnet = CustomHRNet(num_classes=2)
    deeplab = deeplabv3_resnet101(weights=None, num_classes=2)

    convert_batchnorm_to_groupnorm(hrnet)
    convert_batchnorm_to_groupnorm(deeplab)

    model = DualNet(hrnet, deeplab)
    weight = torch.load(ckpt_path, map_location=device)
    # 체크포인트 구조 대응
    if isinstance(weight, dict) and "model_state_dict" in weight:
        state = weight["model_state_dict"]
    else:
        state = weight
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# 🔥 Global sarcomere segmentation model
seg_model = build_sarco_model(BEST_SEG_CHECK_SAR, device)


@torch.no_grad()
def segment_sarcomere(img_path: Path) -> np.ndarray:
    """
    segmentation-only 코드와 100% 동일한 마스크 생성 함수
    """

    # 1) TIFF → max projection 포함하는 전처리
    gray = imread_gray(img_path)

    # 2) albumentations는 RGB를 요구 → 동일하게 변환
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 3) 동일 transform
    t = transform_infer_seg(image=img_rgb)
    tensor = t["image"].unsqueeze(0).to(device)

    # 4) Segmentation prediction
    pred = seg_model(tensor)

    # 5) mask 생성
    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255

    return mask


# ============================================================
# 🔥 Segmentation-only 코드와 동일한 TIFF/PNG 전처리
# ============================================================

bright_boost = 1  # 너 segmentation-only 코드와 동일하게 설정

def imread_gray(path: Path, boost=bright_boost) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        arr = tif_imread(str(path))
        arr = np.asarray(arr)

        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3:
            img = arr[..., 0] if arr.shape[-1] in (3, 4) else arr.max(axis=0)
        else:
            while arr.ndim > 2:
                arr = arr.max(axis=0)
            img = arr
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    img = np.clip(img * boost, 0, 255)
    return img.astype(np.uint8)


def save_gray(path: Path, arr: np.ndarray):
    arr_uint8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(path), arr_uint8)

# ============================================================
# 🔥 Guided Backpropagation
# ============================================================
class GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        guided_grad = grad_output.clone()
        guided_grad[input < 0] = 0
        guided_grad[grad_output < 0] = 0
        return guided_grad


def apply_guided_backprop(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            module.forward = lambda x: GuidedBackpropReLU.apply(x)
    return model


# ============================================================
# 🔥 LRP (Layer-wise Relevance Propagation) – 그대로 둠
# ============================================================
class LRP:
    def __init__(self, model):
        self.model = model.eval()

    def relprop(self, R, module, input, output, eps=1e-6):
        if isinstance(module, nn.Conv2d):
            z = F.conv2d(input, module.weight, module.bias, module.stride,
                         module.padding, module.dilation, module.groups)
            s = R / (z + eps)
            c = F.conv_transpose2d(s, module.weight, None, module.stride,
                                   module.padding, module.dilation, module.groups)
            return input * c

        elif isinstance(module, nn.Linear):
            z = module(input)
            s = R / (z + eps)
            c = torch.matmul(s, module.weight)
            return input * c

        elif isinstance(module, nn.ReLU):
            return R

        elif isinstance(module, nn.BatchNorm2d):
            return R

        else:
            return R

    def run_lrp(self, input_tensor, target_class):
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        R = one_hot.clone().to(device)

        modules = list(self.model.modules())
        modules = [m for m in modules if isinstance(m, (nn.Conv2d, nn.ReLU, nn.Linear, nn.BatchNorm2d))]
        modules = modules[::-1]

        x = input_tensor
        for layer in modules:
            try:
                R = self.relprop(R, layer, x, None)
            except:
                pass

        relevance_map = R[0].sum(dim=0).detach().cpu().numpy()
        relevance_map -= relevance_map.min()
        relevance_map /= (relevance_map.max() + 1e-8)
        return relevance_map


# ============================================================
# Dataset for mask-based sarcomere classification
# ============================================================
# Each input image is first converted into a sarcomere segmentation mask.
# The mask is expanded to three channels and used as EfficientNet input.
# KR:
# 각 입력 이미지는 먼저 sarcomere segmentation mask로 변환됩니다.
# 생성된 mask는 3-channel image로 확장된 뒤 EfficientNet 입력으로 사용됩니다.
# ============================================================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, mode="Train", transform=None):
        self.root_dir = root_dir
        self.mode = mode     # ⭐⭐⭐ 반드시 추가
        self.transform = transform

        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAME)}

        self.samples = []


        for cls_name in CLASS_NAME:
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".tif")):
                    img_path = os.path.join(class_dir, fname)

                    # ⭐ 여기서 바로 매핑!
                    label = self.class_to_idx[cls_name]

                    self.samples.append((img_path, label))


        print(f"[Dataset:{mode}] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = Path(img_path) 

        cls_name = CLASS_NAME[label]    # label → 클래스 문자열 변환
        cache_dir = SAVE_MASK_DIR / self.mode / cls_name

        cache_dir.mkdir(parents=True, exist_ok=True)

        mask_path = cache_dir / f"{img_path.stem}_mask.png"

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = segment_sarcomere(img_path)
            cv2.imwrite(str(mask_path), mask)

        mask_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
        img = Image.fromarray(mask_rgb)

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# 🔧 Image Transform
# ============================================================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ============================================================
# 🔧 DataLoader
# ============================================================
if TRAIN_MODE == "on":
    train_loader = DataLoader(
        CustomImageDataset(TRAIN_DIR, mode="Train", transform=train_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        CustomImageDataset(VAL_DIR, mode="Val", transform=eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

else:
    test_loader = DataLoader(
        CustomImageDataset(TEST_DIR, mode="Test", transform=eval_tf),
        batch_size=1, shuffle=False, num_workers=0
    )


# ============================================================
# 🔧 Model (EfficientNet)
# ============================================================
model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=NUM_CLASSES)

# ============================================================
# 🔄 Resume Training (KEEP_MODE)
# ============================================================
resume_epoch = 0
resume_optimizer_state = None

if KEEP_MODE.lower() == "on" and TRAIN_MODE == "on":
    ckpts = sorted(glob.glob(str(CHECKPOINT_DIR / "epoch_*.pth")))
    if len(ckpts) > 0:
        latest_ckpt = ckpts[-1]
        resume_epoch = int(Path(latest_ckpt).stem.split("_")[-1])

        print(f"\n🔄 KEEP_MODE=ON → Resuming from epoch {resume_epoch}")
        print(f"   Loading: {latest_ckpt}")

        model.load_state_dict(torch.load(latest_ckpt, map_location=device))

        opt_state_path = CHECKPOINT_DIR / f"optimizer_{resume_epoch:03d}.pth"
        if opt_state_path.exists():
            resume_optimizer_state = torch.load(opt_state_path, map_location=device)
            print("   ✓ Optimizer state loaded")
        else:
            print("   ⚠ Optimizer state NOT found")
    else:
        print("⚠ KEEP_MODE=ON 이지만 checkpoint 없음 → 새로 시작")
else:
    if TRAIN_MODE == "on":
        print("KEEP_MODE=OFF → Training from scratch")

model = model.to(device)


# ============================================================
# 🔧 Train
# ============================================================
def train_model(start_epoch=1, resume_optimizer_state=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    writer = SummaryWriter(log_dir=str(TB_LOG_DIR), purge_step=start_epoch - 1)

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("   ✓ Optimizer state restored for resumed training")

    best_acc = 0.0

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ----------------- Validation -----------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds   = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0

        print(f"[Epoch {epoch}] Loss={avg_train_loss:.4f} | Val Acc={val_acc:.4f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # ---- Model checkpoint 저장 ----
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # ---- Optimizer checkpoint 저장 ----
        opt_state_path = CHECKPOINT_DIR / f"optimizer_{epoch:03d}.pth"
        torch.save(optimizer.state_dict(), opt_state_path)

        # ---- Best 모델 저장 ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_CKPT)
            print("[INFO] Best model updated.")

    print("Training Completed.")


# ============================================================
# 🔧 Single checkpoint evaluation
# ============================================================
def evaluate_single_checkpoint(checkpoint, loader):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    if total == 0:
        return 0.0

    return correct / total * 100.0


# ============================================================
# 🔧 Evaluate all checkpoints → Excel
# ============================================================
def evaluate_all_checkpoints():
    ckpts = sorted(glob.glob(str(CHECKPOINT_DIR / "epoch_*.pth")))
    results = []

    if not ckpts:
        print(f"[WARN] No checkpoint found in: {CHECKPOINT_DIR}")
        return None

    print("\n[ Evaluating All Checkpoints ]\n")

    for ckpt in ckpts:
        acc = evaluate_single_checkpoint(ckpt, test_loader)
        print(f"{os.path.basename(ckpt)} → {acc:.2f}%")
        results.append([os.path.basename(ckpt), acc])

    df = pd.DataFrame(results, columns=["checkpoint", "accuracy"])
    out_xlsx = ROOT / f"{EXCEL_NAME}.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"\nSaved → {out_xlsx}")

    return df


# ============================================================
# Grad-CAM
# ============================================================
# Grad-CAM is computed using the same segmentation-mask input used for classification.
# KR: Grad-CAM도 classification과 동일하게 segmentation mask 입력을 기준으로 계산합니다.
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model  = model
        self.target_layer = target_layer
        self.grad   = None
        self.act    = None

        self.target_layer.register_forward_hook(self._save_act)
        self.target_layer.register_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.act = o.detach()

    def _save_grad(self, m, gi, go):
        self.grad = go[0].detach()

    def generate(self, input_tensor, class_idx):
        out = self.model(input_tensor)               # [1, num_classes]
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1

        self.model.zero_grad()
        out.backward(gradient=one_hot, retain_graph=True)

        weights = self.grad.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.act).sum(dim=1).squeeze()   # [H, W]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-9)

        return cam.cpu().numpy()


# ============================================================
# 🔥 Grad-CAM++
# ============================================================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.grad = None
        self.act  = None

        self.target_layer.register_forward_hook(self._save_act)
        self.target_layer.register_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.act = o.detach()

    def _save_grad(self, m, gi, go):
        self.grad = go[0].detach()

    def generate(self, input_tensor, class_idx):
        out = self.model(input_tensor)

        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1

        self.model.zero_grad()
        out.backward(gradient=one_hot, retain_graph=True)

        grad = self.grad
        act  = self.act
        B, C, H, W = act.shape

        grad_2 = grad * grad
        grad_3 = grad_2 * grad

        denom = 2 * grad_2 + (act * grad_3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grad_2 / denom

        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).squeeze()

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-9)

        return cam.cpu().numpy()


def run_gradcam_single(checkpoint, img_path, save_path="gradcam_output.png"):
    print("Loading checkpoint:", checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    target_layer = model._blocks[-1]._project_conv

    cam_grad   = GradCAM(model, target_layer)
    cam_gradpp = GradCAMPlusPlus(model, target_layer)

    # 🔥 원본 → sarcomere 세그멘테이션 마스크 생성 후, 그걸 입력/overlay로 사용
    img_path = Path(img_path)
    mask = segment_sarcomere(img_path)  # (H, W)
    img_np_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_np_rgb)

    input_tensor = eval_tf(img).unsqueeze(0).to(device)

    out = model(input_tensor)
    pred = out.argmax(dim=1).item()
    prob = torch.softmax(out, dim=1)[0][pred].item()

    # -------------------------------
    # 🔥 GradCAM
    # -------------------------------
    cam_g = cam_grad.generate(input_tensor, pred)
    cam_g = cv2.resize(cam_g, (img_np_rgb.shape[1], img_np_rgb.shape[0]))
    heat_g = cv2.applyColorMap((cam_g * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_g = cv2.cvtColor(heat_g, cv2.COLOR_BGR2RGB)
    overlay_g = (0.4 * img_np_rgb + 0.6 * heat_g).astype(np.uint8)

    # -------------------------------
    # 🔥 GradCAM++
    # -------------------------------
    cam_gp = cam_gradpp.generate(input_tensor, pred)
    cam_gp = cv2.resize(cam_gp, (img_np_rgb.shape[1], img_np_rgb.shape[0]))
    heat_gp = cv2.applyColorMap((cam_gp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_gp = cv2.cvtColor(heat_gp, cv2.COLOR_BGR2RGB)
    overlay_gp = (0.4 * img_np_rgb + 0.6 * heat_gp).astype(np.uint8)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    ax[0, 0].imshow(img_np_rgb); ax[0, 0].set_title("Segmentation Mask (Sarcomere)")
    ax[0, 1].imshow(heat_g);     ax[0, 1].set_title("Grad-CAM")
    ax[0, 2].imshow(overlay_g);  ax[0, 2].set_title(f"Grad-CAM Overlay (pred={pred})")

    ax[1, 0].imshow(img_np_rgb); ax[1, 0].set_title("Segmentation Mask (Sarcomere)")
    ax[1, 1].imshow(heat_gp);    ax[1, 1].set_title("Grad-CAM++")
    ax[1, 2].imshow(overlay_gp); ax[1, 2].set_title(f"Grad-CAM++ Overlay (pred={pred})")

    for r in ax:
        for a in r:
            a.axis("off")

    plt.suptitle(
    f"Class={CLASS_NAME[pred]} ({pred}), Prob={prob:.3f}", 
    fontsize=16)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def run_gradcam_batch(checkpoint_path, test_root=TEST_DIR, save_root="gradcam_results"):

    print("\n[INFO] Grad-CAM batch processing started.")
    print(f" - checkpoint: {checkpoint_path}")
    print(f" - test_root:  {test_root}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    target_layer = model._blocks[-1]._project_conv

    cam_grad   = GradCAM(model, target_layer)
    cam_gradpp = GradCAMPlusPlus(model, target_layer)

    save_root = ROOT / save_root / Path(checkpoint_path).stem
    save_root.mkdir(parents=True, exist_ok=True)

    classes = CLASS_NAME

    for cls in classes:
        cls_folder = Path(test_root) / cls
        if not cls_folder.exists():
            continue

        save_cls_dir = save_root / cls
        save_cls_dir.mkdir(parents=True, exist_ok=True)

        images = list(cls_folder.glob("*.*"))
        print(f"\n[Class {cls}] 총 {len(images)}개 처리")

        for img_path in images:
            img_path = Path(img_path)

            # 원본 → seg mask
            mask = segment_sarcomere(img_path)
            # 🔥 원본 이미지 읽기 (Grad-CAM overlay용)
            orig = cv2.imread(str(img_path), cv2.IMREAD_COLOR)   # 원본 컬러 그대로
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)     # 시각화용 변환
            img_np_rgb = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
            img = Image.fromarray(img_np_rgb)

            input_tensor = eval_tf(img).unsqueeze(0).to(device)

            out = model(input_tensor)
            pred = out.argmax(dim=1).item()
            prob = torch.softmax(out, dim=1)[0][pred].item()

            cam_g = cam_grad.generate(input_tensor, pred)
            cam_g = cv2.resize(cam_g, (img_np_rgb.shape[1], img_np_rgb.shape[0]))
            heat_g = cv2.applyColorMap((cam_g * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_g = cv2.cvtColor(heat_g, cv2.COLOR_BGR2RGB)
            overlay_g_mask = (0.4 * img_np_rgb + 0.6 * heat_g).astype(np.uint8)
            overlay_g_orig = (0.4 * orig_rgb + 0.6 * heat_g).astype(np.uint8)

            cam_gp = cam_gradpp.generate(input_tensor, pred)
            cam_gp = cv2.resize(cam_gp, (img_np_rgb.shape[1], img_np_rgb.shape[0]))
            heat_gp = cv2.applyColorMap((cam_gp * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_gp = cv2.cvtColor(heat_gp, cv2.COLOR_BGR2RGB)
            overlay_gp_orig = (0.4 * orig_rgb + 0.6 * heat_gp).astype(np.uint8)
            overlay_gp_mask = (0.4 * img_np_rgb + 0.6 * heat_gp).astype(np.uint8)


            base = img_path.stem
            class_name = CLASS_NAME[pred]
            out_path = save_cls_dir / f"{base}_pred={class_name}_prob={prob:.3f}.png"

            # ===============================
            # 🔥 Nature-style Grad-CAM Figure (Revised)
            # ===============================
            fig = plt.figure(figsize=(22, 12))  # 논문형 크게
            gs = fig.add_gridspec(2, 4, wspace=0.2, hspace=0.2)

            # -------- Row 1 --------
            ax1 = fig.add_subplot(gs[0,0]); ax1.imshow(orig_rgb); ax1.set_title("Original", fontsize=20, fontweight='bold')
            ax2 = fig.add_subplot(gs[0,1]); ax2.imshow(img_np_rgb); ax2.set_title("Segmentation Mask", fontsize=20, fontweight='bold')
            ax3 = fig.add_subplot(gs[0,2]); ax3.imshow(heat_g); ax3.set_title("Grad-CAM Heatmap", fontsize=20, fontweight='bold')
            ax4 = fig.add_subplot(gs[0,3]); ax4.imshow(heat_gp); ax4.set_title("Grad-CAM++ Heatmap", fontsize=20, fontweight='bold')

            # -------- Row 2 --------
            ax5 = fig.add_subplot(gs[1,0]); ax5.imshow(overlay_g_orig); ax5.set_title("Grad-CAM on Original", fontsize=20, fontweight='bold')
            ax6 = fig.add_subplot(gs[1,1]); ax6.imshow(overlay_gp_orig); ax6.set_title("Grad-CAM++ on Original", fontsize=20, fontweight='bold')
            ax7 = fig.add_subplot(gs[1,2]); ax7.imshow(overlay_g_mask); ax7.set_title("Grad-CAM on Mask", fontsize=20, fontweight='bold')
            ax8 = fig.add_subplot(gs[1,3]); ax8.imshow(overlay_gp_mask); ax8.set_title("Grad-CAM++ on Mask", fontsize=20, fontweight='bold')

            # Hide axes
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
                ax.axis("off")

            # Main Title
            cls_name = CLASS_NAME[pred] if pred < len(CLASS_NAME) else f"Class_{pred}"

            plt.suptitle(
                f"Pred={cls_name} (id={pred}), Prob={prob:.3f}",
                fontsize=26, fontweight='bold',
                y=0.98
)


            plt.savefig(out_path, dpi=400, bbox_inches="tight")
            plt.close()

            # ===============================
            # 🔥 Grad-CAM++ on Mask 단독 저장
            # ===============================
            single_out_path = save_cls_dir / f"{base}_GradCAMpp_on_mask.png"

            plt.figure(figsize=(6, 6))
            plt.imshow(overlay_gp_mask)
            plt.axis("off")
            # plt.title("Grad-CAM++ on Mask", fontsize=18, fontweight="bold")
            plt.savefig(single_out_path, dpi=400, bbox_inches="tight", pad_inches=0)
            plt.close()



            torch.cuda.empty_cache()

        print(f"✓ Saved: {save_cls_dir}")

    print("\n🎉 Grad-CAM + Grad-CAM++ Batch Completed!")

def plot_confusion_matrix_test(checkpoint, loader, save_path):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix")
    print(cm)

    # normalize (%)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # plot
    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAME,
        yticklabels=CLASS_NAME,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size":18, "weight":"bold"}
    )

    plt.xlabel("Predicted label", fontsize=18, fontweight="bold")
    plt.ylabel("True label", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

    # classification report
    print(classification_report(y_true, y_pred, target_names=CLASS_NAME))
# ============================================================
# Folder-level classification mode
# ============================================================
# Classifies all images in a selected folder into Pre, Nascent, or Mature.
# Each image is first converted into a sarcomere segmentation mask before classification.
# KR:
# 선택한 folder 내 모든 이미지를 Pre, Nascent, Mature로 분류합니다.
# 각 이미지는 classification 전에 먼저 sarcomere segmentation mask로 변환됩니다.
# ============================================================
def run_classification_on_folder(
        model,
        checkpoint_path,
        input_dir,
        output_dir="classification_result",
        copy_mode=True,
        device="cuda"
    ):

    print("\n[Classification Mode] Test Mode와 동일하게 분류 시작")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # CLASS_NAME 기반 폴더 생성
    class_folders = {i: (output_dir / CLASS_NAME[i]) for i in range(len(CLASS_NAME))}

    # 실제 폴더 생성
    for d in class_folders.values():
        d.mkdir(parents=True, exist_ok=True)


    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    images = sorted(
        [p for p in input_dir.glob("*") if p.suffix.lower() in valid_exts],
        key=lambda x: x.name
    )

    print(f"📌 분류 대상 이미지 개수: {len(images)}")

    tf = eval_tf

    for img_path in tqdm(images):

        img_path = Path(img_path)
        mask = segment_sarcomere(img_path)
        np_img = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
        img_rgb = Image.fromarray(np_img)

        input_tensor = tf(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_tensor)
            pred = out.argmax(dim=1).item()

        print(f"[CHECK] {img_path.name} → pred={pred}")

        dst = class_folders[pred] / img_path.name
        if copy_mode:
            shutil.copy(img_path, dst)
        else:
            shutil.move(img_path, dst)

    print(f"\n[INFO] Classification results saved to: {output_dir}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    if TRAIN_MODE.lower() == "on":
        train_model(
            start_epoch = resume_epoch + 1 if (KEEP_MODE=="on") else 1,
            resume_optimizer_state = resume_optimizer_state
        )

    elif TEST_MODE.lower() == "on":
        if not BEST_CKPT.exists():
            print(f"⚠️ BEST_CKPT not found: {BEST_CKPT}")
        else:
            print("\n[ Best Model Test Accuracy ]")
            acc = evaluate_single_checkpoint(BEST_CKPT, test_loader)
            print(f"Best Model Accuracy = {acc:.2f}%")

        print("\n[ Evaluate All Checkpoints ]")
        df = evaluate_all_checkpoints()
        if df is not None:
            print(df)
            
    if MATRIX.lower() == "on":
        ckpt_name = Path(BEST_CKPT).stem
        cm_save = CHECKPOINT_DIR / f"confusion_matrix_{ckpt_name}.png"
        plot_confusion_matrix_test(BEST_CKPT, test_loader, cm_save)

    elif GRAD_CAM_MODE == "single":
        example_ckpt = str(CLASSIFY_EPOCH)
        example_img  = str(CLASSIFY_GRAD_INPUT_DIR)  # 필요하면 수정
        run_gradcam_single(example_ckpt, example_img,
                           save_path="gradcam_single_example.png")

    elif GRAD_CAM_MODE == "batch":
        example_ckpt = str(CLASSIFY_EPOCH)
        run_gradcam_batch(example_ckpt, test_root=CLASSIFY_GRAD_INPUT_DIR, save_root=fr"{CLASSIFY_GRAD_INPUT_DIR}\{GRAD_CAM_FOLDER_NAME}")

    elif CLASSIFICATION_MODE == "on":
        ckpt = str(CLASSIFY_EPOCH)
        input_dir = CLASSIFY_GRAD_INPUT_DIR

        print(f"\n⚡ Classification Mode ON")
        print(f"   사용 모델: {ckpt}")
        print(f"   입력 폴더: {input_dir}")

        run_classification_on_folder(
            model,
            checkpoint_path=ckpt,
            input_dir=input_dir,
            output_dir = CLASSIFY_GRAD_INPUT_DIR / "classification_result",
            copy_mode=False,
            device=device
        )

    else:
        print("END")
