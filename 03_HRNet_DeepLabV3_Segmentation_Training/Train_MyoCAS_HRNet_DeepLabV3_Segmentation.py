"""
Train_MyoCAS_DualNet_Segmentation.py

Training, testing, and prediction script for the MyoCAS DualNet segmentation model.

This script trains a hybrid segmentation model that combines:
    1. CustomHRNet
    2. DeepLabV3-ResNet101
    3. A fusion head for final binary segmentation

The model is designed for phase-contrast image-based myotube/sarcomere-related
segmentation using paired phase images and binary mask images.

KR:
MyoCAS DualNet segmentation model의 학습, 평가, 예측을 수행하는 script입니다.

이 script는 다음 두 모델 출력을 결합한 hybrid segmentation model을 사용합니다.
    1. CustomHRNet
    2. DeepLabV3-ResNet101
    3. 최종 binary segmentation을 위한 fusion head

Phase-contrast image와 binary mask image pair를 이용해 myotube/sarcomere 관련
segmentation을 수행하기 위한 코드입니다.
"""

import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================
# Main execution mode.
# "train"   : train the DualNet segmentation model.
# "test"    : evaluate all saved checkpoints on the test dataset.
# "predict" : apply one trained checkpoint to unlabeled phase images.
#
# KR:
# 실행 모드입니다.
# "train"   : DualNet segmentation model을 학습합니다.
# "test"    : 저장된 모든 checkpoint를 test dataset에서 평가합니다.
# "predict" : 하나의 학습된 checkpoint를 사용해 phase image를 예측합니다.
MODE = "train"

# CUDA device ID.
# KR: 사용할 CUDA device 번호입니다.
DEVICE_ID = "1"

# Training parameters.
# KR: 학습 설정값입니다.
batch_size = 1
EPOCH = 50

# Root dataset directory.
# Expected structure:
#   base_dir/
#       train/
#           phase/
#           mask/
#       val/
#           phase/
#           mask/
#       test/
#           phase/
#           mask/
#
# KR:
# Dataset root directory입니다.
# 아래와 같은 구조를 권장합니다.
#   base_dir/
#       train/phase, train/mask
#       val/phase, val/mask
#       test/phase, test/mask
base_dir = r"C:\Users\Hyeon Jun\Desktop\test2\test\github test\se\Train_folder\Sarcomere"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Subfolder names for phase images and masks.
# KR: phase image와 mask image가 들어 있는 하위 folder 이름입니다.
Train_phase_dir = "actinin"
Train_mask_dir = "mask"
Test_phase_dir = "actinin"
Test_mask_dir = "mask"

# Output directories.
# KR: checkpoint, test prediction, single-checkpoint prediction 결과 저장 경로입니다.
ckpt_dir = os.path.join(base_dir, "ckpt_dualnet")
pred_dir = os.path.join(test_dir, "predictions_dualnet")
gt_mask_dir = os.path.join(test_dir, Test_mask_dir)

# Single-checkpoint prediction settings.
# KR: predict mode에서 사용할 checkpoint 및 저장 경로입니다.
PREDICT_CKPT_PATH = os.path.join(ckpt_dir, "dualnet_epoch_050.pth")
predict_input_dir = test_dir
predict_phase_dir = Test_phase_dir
predict_save_dir = os.path.join(test_dir, "predict_one_checkpoint")

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(predict_save_dir, exist_ok=True)

device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# ============================================================
# CustomHRNet backbone
# ============================================================
# This section defines a compact custom HRNet-like backbone.
# It extracts multi-scale features while maintaining relatively high spatial resolution.
#
# KR:
# 이 section은 compact custom HRNet-like backbone을 정의합니다.
# 비교적 높은 spatial resolution을 유지하면서 multi-scale feature를 추출합니다.
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class CustomHRNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Stem block.
        # KR: 입력 image를 초기 feature map으로 변환하는 stem block입니다.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)

        # Stage 2 branches.
        # KR: 서로 다른 resolution의 feature branch를 생성합니다.
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

        # Stage 3 branches.
        # KR: multi-scale feature를 추가로 확장합니다.
        self.stage3_branch1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.stage3_branch2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)

        self.stage3_branch3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Feature fusion and classifier.
        # KR: multi-scale feature를 concatenate한 뒤 최종 class map을 출력합니다.
        self.fuse_conv = nn.Conv2d(32 + 64 + 128, 256, 1)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
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

        b2_3_up = F.interpolate(
            b2_3,
            size=b1_3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        b3_3_up = F.interpolate(
            b3_3,
            size=b1_3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        feat = torch.cat([b1_3, b2_3_up, b3_3_up], dim=1)
        feat = F.relu(self.fuse_conv(feat))
        feat = F.interpolate(
            feat,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        return self.classifier(feat)


# ============================================================
# Training and validation dataset
# ============================================================
# This dataset loads paired phase images and binary mask images.
# Mask files can have either the same base name or the "_mask" suffix.
#
# KR:
# 이 dataset은 phase image와 binary mask image pair를 불러옵니다.
# Mask file은 같은 base name을 가지거나 "_mask" suffix를 가질 수 있습니다.
# ============================================================

class PhaseToBlackSegDataset(Dataset):
    def __init__(self, base_dir, phase_dir, black_dir, augment=False):
        self.phase_dir = os.path.join(base_dir, phase_dir)
        self.black_dir = os.path.join(base_dir, black_dir)

        self.phase_paths = []

        for f in os.listdir(self.phase_dir):
            if not (f.endswith(".tif") or f.endswith(".png")):
                continue

            name_root, _ = os.path.splitext(f)

            mask_candidates = [
                f"{name_root}_mask.tif",
                f"{name_root}_mask.png",
                f"{name_root}.tif",
                f"{name_root}.png",
            ]

            if any(os.path.exists(os.path.join(self.black_dir, m)) for m in mask_candidates):
                self.phase_paths.append(f)

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ])

        self.augment = augment

    def __len__(self):
        return len(self.phase_paths)

    def __getitem__(self, idx):
        filename = self.phase_paths[idx]
        name_root, _ = os.path.splitext(filename)

        phase_path = os.path.join(self.phase_dir, filename)

        mask_candidates = [
            f"{name_root}_mask.tif",
            f"{name_root}_mask.png",
            f"{name_root}.tif",
            f"{name_root}.png",
        ]

        black_path = next(
            (
                os.path.join(self.black_dir, cand)
                for cand in mask_candidates
                if os.path.exists(os.path.join(self.black_dir, cand))
            ),
            None
        )

        if black_path is None:
            raise FileNotFoundError(f"Mask not found for: {filename}")

        phase = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
        black = cv2.imread(black_path, cv2.IMREAD_GRAYSCALE)

        mask = (black >= 128).astype(np.uint8)

        augmented = self.transform(image=phase, mask=mask)

        image_tensor = augmented["image"].repeat(3, 1, 1)
        mask_tensor = augmented["mask"].long()

        return image_tensor, mask_tensor, filename


# ============================================================
# Inference dataset
# ============================================================
# This dataset loads phase images only and is used for prediction or checkpoint evaluation.
#
# KR:
# 이 dataset은 phase image만 불러오며 prediction 또는 checkpoint evaluation에 사용됩니다.
# ============================================================

class InferenceDataset(Dataset):
    def __init__(self, base_dir, phase_dir):
        self.phase_dir = os.path.join(base_dir, phase_dir)

        self.phase_paths = sorted([
            f for f in os.listdir(self.phase_dir)
            if f.endswith(".png") or f.endswith(".tif")
        ])

        # Keep this normalization consistent with the checkpoint training setting.
        # KR: 이 normalization 값은 checkpoint 학습 조건과 일치해야 합니다.
        self.transform = A.Compose([
            A.Normalize(mean=3, std=3),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.phase_paths)

    def __getitem__(self, idx):
        filename = self.phase_paths[idx]
        phase_path = os.path.join(self.phase_dir, filename)

        img = cv2.imread(phase_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = self.transform(image=img)["image"].repeat(3, 1, 1)

        return img_tensor, filename


# ============================================================
# BatchNorm to GroupNorm conversion
# ============================================================
# This function replaces BatchNorm2d layers with GroupNorm layers.
# This is useful when training with very small batch sizes.
#
# KR:
# BatchNorm2d layer를 GroupNorm layer로 변환합니다.
# Batch size가 매우 작은 경우 training stability를 높이는 데 유용합니다.
# ============================================================

def convert_batchnorm_to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(32, child.num_features))
        else:
            convert_batchnorm_to_groupnorm(child)


# ============================================================
# DualNet model
# ============================================================
# DualNet combines the output of CustomHRNet and DeepLabV3.
# The two segmentation logits are concatenated and passed through a fusion head.
#
# KR:
# DualNet은 CustomHRNet과 DeepLabV3의 출력을 결합합니다.
# 두 segmentation logit을 concatenate한 뒤 fusion head를 통해 최종 출력을 생성합니다.
# ============================================================

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

        hr = self.hrnet(x)
        dl = self.deeplabv3(x)["out"]

        if hr.shape[-2:] != dl.shape[-2:]:
            dl = F.interpolate(
                dl,
                size=hr.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        out = self.fuse(torch.cat([hr, dl], dim=1))
        out = F.interpolate(
            out,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        return out


# ============================================================
# Loss function
# ============================================================
# Combined Dice, IoU, and CrossEntropy loss for binary segmentation.
#
# KR:
# Binary segmentation을 위한 Dice, IoU, CrossEntropy 결합 loss입니다.
# ============================================================

class DiceIoUCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, iou_weight=0.4, ce_weight=0.1, weight=None):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)

        smooth = 1e-6

        inputs_soft = F.softmax(inputs, dim=1)[:, 1]
        targets_bin = (targets == 1).float()

        intersection = (inputs_soft * targets_bin).sum()

        dice_loss = 1 - (
            (2 * intersection + smooth) /
            (inputs_soft.sum() + targets_bin.sum() + smooth)
        )

        union = inputs_soft.sum() + targets_bin.sum() - intersection

        iou_loss = 1 - (
            (intersection + smooth) /
            (union + smooth)
        )

        return (
            self.dice_weight * dice_loss +
            self.iou_weight * iou_loss +
            self.ce_weight * ce_loss
        )


# ============================================================
# Training function
# ============================================================
# Runs one training epoch, evaluates validation loss, updates scheduler,
# and saves a checkpoint for the current epoch.
#
# KR:
# 한 epoch의 training을 수행하고 validation loss를 계산한 뒤,
# scheduler를 업데이트하고 현재 epoch checkpoint를 저장합니다.
# ============================================================

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epoch, ckpt_dir):
    start_time = time.time()

    model.train()
    total_loss = 0

    for x, y, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch:03d}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc=f"[Val] Epoch {epoch:03d}"):
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item()

    scheduler.step(val_loss)

    ckpt_path = os.path.join(ckpt_dir, f"dualnet_epoch_{epoch:03d}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        ckpt_path
    )

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch} | "
        f"Train: {total_loss:.4f} | "
        f"Val: {val_loss:.4f} | "
        f"{elapsed:.2f}s"
    )

    return total_loss, val_loss, elapsed


# ============================================================
# Evaluation metrics
# ============================================================
# Dice coefficient and IoU score for binary segmentation evaluation.
#
# KR:
# Binary segmentation 평가를 위한 Dice coefficient와 IoU score입니다.
# ============================================================

def dice_coefficient(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()

    return 2 * intersection / (pred.sum() + target.sum() + 1e-7)


def iou_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    return intersection / (union + 1e-7)


# ============================================================
# Test and evaluation
# ============================================================
# Loads one checkpoint, predicts masks for the test phase images,
# saves predicted masks, and evaluates Dice/IoU if GT masks are available.
#
# KR:
# 하나의 checkpoint를 불러와 test phase image의 mask를 예측하고,
# 예측 mask를 저장한 뒤 GT mask가 있으면 Dice/IoU를 계산합니다.
# ============================================================

def test_and_evaluate(model, test_loader, device, ckpt_path, save_dir, gt_mask_dir):
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for x, names in tqdm(test_loader, desc=f"[Test: {os.path.basename(ckpt_path)}]"):
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()

            for i, pred in enumerate(preds):
                name_root, _ = os.path.splitext(names[i])
                save_path = os.path.join(save_dir, f"{name_root}.png")

                cv2.imwrite(save_path, (pred * 255).astype(np.uint8))

                mask_candidates = [
                    os.path.join(gt_mask_dir, f"{name_root}_mask.png"),
                    os.path.join(gt_mask_dir, f"{name_root}_mask.tif"),
                    os.path.join(gt_mask_dir, f"{name_root}.png"),
                    os.path.join(gt_mask_dir, f"{name_root}.tif"),
                ]

                gt_path = next((p for p in mask_candidates if os.path.exists(p)), None)

                if gt_path:
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                    if gt is not None:
                        pred_bin = (pred >= 1).astype(np.uint8)
                        gt_bin = (gt >= 128).astype(np.uint8)

                        dice_scores.append(dice_coefficient(pred_bin, gt_bin))
                        iou_scores.append(iou_score(pred_bin, gt_bin))

    if dice_scores:
        dice = np.mean(dice_scores)
        iou = np.mean(iou_scores)

        print(
            f"Done: {os.path.basename(ckpt_path)} "
            f"-> Dice {dice:.4f}, IoU {iou:.4f}"
        )

        return dice, iou

    print("[WARN] No GT masks were found. Evaluation was skipped.")
    return 0, 0


# ============================================================
# Single-checkpoint prediction
# ============================================================
# Loads one trained checkpoint and saves predicted binary masks.
#
# KR:
# 하나의 학습된 checkpoint를 불러와 predicted binary mask를 저장합니다.
# ============================================================

def predict_one_checkpoint(model, infer_loader, device, ckpt_path, save_dir):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Predicting with checkpoint: {ckpt_path}")

    with torch.no_grad():
        for x, names in tqdm(infer_loader, desc=f"[Predict: {os.path.basename(ckpt_path)}]"):
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()

            for i, pred in enumerate(preds):
                name_root, _ = os.path.splitext(names[i])
                save_path = os.path.join(save_dir, f"{name_root}.png")

                cv2.imwrite(save_path, (pred * 255).astype(np.uint8))

    print(f"[INFO] Prediction saved to: {save_dir}")


# ============================================================
# Main execution
# ============================================================
# Builds the DualNet model and runs train, test, or predict mode.
#
# KR:
# DualNet model을 구성하고 train, test, predict mode 중 하나를 실행합니다.
# ============================================================

def main():
    hrnet = CustomHRNet(num_classes=2)
    deeplab = deeplabv3_resnet101(weights=None, num_classes=2)

    convert_batchnorm_to_groupnorm(hrnet)
    convert_batchnorm_to_groupnorm(deeplab)

    model = DualNet(hrnet, deeplab, num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=5
    )

    criterion = DiceIoUCELoss()

    if MODE.lower() == "train":
        train_loader = DataLoader(
            PhaseToBlackSegDataset(train_dir, Train_phase_dir, Train_mask_dir, augment=True),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            PhaseToBlackSegDataset(val_dir, Train_phase_dir, Train_mask_dir, augment=False),
            batch_size=batch_size,
            shuffle=False
        )

        last_epoch = 0

        existing = sorted([
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".pth")
        ])

        if existing:
            ckpt = torch.load(os.path.join(ckpt_dir, existing[-1]), map_location=device)

            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            last_epoch = ckpt["epoch"]

            print(f"[INFO] Resuming from checkpoint: {existing[-1]}")

        total_epochs = EPOCH
        epoch_times = []

        for epoch in range(last_epoch + 1, total_epochs + 1):
            train_loss, val_loss, elapsed = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                scheduler,
                device,
                epoch,
                ckpt_dir
            )

            epoch_times.append(elapsed)

            avg = sum(epoch_times) / len(epoch_times)
            eta = avg * (total_epochs - epoch)

            print(f"[INFO] Estimated time left: {int(eta // 60)}m {int(eta % 60)}s\n")

    elif MODE.lower() == "test":
        test_loader = DataLoader(
            InferenceDataset(test_dir, Test_phase_dir),
            batch_size=batch_size,
            shuffle=False
        )

        results = []

        for ckpt in sorted(os.listdir(ckpt_dir)):
            if ckpt.endswith(".pth"):
                ckpt_path = os.path.join(ckpt_dir, ckpt)
                save_dir = os.path.join(pred_dir, ckpt.replace(".pth", ""))

                dice, iou = test_and_evaluate(
                    model,
                    test_loader,
                    device,
                    ckpt_path,
                    save_dir,
                    gt_mask_dir
                )

                results.append((ckpt[:-4], dice, iou))

        results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        print("\nTop 10 best checkpoints:")

        for i, (name, dice, iou) in enumerate(results[:10], 1):
            print(f"{i:02d}. {name:25s} | Dice: {dice:.4f} | IoU: {iou:.4f}")

    elif MODE.lower() == "predict":
        infer_loader = DataLoader(
            InferenceDataset(predict_input_dir, predict_phase_dir),
            batch_size=batch_size,
            shuffle=False
        )

        predict_one_checkpoint(
            model=model,
            infer_loader=infer_loader,
            device=device,
            ckpt_path=PREDICT_CKPT_PATH,
            save_dir=predict_save_dir
        )

    else:
        raise ValueError("MODE must be one of: 'train', 'test', 'predict'")


if __name__ == "__main__":
    main()