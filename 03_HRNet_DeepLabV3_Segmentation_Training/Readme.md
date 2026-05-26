# HRNet + DeepLabV3 Segmentation Training

This module trains, tests, and applies a custom dual-branch segmentation model for MyoCAS.

The model combines:

```text
CustomHRNet
DeepLabV3-ResNet101
Fusion head
```

The model performs binary segmentation using paired α-actinin fluorescence images and binary mask images.

This module is used to train segmentation models for:

```text
1. Myotube area segmentation
2. Sarcomere segmentation
```

The trained myotube area segmentation model can be used for myotube area quantification from α-actinin-stained images.

The trained sarcomere segmentation model can be used in the downstream sarcomere maturation classification and Grad-CAM pipeline.

---

## Folder structure

Recommended folder structure:

```text
03_HRNet_DeepLabV3_Segmentation_Training/
├── README.md
├── Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
└── Train_folder/
    ├── Myotube_area/
    │   ├── Run_folder/
    │   │   ├── Alpha_Actinin_Images/
    │   │   │   ├── example_001.png
    │   │   │   ├── example_002.png
    │   │   │   └── ...
    │   │   ├── Segmentation_Results/
    │   │   │   ├── example_001_mask.png
    │   │   │   ├── example_002_mask.png
    │   │   │   └── ...
    │   │   └── Stitching_Results/
    │   │       ├── stitched_example_001.png
    │   │       └── ...
    │   └── Train_folder/
    │       ├── train/
    │       │   ├── actinin/
    │       │   │   ├── example_001.png
    │       │   │   └── example_002.png
    │       │   └── mask/
    │       │       ├── example_001_mask.png
    │       │       └── example_002_mask.png
    │       ├── val/
    │       │   ├── actinin/
    │       │   │   ├── example_003.png
    │       │   │   └── example_004.png
    │       │   └── mask/
    │       │       ├── example_003_mask.png
    │       │       └── example_004_mask.png
    │       └── test/
    │           ├── actinin/
    │           │   ├── example_005.png
    │           │   └── example_006.png
    │           └── mask/
    │               ├── example_005_mask.png
    │               └── example_006_mask.png
    └── Sarcomere/
        ├── train/
        │   ├── actinin/
        │   │   ├── example_001.png
        │   │   └── example_002.png
        │   └── mask/
        │       ├── example_001_mask.png
        │       └── example_002_mask.png
        ├── val/
        │   ├── actinin/
        │   │   ├── example_003.png
        │   │   └── example_004.png
        │   └── mask/
        │       ├── example_003_mask.png
        │       └── example_004_mask.png
        └── test/
            ├── actinin/
            │   ├── example_005.png
            │   └── example_006.png
            └── mask/
                ├── example_005_mask.png
                └── example_006_mask.png
```

For this repository, all example input images and masks are provided as PNG files.

Important: use the following folder names expected by the script:

```text
train
val
test
actinin
mask
```

---

## Main script

```text
Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
```

This script supports three execution modes:

```python
MODE = "train"    # Train the segmentation model
MODE = "test"     # Evaluate all saved checkpoints on the test dataset
MODE = "predict"  # Apply one checkpoint to input images
```

---

## Model overview

The segmentation model is a custom dual-branch model.

It combines:

```text
CustomHRNet branch
DeepLabV3-ResNet101 branch
Fusion head
```

The model outputs a binary segmentation mask.

```text
Input α-actinin image
↓
CustomHRNet branch
+
DeepLabV3 branch
↓
Feature/logit fusion
↓
Binary segmentation output
```

The output classes are:

```text
0 : background
1 : target structure
```

For `Myotube_area`, the target structure is the myotube area.

For `Sarcomere`, the target structure is the sarcomere-positive region.

---

## Dataset structure for training

For training, validation, and testing, each dataset should contain paired PNG images:

```text
train/
├── actinin/
└── mask/

val/
├── actinin/
└── mask/

test/
├── actinin/
└── mask/
```

The `actinin` folder contains input α-actinin fluorescence images.

The `mask` folder contains binary ground-truth mask images.

Example:

```text
train/
├── actinin/
│   ├── example_001.png
│   └── example_002.png
└── mask/
    ├── example_001_mask.png
    └── example_002_mask.png
```

Mask files should use either the same base name or the `_mask` suffix.

Examples:

```text
example_001.png
example_001_mask.png
```

or

```text
example_001.png
example_001.png
```

The script searches for mask files using the image base name.

Mask candidate names searched by the script:

```text
{name_root}_mask.tif
{name_root}_mask.png
{name_root}.tif
{name_root}.png
```

Although the script can search for both TIFF and PNG masks, the example files in this repository are provided as PNG files.

---

## Training myotube area segmentation

To train the model for myotube area segmentation, set `base_dir` to the myotube area training folder.

Example:

```python
base_dir = r"path/to/Train_folder/Myotube_area/Train_folder"
```

Expected structure:

```text
Myotube_area/
└── Train_folder/
    ├── train/
    │   ├── actinin/
    │   │   ├── example_001.png
    │   │   └── example_002.png
    │   └── mask/
    │       ├── example_001_mask.png
    │       └── example_002_mask.png
    ├── val/
    │   ├── actinin/
    │   │   ├── example_003.png
    │   │   └── example_004.png
    │   └── mask/
    │       ├── example_003_mask.png
    │       └── example_004_mask.png
    └── test/
        ├── actinin/
        │   ├── example_005.png
        │   └── example_006.png
        └── mask/
            ├── example_005_mask.png
            └── example_006_mask.png
```

Then set:

```python
MODE = "train"
```

and run:

```bash
python Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
```

The resulting checkpoint can be used for myotube area segmentation and area quantification from α-actinin-stained images.

---

## Training sarcomere segmentation

To train the model for sarcomere segmentation, set `base_dir` to the sarcomere training folder.

Example:

```python
base_dir = r"path/to/Train_folder/Sarcomere"
```

Expected structure:

```text
Sarcomere/
├── train/
│   ├── actinin/
│   │   ├── example_001.png
│   │   └── example_002.png
│   └── mask/
│       ├── example_001_mask.png
│       └── example_002_mask.png
├── val/
│   ├── actinin/
│   │   ├── example_003.png
│   │   └── example_004.png
│   └── mask/
│       ├── example_003_mask.png
│       └── example_004_mask.png
└── test/
    ├── actinin/
    │   ├── example_005.png
    │   └── example_006.png
    └── mask/
        ├── example_005_mask.png
        └── example_006_mask.png
```

Then set:

```python
MODE = "train"
```

and run:

```bash
python Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
```

The resulting sarcomere segmentation checkpoint can be used in the downstream sarcomere maturation classification and Grad-CAM pipeline.

---

## Important settings

### Execution mode

```python
MODE = "train"
```

Available options:

```text
train   : train the segmentation model
test    : evaluate saved checkpoints
predict : generate masks using one selected checkpoint
```

### Dataset root directory

```python
base_dir = r"path/to/segmentation_dataset"
```

`base_dir` must directly contain:

```text
train/
val/
test/
```

For myotube area segmentation:

```python
base_dir = r"path/to/Train_folder/Myotube_area/Train_folder"
```

For sarcomere segmentation:

```python
base_dir = r"path/to/Train_folder/Sarcomere"
```

### Input and mask folder names

```python
Train_phase_dir = "actinin"
Train_mask_dir  = "mask"
Test_phase_dir  = "actinin"
Test_mask_dir   = "mask"
```

### Training parameters

```python
batch_size = 1
EPOCH = 50
```

### CUDA device

```python
DEVICE_ID = "1"
```

If CUDA is available, the script uses:

```python
cuda:{DEVICE_ID}
```

If CUDA is not available, the script automatically uses CPU.

---

## Loss function

The model is trained using a combined loss:

```text
Dice loss
IoU loss
CrossEntropy loss
```

The default weighting is:

```python
dice_weight = 0.5
iou_weight  = 0.4
ce_weight   = 0.1
```

This combined loss is designed for binary segmentation tasks where both overlap accuracy and pixel-wise classification are important.

---

## Normalization

The training and validation dataset uses:

```python
A.Normalize(mean=0.5, std=0.5)
```

The inference dataset uses:

```python
A.Normalize(mean=3, std=3)
```

These settings should remain consistent with the checkpoint used for prediction.

---

## Output files

During training, model checkpoints are saved in:

```text
ckpt_dualnet/
```

Example:

```text
ckpt_dualnet/
├── dualnet_epoch_001.pth
├── dualnet_epoch_002.pth
├── ...
└── dualnet_epoch_050.pth
```

During test mode, predicted masks are saved in:

```text
test/
└── predictions_dualnet/
```

During predict mode, predicted masks are saved in:

```text
test/
└── predict_one_checkpoint/
```

The exact output folders are determined by:

```python
ckpt_dir
pred_dir
predict_save_dir
```

---

## Test mode

To evaluate all saved checkpoints on the test dataset, set:

```python
MODE = "test"
```

The script loads each `.pth` checkpoint in `ckpt_dualnet/`, predicts masks for the test images, and calculates:

```text
Dice coefficient
IoU score
```

if ground-truth masks are available.

The top-performing checkpoints are printed based on Dice and IoU.

---

## Predict mode

To apply one trained checkpoint to input images, set:

```python
MODE = "predict"
```

Set the checkpoint path:

```python
PREDICT_CKPT_PATH = r"path/to/dualnet_epoch_050.pth"
```

The script saves predicted binary masks to:

```python
predict_save_dir
```

---

## Myotube area Run_folder

The `Run_folder` inside `Myotube_area` contains example files related to applying the trained myotube area segmentation model.

It may include:

```text
Alpha_Actinin_Images/
Segmentation_Results/
Stitching_Results/
```

These folders are provided as examples of input images and expected segmentation/stitching outputs.

The training script itself uses the paired dataset in:

```text
Myotube_area/Train_folder/
```

---

## Relationship to downstream MyoCAS analysis

The trained myotube area segmentation model can be used to quantify myotube area from α-actinin-stained images.

The trained sarcomere segmentation model can be used to generate sarcomere masks for downstream sarcomere maturation classification and Grad-CAM visualization.

In the downstream sarcomere classification pipeline, the sarcomere segmentation checkpoint generated here is used to convert input images into sarcomere masks before EfficientNet-B4 classification.

---

## Notes for reproducibility

- Use the same folder names expected by the script: `train`, `val`, `test`, `actinin`, and `mask`.
- The example files in this repository are provided as PNG files.
- Keep the same preprocessing and normalization settings when using trained checkpoints.
- Keep mask naming consistent with the input image base name.
- Use the same model architecture when loading saved checkpoints.
- Record the checkpoint file name used for downstream analysis.
- Record whether the checkpoint was trained for `Myotube_area` or `Sarcomere`.
- The sarcomere segmentation checkpoint used downstream should match the sarcomere mask generation setting used during classifier training.