# Sarcomere Segmentation, Classification, and Grad-CAM

This module performs sarcomere mask-based maturation classification and Grad-CAM visualization for MyoCAS.

The workflow is:

```text
α-actinin-stained input image
↓
Sarcomere segmentation using HRNet + DeepLabV3 checkpoint
↓
Binary sarcomere mask generation
↓
Mask converted to 3-channel image
↓
EfficientNet-B4 classification
↓
Pre / Nascent / Mature prediction
↓
Optional Grad-CAM and Grad-CAM++ visualization
```

This module is used after training the sarcomere segmentation model in:

```text
03_HRNet_DeepLabV3_Segmentation_Training
```

The sarcomere segmentation checkpoint generated from that module is required before training or running the classifier.

---

## Main script

```text
MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
```

This script supports:

```text
1. Sarcomere segmentation mask generation
2. EfficientNet-B4 classifier training
3. Checkpoint evaluation
4. Confusion matrix generation
5. Folder-level classification
6. Grad-CAM and Grad-CAM++ visualization
```

---

## Required weight files

This module uses two different weight files.

### 1. Sarcomere segmentation checkpoint

```text
HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
```

This checkpoint is required first.

It is used to convert α-actinin-stained input images into binary sarcomere segmentation masks.

In the script, set:

```python
BEST_SEG_CHECK_SAR = r"weights/HRNet_DeepLabV3_best_Sarcomere_segmentation.pth"
```

### 2. Sarcomere classification checkpoint

```text
efficient_net_sarcomere_best.pth
```

This checkpoint is generated after training the EfficientNet-B4 classifier.

It is used for final sarcomere maturation classification and Grad-CAM visualization.

In the script, set:

```python
CLASSIFY_EPOCH = r"weights/efficient_net_sarcomere_best.pth"
```

---

## Folder structure

Recommended folder structure:

```text
04_Sarcomere_Segmentation_Classification_GradCAM/
├── README.md
├── MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
├── weights/
│   ├── HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
│   └── efficient_net_sarcomere_best.pth
├── Run_folder/
│   ├── Pre/
│   │   └── example_pre.png
│   ├── Nascent/
│   │   └── example_nascent.png
│   ├── Mature/
│   │   └── example_mature.png
│   ├── classification_result/
│   │   ├── Pre/
│   │   ├── Nascent/
│   │   └── Mature/
│   └── GradCAM_Results/
│       └── efficient_net_sarcomere_best/
│           ├── Pre/
│           ├── Nascent/
│           └── Mature/
└── Train_folder/
    ├── Train/
    │   ├── Pre/
    │   │   ├── example_pre_001.png
    │   │   ├── example_pre_002.png
    │   │   └── ...
    │   ├── Nascent/
    │   │   ├── example_nascent_001.png
    │   │   ├── example_nascent_002.png
    │   │   └── ...
    │   └── Mature/
    │       ├── example_mature_001.png
    │       ├── example_mature_002.png
    │       └── ...
    ├── Val/
    │   ├── Pre/
    │   ├── Nascent/
    │   └── Mature/
    └── Test/
        ├── Pre/
        ├── Nascent/
        └── Mature/
```

For this repository, all example images are PNG files.

The example images in both `Run_folder` and `Train_folder` are α-actinin-stained images.

---

## Class labels

The classifier predicts three sarcomere maturation classes:

```text
0 : Pre
1 : Nascent
2 : Mature
```

The script uses:

```python
CLASS_NAME = ["Pre", "Nascent", "Mature"]
```

Therefore, the training folders should be named exactly:

```text
Pre
Nascent
Mature
```

If you want to use folder names such as `Pre_myofibril`, `Nascent_myofibril`, or `Mature_myofibril`, you must also update `CLASS_NAME` in the script.

Recommended public folder names:

```text
Pre
Nascent
Mature
```

Scientific meaning:

```text
Pre     : premyofibril-like sarcomere pattern
Nascent : nascent myofibril-like sarcomere pattern
Mature  : mature sarcomere / striated pattern
```

---

## Main workflow

This module does not classify the original α-actinin image directly.

Instead, each input image is first converted into a sarcomere segmentation mask.

```text
Input α-actinin image
↓
HRNet + DeepLabV3 sarcomere segmentation model
↓
Binary sarcomere mask
↓
3-channel mask image
↓
EfficientNet-B4 classifier
↓
Pre / Nascent / Mature prediction
```

This design ensures that the classifier focuses on sarcomere-positive structural patterns rather than unrelated background intensity.

---

## Training data

Training data should be placed in:

```text
Train_folder/
```

Expected structure:

```text
Train_folder/
├── Train/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
├── Val/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
└── Test/
    ├── Pre/
    ├── Nascent/
    └── Mature/
```

Each class folder should contain α-actinin-stained PNG images.

Example:

```text
Train_folder/
├── Train/
│   ├── Pre/
│   │   ├── pre_001.png
│   │   ├── pre_002.png
│   │   └── ...
│   ├── Nascent/
│   │   ├── nascent_001.png
│   │   ├── nascent_002.png
│   │   └── ...
│   └── Mature/
│       ├── mature_001.png
│       ├── mature_002.png
│       └── ...
├── Val/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
└── Test/
    ├── Pre/
    ├── Nascent/
    └── Mature/
```

In the example repository, each class folder may contain a small number of example PNG files.

For actual training, use a larger and balanced dataset.

---

## Run folder

`Run_folder` contains example images for testing the trained workflow.

Recommended structure:

```text
Run_folder/
├── Pre/
│   └── example_pre.png
├── Nascent/
│   └── example_nascent.png
├── Mature/
│   └── example_mature.png
├── classification_result/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
└── GradCAM_Results/
    └── efficient_net_sarcomere_best/
        ├── Pre/
        ├── Nascent/
        └── Mature/
```

The `Pre`, `Nascent`, and `Mature` folders contain α-actinin-stained example images.

The `classification_result` folder contains images sorted by the predicted class.

The `GradCAM_Results` folder contains Grad-CAM and Grad-CAM++ visualization outputs.

---

## Important settings

### Training mode

```python
TRAIN_MODE = "on"
```

Available options:

```text
"on"  : train the EfficientNet-B4 sarcomere maturation classifier
"off" : do not train
```

During training, each image is first segmented using the sarcomere segmentation checkpoint, and the generated mask is used as the classifier input.

---

### Test mode

```python
TEST_MODE = "off"
```

Available options:

```text
"on"  : evaluate saved classifier checkpoints on the test dataset
"off" : skip checkpoint evaluation
```

When `TEST_MODE = "on"`, the script evaluates classifier checkpoints using the test dataset and reports accuracy.

---

### Confusion matrix mode

```python
MATRIX = "off"
```

Available options:

```text
"on"  : generate a confusion matrix using BEST_CKPT and the test dataset
"off" : skip confusion matrix generation
```

The confusion matrix is saved as a PNG image.

---

### Grad-CAM mode

```python
GRAD_CAM_MODE = "off"
```

Available options:

```text
"off"    : disable Grad-CAM
"single" : generate Grad-CAM and Grad-CAM++ for one selected image
"batch"  : generate Grad-CAM and Grad-CAM++ for all images in a folder
```

Grad-CAM is computed using the same sarcomere segmentation mask input used for classification.

---

### Folder-level classification mode

```python
CLASSIFICATION_MODE = "off"
```

Available options:

```text
"on"  : classify all images in CLASSIFY_GRAD_INPUT_DIR into Pre/Nascent/Mature folders
"off" : skip folder-level classification
```

When enabled, the script segments each input image, classifies it, and saves or moves it into the corresponding predicted class folder.

---

### Continue training mode

```python
KEEP_MODE = "on"
```

Available options:

```text
"on"  : resume training from the latest epoch checkpoint if available
"off" : train from scratch
```

---

## Important mode combinations

Only turn on the mode you want to run.

### Train classifier

```python
TRAIN_MODE = "on"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "off"
```

### Evaluate classifier checkpoints

```python
TRAIN_MODE = "off"
TEST_MODE = "on"
MATRIX = "off"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "off"
```

### Generate confusion matrix

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "on"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "off"
```

### Generate batch Grad-CAM

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "batch"
CLASSIFICATION_MODE = "off"
```

### Classify all images in a folder

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "on"
```

Important: if `GRAD_CAM_MODE = "batch"` and `CLASSIFICATION_MODE = "on"` are both enabled, the Grad-CAM branch is executed first.  
For folder-level classification, set `GRAD_CAM_MODE = "off"`.

---

## Root path setting

For training, set `ROOT` to the training dataset folder.

Example:

```python
ROOT = Path(r"path/to/Train_folder")
```

Expected structure:

```text
ROOT/
├── Train/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
├── Val/
│   ├── Pre/
│   ├── Nascent/
│   └── Mature/
└── Test/
    ├── Pre/
    ├── Nascent/
    └── Mature/
```

In the script:

```python
TRAIN_DIR = ROOT / "Train"
VAL_DIR = ROOT / "Val"
TEST_DIR = ROOT / "Test"
```

---

## Checkpoint settings

### Sarcomere segmentation checkpoint

Set:

```python
BEST_SEG_CHECK_SAR = r"weights/HRNet_DeepLabV3_best_Sarcomere_segmentation.pth"
```

This checkpoint must exist before classifier training, testing, classification, or Grad-CAM.

It is used to generate sarcomere masks from α-actinin images.

### EfficientNet-B4 classification checkpoint

Set:

```python
CLASSIFY_EPOCH = r"weights/efficient_net_sarcomere_best.pth"
```

This checkpoint is used for folder-level classification and Grad-CAM visualization.

During training, the best model is saved as:

```python
BEST_CKPT = CHECKPOINT_DIR / "best.pth"
```

You can copy or rename the trained best model to:

```text
weights/efficient_net_sarcomere_best.pth
```

for public use or downstream inference.

---

## Training the classifier

To train the EfficientNet-B4 classifier:

1. Prepare the training data in `Train_folder`.
2. Place the sarcomere segmentation checkpoint in `weights/`.
3. Set `ROOT`.
4. Set the modes as follows:

```python
TRAIN_MODE = "on"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "off"
```

5. Run:

```bash
python MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
```

During training:

```text
α-actinin image
↓
sarcomere segmentation mask
↓
EfficientNet-B4 classifier training
```

The generated segmentation masks are cached in:

```text
Saved_Segmentation_Masks/
```

if:

```python
SAVE_SEG_MASK = True
```

---

## Folder-level classification

To classify all images in `Run_folder`, set:

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "on"
```

Set:

```python
CLASSIFY_EPOCH = r"weights/efficient_net_sarcomere_best.pth"
CLASSIFY_GRAD_INPUT_DIR = Path(r"path/to/Run_folder")
```

Then run:

```bash
python MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
```

The output is saved in:

```text
Run_folder/
└── classification_result/
    ├── Pre/
    ├── Nascent/
    └── Mature/
```

In the current script, `copy_mode=False` is used in the final classification call.  
This means that images may be moved into the predicted class folders.

If you want to preserve the original files, change:

```python
copy_mode=False
```

to:

```python
copy_mode=True
```

---

## Grad-CAM and Grad-CAM++

Grad-CAM and Grad-CAM++ can be generated to visualize regions that contributed to classification.

The script generates visualizations based on the segmentation-mask input used by the classifier.

For batch Grad-CAM, set:

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "off"
GRAD_CAM_MODE = "batch"
CLASSIFICATION_MODE = "off"
```

Set:

```python
CLASSIFY_EPOCH = r"weights/efficient_net_sarcomere_best.pth"
CLASSIFY_GRAD_INPUT_DIR = Path(r"path/to/Run_folder")
GRAD_CAM_FOLDER_NAME = "GradCAM_Results"
```

Then run:

```bash
python MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
```

Output examples:

```text
Run_folder/
└── GradCAM_Results/
    └── efficient_net_sarcomere_best/
        ├── Pre/
        │   ├── example_pre_pred=Pre_prob=0.999.png
        │   └── example_pre_GradCAMpp_on_mask.png
        ├── Nascent/
        └── Mature/
```

Each Grad-CAM figure may include:

```text
Original image
Segmentation mask
Grad-CAM heatmap
Grad-CAM++ heatmap
Grad-CAM overlay on original image
Grad-CAM++ overlay on original image
Grad-CAM overlay on mask
Grad-CAM++ overlay on mask
```

---

## Confusion matrix

To generate a confusion matrix:

```python
TRAIN_MODE = "off"
TEST_MODE = "off"
MATRIX = "on"
GRAD_CAM_MODE = "off"
CLASSIFICATION_MODE = "off"
```

The script uses:

```python
BEST_CKPT = CHECKPOINT_DIR / "best.pth"
```

and saves a confusion matrix figure in the checkpoint directory.

---

## Output files and folders

Depending on the selected mode, this module can generate:

```text
checkpoints/
tensorboard_logs/
Saved_Segmentation_Masks/
Checkpoint_Evaluation_Results.xlsx
classification_result/
GradCAM_Results/
confusion_matrix_best.png
```

### `checkpoints/`

Contains classifier training checkpoints.

```text
epoch_001.pth
epoch_002.pth
...
best.pth
optimizer_001.pth
optimizer_002.pth
...
```

### `Saved_Segmentation_Masks/`

Contains cached sarcomere masks generated from α-actinin images.

These masks are reused during classifier training and evaluation.

### `classification_result/`

Contains images sorted into predicted classes:

```text
Pre/
Nascent/
Mature/
```

### `GradCAM_Results/`

Contains Grad-CAM and Grad-CAM++ visualization results.

---

## Notes on input images

All example images in this repository are PNG files.

The images are α-actinin-stained fluorescence images.

The script may support additional image formats, but the public example data are provided as PNG files for simplicity.

---

## Relationship to the previous segmentation module

This module depends on the sarcomere segmentation checkpoint trained in:

```text
03_HRNet_DeepLabV3_Segmentation_Training
```

Specifically, this file is required:

```text
weights/HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
```

That checkpoint is used to generate sarcomere masks before classification.

After the EfficientNet-B4 classifier is trained, the resulting classification checkpoint can be used for:

```text
1. Sarcomere maturation classification
2. Folder-level image sorting
3. Grad-CAM / Grad-CAM++ visualization
```

---

## Notes for reproducibility

- Use the same sarcomere segmentation checkpoint for training and inference.
- Keep the class folder names consistent with `CLASS_NAME = ["Pre", "Nascent", "Mature"]`.
- Keep input images as PNG files when using the provided example data.
- Record the segmentation checkpoint used for mask generation.
- Record the EfficientNet-B4 classifier checkpoint used for classification.
- Keep the same preprocessing and mask-generation settings across training, testing, classification, and Grad-CAM.
- Do not compare classifier checkpoints trained with different segmentation checkpoints unless this difference is explicitly reported.
- For Grad-CAM interpretation, remember that the classifier input is the segmentation mask, not the raw α-actinin image.