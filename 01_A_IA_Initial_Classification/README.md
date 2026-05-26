# A/IA Initial Classification

This module performs the first-stage active/inactive region classification in MyoCAS.

In this step, time-lapse microscopy videos or subregion-level videos are analyzed using temporal motion features. Each region is classified as either active (A) or inactive (IA).

This module is used before optical flow-derived contractile displacement analysis and downstream active/inactive/noise (A/IA/N) classification.

---

## Classes

```text
A  : active region
IA : inactive region
```

In the training script, labels are assigned as:

```text
Active   -> label 1
Inactive -> label 0
```

Folder names must contain either `Active` or `Inactive` because labels are assigned based on the folder name.

---

## Role in the MyoCAS workflow

This module corresponds to the initial activity screening step in the MyoCAS workflow.

The output of this module is used to identify candidate active regions before optical flow-derived contractile displacement analysis and final A/IA/N region classification.

Overall workflow:

```text
Original time-lapse video
↓
Spatial subregion division, optional
↓
Temporal motion feature extraction
↓
A/IA initial classification
↓
Candidate active region selection
↓
Optical flow-derived displacement analysis
↓
Final A/IA/N classification
```

---

## Folder structure

Recommended folder structure:

```text
01_A_IA_initial_classification/
├── README.md
├── train_A_IA_classifier.py
├── split_and_classify_A_IA_regions.py
├── weights/
│   └── myocas_A_IA_classifier_epoch034_acc0.9370.pth
├── Run_folder/
│   ├── example_1.avi
│   └── example_2.avi
└── Train_folder/
    ├── Train/
    │   ├── Active/
    │   │   └── example_active_001.avi
    │   └── Inactive/
    │       └── example_inactive_001.avi
    ├── Val/
    │   ├── Active/
    │   │   └── example_active_001.avi
    │   └── Inactive/
    │       └── example_inactive_001.avi
    └── Test/
        ├── Active/
        │   └── example_active_001.avi
        └── Inactive/
            └── example_inactive_001.avi
```

Recommended file names for public GitHub release:

```text
train_A_IA_classifier.py
split_and_classify_A_IA_regions.py
```

---

## Files

### `train_A_IA_classifier.py`

This script trains, tests, and applies the first-stage A/IA classifier.

The script extracts temporal motion features from subregion-level time-lapse videos and trains a DNN-based binary classifier.

Supported modes:

```python
MODE = "train"    # Train the A/IA classifier
MODE = "test"     # Evaluate saved checkpoint files
MODE = "predict"  # Apply the trained classifier to unlabeled videos
```

---

### `split_and_classify_A_IA_regions.py`

This script divides original time-lapse microscopy videos into spatial subregions and classifies each subregion as active or inactive using the trained A/IA classifier.

This script is intended for prediction.

It can be used to divide original videos into 12 or 48 spatial subregions before applying the trained A/IA classifier.

---

## Important note about `BASIC_PATH`

`BASIC_PATH` has different meanings depending on the script.

### In `train_A_IA_classifier.py`

For training and testing, `BASIC_PATH` should point to `Train_folder`, which directly contains the `Train`, `Val`, and `Test` folders.

Example:

```python
BASIC_PATH = r"path/to/Train_folder"
```

Expected structure:

```text
Train_folder/
├── Train/
│   ├── Active/
│   └── Inactive/
├── Val/
│   ├── Active/
│   └── Inactive/
└── Test/
    ├── Active/
    └── Inactive/
```

### In `split_and_classify_A_IA_regions.py`

`BASIC_PATH` should point to `Run_folder`, which directly contains the original videos for prediction.

Example:

```python
BASIC_PATH = r"path/to/Run_folder"
```

Expected structure:

```text
Run_folder/
├── example_1.avi
└── example_2.avi
```

If `DIVIDED_MODE = "on"`, the script divides the original videos into spatial subregions and saves them in the `Divided/` folder.

---

## Input data structure for training

For training and testing, organize the input videos as follows:

```text
Train_folder/
├── Train/
│   ├── Active/
│   └── Inactive/
├── Val/
│   ├── Active/
│   └── Inactive/
└── Test/
    ├── Active/
    └── Inactive/
```

Set:

```python
BASIC_PATH = r"path/to/Train_folder"
```

Folder names must contain either `Active` or `Inactive`.

```text
Active   -> label 1
Inactive -> label 0
```

The input videos should be subregion-level time-lapse videos.

---

## Input data for prediction

### Prediction with `train_A_IA_classifier.py`

For prediction with `train_A_IA_classifier.py`, place already divided subregion-level videos directly in the directory specified by `BASIC_PATH`.

Example:

```text
BASIC_PATH/
├── region_001.avi
├── region_002.avi
├── region_003.avi
└── ...
```

Then set:

```python
MODE = "predict"
BASIC_PATH = r"path/to/divided_region_videos"
BEST_MODEL = r"weights/myocas_A_IA_classifier_epoch034_acc0.9370.pth"
```

This mode directly classifies videos already present in `BASIC_PATH`.

---

### Prediction with `split_and_classify_A_IA_regions.py`

For prediction with `split_and_classify_A_IA_regions.py`, place original time-lapse microscopy videos directly in `Run_folder`.

Example:

```text
Run_folder/
├── example_1.avi
└── example_2.avi
```

Then set:

```python
BASIC_PATH = r"path/to/Run_folder"
BEST_MODEL = r"weights/myocas_A_IA_classifier_epoch034_acc0.9370.pth"
DIVIDED_MODE = "on"
```

If `DIVIDED_MODE = "on"`, each original video in `Run_folder` is divided into spatial subregions before A/IA classification.

---

## Supported video formats

Supported video formats include:

```text
.mp4, .mov, .wmv, .mkv, .mpeg, .flv, .webm, .avi
```

---

## Video division

The number of spatial subregions is controlled by `ROWS` and `COLS` in `split_and_classify_A_IA_regions.py`.

Examples:

```python
ROWS = 3
COLS = 4   # 3 × 4 = 12 subregions
```

```python
ROWS = 6
COLS = 8   # 6 × 8 = 48 subregions
```

If `DIVIDED_MODE = "on"`, each original video is divided into spatial subregions before A/IA classification.

```python
DIVIDED_MODE = "on"
```

If `DIVIDED_MODE = "off"`, videos in `BASIC_PATH` are classified directly without additional division.

```python
DIVIDED_MODE = "off"
```

---

## Feature extraction

Both scripts use the same temporal motion feature extraction procedure.

For each video, the following steps are performed:

1. Load sampled grayscale frames from the video.
2. Calculate a median reference image.
3. Generate a temporal motion curve from absolute pixel-wise differences between each frame and the median reference image.
4. Extract six temporal motion features.
5. Apply the DNN-based A/IA classifier.

The six temporal motion features are:

```text
1. Temporal standard deviation
2. Maximum value
3. Minimum value
4. Full range
5. Percentile-based range
6. Mean value
```

The frame sampling interval is controlled by:

```python
FRAME_STEP = 2
```

The foreground threshold is controlled by:

```python
BG_THR = 50
```

---

## Model architecture

The A/IA classifier is a fully connected neural network.

The model receives six temporal motion features as input and outputs one logit for binary classification.

During evaluation or prediction, sigmoid activation is applied to the output logit.

```text
Input: 6 temporal motion features
↓
Fully connected DNN
↓
Output: 1 logit
↓
Sigmoid
↓
Active / Inactive prediction
```

This model is different from the downstream A/IA/N classifier.

The A/IA initial classifier is a DNN-based binary classifier, whereas the downstream A/IA/N classifier uses a BiLSTM-based hybrid model.

---

## Model checkpoint

The trained checkpoint should contain:

```text
model  : trained DNN weights
scaler : fitted StandardScaler
acc    : validation accuracy
```

The fitted `StandardScaler` saved in the checkpoint must be used during prediction.

Place trained model weights in the `weights/` directory.

Example:

```text
weights/
└── myocas_A_IA_classifier_epoch034_acc0.9370.pth
```

Before running prediction, set `BEST_MODEL` to the checkpoint path.

Example:

```python
BEST_MODEL = r"weights/myocas_A_IA_classifier_epoch034_acc0.9370.pth"
```

---

## Output

### Training output

During training, checkpoint files are saved to the weight directory.

Example:

```text
dnn_weights/
├── epoch_000_acc_0.8500.pth
├── epoch_001_acc_0.8750.pth
├── ...
└── best_model.pth
```

Feature cache files may also be generated:

```text
train_feature_cache_features.csv
val_feature_cache_features.csv
test_feature_cache_features.csv
predict_feature_cache_features.csv
```

These cache files store extracted features to avoid repeated feature extraction.

---

### Prediction output

Prediction can generate:

```text
Divided/
Active Objects/
Inactive Objects/
prediction_result.csv
```

Output meaning:

```text
Divided/              : generated only when split_and_classify_A_IA_regions.py is used with DIVIDED_MODE = "on"
Active Objects/       : videos predicted as active regions
Inactive Objects/     : videos predicted as inactive regions
prediction_result.csv : summary of A/IA prediction results
```

Note that `Divided/` is not always generated.  
It is generated only when videos are divided into spatial subregions by `split_and_classify_A_IA_regions.py`.

---

## Basic usage

### 1. Train the A/IA classifier

Edit the settings in `train_A_IA_classifier.py`:

```python
MODE = "train"
BASIC_PATH = r"path/to/Train_folder"
```

Then run:

```bash
python train_A_IA_classifier.py
```

---

### 2. Test saved checkpoints

Edit the settings in `train_A_IA_classifier.py`:

```python
MODE = "test"
BASIC_PATH = r"path/to/Train_folder"
```

Then run:

```bash
python train_A_IA_classifier.py
```

In test mode, all `.pth` checkpoint files in the weight directory are evaluated on the test dataset.

---

### 3. Predict A/IA labels for already divided videos

Edit the settings in `train_A_IA_classifier.py`:

```python
MODE = "predict"
BASIC_PATH = r"path/to/divided_region_videos"
BEST_MODEL = r"weights/myocas_A_IA_classifier_epoch034_acc0.9370.pth"
```

Then run:

```bash
python train_A_IA_classifier.py
```

This mode directly classifies videos already present in `BASIC_PATH`.

---

### 4. Divide original videos and classify subregions

Edit the settings in `split_and_classify_A_IA_regions.py`:

```python
BASIC_PATH = r"path/to/Run_folder"
BEST_MODEL = r"weights/myocas_A_IA_classifier_epoch034_acc0.9370.pth"

DIVIDED_MODE = "on"

ROWS = 3
COLS = 4
```

Then run:

```bash
python split_and_classify_A_IA_regions.py
```

This mode divides each original video into spatial subregions and then classifies each subregion as active or inactive.

---

## Important note about file movement

During prediction, videos are moved into active or inactive output folders.

If you want to preserve the original files, replace the file-moving step with file copying in the script.

Use:

```python
shutil.copy2(src, dst)
```

instead of:

```python
shutil.move(src, dst)
```

This is especially important when you want to keep the original divided videos for later inspection or repeated analysis.

---

## Notes for reproducibility

- The same feature extraction method must be used for training and prediction.
- The fitted `StandardScaler` saved in the checkpoint must be used during prediction.
- The number and order of input features must remain unchanged.
- The trained checkpoint must match the six-feature DNN classifier.
- Video frame sampling is controlled by `FRAME_STEP`.
- Foreground masking is controlled by `BG_THR`.
- The number of subregions is controlled by `ROWS` and `COLS`.
- The same video preprocessing settings should be used when comparing results across experiments.

---

## Relationship to downstream MyoCAS analysis

This module performs the first-stage A/IA classification.

Downstream MyoCAS analysis uses selected regions for optical flow-derived contractile displacement analysis and final active/inactive/noise region classification.

In other words, this module is used to screen and select candidate active regions, while the downstream MyoCAS pipeline performs detailed displacement quantification and final A/IA/N classification.