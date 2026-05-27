# MyoCAS: Myotube Contraction Analysis System

MyoCAS is a image-analysis pipeline for quantitative analysis of myotube contractility from time-lapse microscopy videos.

This repository contains two main scripts:

```text
Run_MyoCAS_Analysis.py
Train_MyoCAS_A_IA_N_Classifier.py
```

`Run_MyoCAS_Analysis.py` performs the main MyoCAS analysis workflow, including frame extraction, pixel-intensity-based candidate contraction/relaxation frame detection, optical-flow displacement analysis, optional myotube segmentation, FFT/wavelet feature extraction, A/IA/N classification, Excel export, figure export, and optional video generation.

`Train_MyoCAS_A_IA_N_Classifier.py` trains and evaluates the MyoCAS A/IA/N classifier. The classifier uses a two-channel displacement time series processed by a bidirectional LSTM branch, together with six scalar displacement/FFT-derived features processed by a fully connected branch.

---

## 1. Repository structure

Recommended repository structure:

Note: The `weights/` and `data/` folders may not be included directly in the GitHub source tree. The prepared example folders and trained weight files are provided through the GitHub Release assets. Users can also create these folders manually when using their own data and checkpoints.

```text
MyoCAS/
├── Run_MyoCAS_Analysis.py
├── Train_MyoCAS_A_IA_N_Classifier.py
├── weights/
│   ├── MyoCAS_A_IA_N_classification.pth
│   └── MyoCAS_Segmentation.pth
├── data/
│   ├── Run_folder/
│   │   ├── 12/
│   │   │   ├── ACTIVE/
│   │   │   │   ├── sample_001.avi
│   │   │   │   └── ...
│   │   │   └── INACTIVE/
│   │   │       ├── sample_002.avi
│   │   │       └── ...
│   │   ├── 48/
│   │   │   ├── ACTIVE/
│   │   │   │   ├── sample_003.avi
│   │   │   │   └── ...
│   │   │   └── INACTIVE/
│   │   │       ├── sample_004.avi
│   │   │       └── ...
│   │   └── ...
│   └── Train_folder/
│       └── Train_datasets/
│           ├── TRAIN/
│           │   ├── ACTIVE/
│           │   ├── INACTIVE/
│           │   └── NOISE/
│           ├── VAL/
│           │   ├── ACTIVE/
│           │   ├── INACTIVE/
│           │   └── NOISE/
│           └── TEST/
│               ├── ACTIVE/
│               ├── INACTIVE/
│               └── NOISE/
└── README.md
```

---

## 2. Installation

### 2.1 Recommended environment

MyoCAS was developed and tested mainly with the following environment:

```text
Python 3.9
PyTorch 2.0.1 + CUDA 11.8
Torchvision 0.15.2 + CUDA 11.8
NumPy 1.26.4
OpenCV 4.9.0
```

Using a virtual environment is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

For Windows PowerShell, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

---

### 2.2 Windows C++ Build Tools

Some Python packages may require Microsoft C++ build tools during installation.

On Windows, install:

```text
Build Tools for Visual Studio
```

Visual Studio 2019, Visual Studio 2022, or the latest Visual Studio Build Tools should be acceptable. During installation, select:

```text
Desktop development with C++
MSVC C++ build tools
Windows 10/11 SDK
C++ CMake tools for Windows
```

This cannot be installed by `pip`. It must be installed separately from the Microsoft Visual Studio downloads page.

---

### 2.3 Install PyTorch

#### Option A: GPU version, CUDA 11.8

Use this if you have an NVIDIA GPU and want to run MyoCAS with CUDA acceleration.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

After installation, check CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output:

```text
2.0.1+cu118
True
11.8
```

If `torch.cuda.is_available()` returns `False`, the script can still run on CPU, but processing will be slower.

#### Option B: CPU-only version

Use this if you do not have an NVIDIA GPU or do not want to use CUDA.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

---

### 2.4 Install MyoCAS dependencies

Install the required packages for `Run_MyoCAS_Analysis.py` and `Train_MyoCAS_A_IA_N_Classifier.py`.

```bash
pip install numpy==1.26.4
pip install opencv-python==4.11.0.86
pip install matplotlib==3.9.4
pip install openpyxl==3.1.5
pip install natsort==8.4.0
pip install pandas==2.2.3
pip install scikit-learn==1.6.1
pip install scipy==1.13.1
pip install plotly==6.0.0
pip install PyWavelets==1.6.0
pip install albumentations==2.0.8
pip install albucore==0.0.24
pip install tqdm==4.67.1
pip install moviepy==1.0.1
pip install imageio==2.37.0
pip install imageio-ffmpeg==0.6.0
pip install pillow==10.4.0
pip install decorator==4.4.2
```

One-line version:

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 matplotlib==3.9.4 openpyxl==3.1.5 natsort==8.4.0 pandas==2.2.3 scikit-learn==1.6.1 scipy==1.13.1 plotly==6.0.0 PyWavelets==1.6.0 albumentations==2.0.8 albucore==0.0.24 tqdm==4.67.1 moviepy==1.0.1 imageio==2.37.0 imageio-ffmpeg==0.6.0 pillow==10.4.0 decorator==4.4.2
```


### 2.5 Standard-library modules

The following modules are used in the scripts but do not need to be installed with `pip` because they are included in Python:

```text
os
shutil
time
math
gc
glob
itertools
datetime
ast
multiprocessing
concurrent.futures
io
```

---

### 2.6 Installation check

After installation, run:

```bash
python -c "import cv2, os, numpy, matplotlib, openpyxl, natsort, pandas, sklearn, scipy, plotly, pywt, torch, torchvision, albumentations, tqdm, moviepy; print('MyoCAS environment check passed'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If the environment is correctly installed, you should see:

```text
MyoCAS environment check passed
Torch: 2.0.1+cu118
CUDA available: True
```

If CUDA is not available, MyoCAS can still run on CPU.

Recommended CPU setting:

```python
GPU_MODE = 'off'

If GPU_MODE = 'on' but CUDA is unavailable, the script attempts to fall back to CPU automatically.

## 3. Main analysis script

### Script

```bash
python Run_MyoCAS_Analysis.py
```

### Purpose

`Run_MyoCAS_Analysis.py` analyzes time-lapse microscopy videos of contracting myotubes.

Main steps:

1. Extract frames from input videos.
2. Detect candidate contraction/relaxation frames using pixel-intensity changes.
3. Track displacement using optical flow.
4. Convert pixel displacement to micrometers.
5. Calculate displacement-derived features.
6. Perform FFT
7. Optionally segment myotube area.
8. Classify each analyzed region as:
   - `ACTIVE`
   - `INACTIVE`
   - `NOISE`
9. Export Excel files, plots, and optional videos.

---

## 4. Input folder structure for analysis

Input videos can be organized by experimental condition and manually assigned activity class.

Example:

```text
data/Run_folder/
├── 12/
│   ├── ACTIVE/
│   │   ├── sample_001.avi
│   │   ├── sample_002.avi
│   │   └── ...
│   └── INACTIVE/
│       ├── sample_003.avi
│       ├── sample_004.avi
│       └── ...
├── 48/
│   ├── ACTIVE/
│   │   ├── sample_005.avi
│   │   └── ...
│   └── INACTIVE/
│       ├── sample_006.avi
│       └── ...
└── ...
```

Set `VIDEO_FILE_PATH` to the folder that directly contains the input videos.

Examples:

```python
VIDEO_FILE_PATH = r"path/to/Run_folder/12/ACTIVE"
```

or

```python
VIDEO_FILE_PATH = r"path/to/Run_folder/12/INACTIVE"
```

or

```python
VIDEO_FILE_PATH = r"path/to/Run_folder/48/ACTIVE"
```

The current script processes one input-video folder at a time. If videos are stored in nested folders such as `Run_folder/12/ACTIVE`, set `VIDEO_FILE_PATH` to that final folder rather than to `Run_folder`.

Supported video extensions:

```python
['mp4', 'mov', 'wmv', 'mkv', 'mpeg', 'flv', 'webm', 'avi']
```

When the script runs, frames are extracted into a folder with the same name as each video:

```text
data/Run_folder/12/ACTIVE/
├── sample_001.avi
├── sample_001/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
```

---

## 5. Required weight files

The analysis script requires trained model checkpoints.

### A/IA/N classifier checkpoint

```python
BEST_CLASSIFICATION_CHECKPOINT_FILE = r"weights/MyoCAS_A_IA_N_classification.pth"
```

This checkpoint is used to classify each analyzed region as active, inactive, or noise.

### Myotube segmentation checkpoint

```python
BEST_SEGMENTATION_CHECKPOINT_FILE = r"weights/MyoCAS_Segmentation.pth"
```

## 6. Important analysis parameters

### Pixel-to-micrometer calibration

```python
PIXEL_UM = 0.78125
```

This value converts optical-flow displacement from pixels to micrometers.

```text
displacement_um = displacement_px × PIXEL_UM
```

Set this value according to the microscope magnification and image resolution used in your experiment.

### Pixel area calibration

```python
AREA_MOVED = round((PIXEL_UM * PIXEL_UM), 4)
```

This value represents the physical area of one pixel and is used for moved-area and segmentation-area calculations.

### Initial motion search window

```python
NUM_FRAMES = 4
```

The initial search window is calculated as:

```text
fps × NUM_FRAMES
```

This helps focus the analysis on early contraction-related motion and reduces the chance of selecting large noise or bubble movement.

### Reference frequency

```python
Hz = 1
```

The dominant frequency is estimated from the displacement signal during FFT analysis. This value is used as a fallback/reference frequency when needed.

### Fallback FPS

```python
VIDEO_FPS = 30
```

The actual FPS is automatically read from each input video when possible. If FPS cannot be detected from video metadata, `VIDEO_FPS` is used.

---

## 7. GPU / CPU mode

```python
GPU_MODE = 'on'
```

Available options:

```text
"on"  : use CUDA GPU acceleration when available
"off" : force CPU processing
```

If `GPU_MODE = 'on'` but CUDA is not available, the script automatically falls back to CPU.

---

## 8. Main analysis modes

### A/IA/N classification

```python
CLASSIFICATION_MODE = 'on'
```

If `CLASSIFICATION_MODE = 'on'`, the trained classifier is used to classify each analyzed region as:

```text
INACTIVE
ACTIVE
NOISE
```

### Video generation

```python
VIDEO_MODE = 'off'
```

If `VIDEO_MODE = 'on'`, MyoCAS generates composite videos showing:

1. Original phase-contrast video.
2. Optical-flow overlay.
3. Displacement time-series plot.
4. Analysis ROI video.

Video generation increases processing time and output file size.

### Frame extraction skip mode

```python
FRAME_EXTRACTION_SKIP = 'on'
```

If extracted frames already exist and pass validation, the script reuses them instead of extracting frames again.

### Extra figure export mode

```python
EXPORT_EXTRA_FIGURES = 'off'
```

Recommended setting for routine analysis:

```python
EXPORT_EXTRA_FIGURES = 'off'
```

Set to `"on"` only when additional plots are needed for manuscript preparation, manual inspection, or figure layout adjustment.

---

## 9. Optical-flow displacement analysis

MyoCAS identifies candidate maximum contraction and relaxation frames based on pixel-intensity changes, then applies optical-flow tracking to quantify displacement.

The final displacement is reported in micrometers using:

```text
displacement_um = displacement_px × PIXEL_UM
```

Representative output files include:

```text
Optical_Flow_Displacement.png
Optical_Flow_Displacement_FFT.png
Analysis_ROI_and_Tracked_Points.png
```

The displacement time series is also used for A/IA/N classification.

---

## 10. Moved pixel and moved area outputs

MyoCAS also exports auxiliary motion-area indicators based on thresholded pixel-intensity differences.

Representative files include:

```text
Moved_Pixel_Count_Time_Series.png
Moved_Region_Area_Time_Series.png
Moved_Area_vs_Displacement.png
```

These outputs are useful for quality control and should not be confused with optical-flow displacement or segmentation-based myotube area.

---

## 11. Scalar features used for A/IA/N classification

The classifier uses six scalar features:

```python
COLUMNS = [
    'MAX BAND ENERGY',
    'MAX Apeak',
    'MAX ABS DISP',
    'ABS NET CHANGE RATIO',
    'NUM SIGN CHANGES DIFF',
    'DIFF STD'
]
```

Feature meaning:

| Feature | Meaning |
|---|---|
| `MAX BAND ENERGY` | FFT energy around the dominant contraction frequency |
| `MAX Apeak` | Maximum FFT peak amplitude |
| `MAX ABS DISP` | Maximum absolute displacement |
| `ABS NET CHANGE RATIO` | Net displacement change normalized by peak-to-peak range |
| `NUM SIGN CHANGES DIFF` | Number of sign changes in the displacement-difference sequence |
| `DIFF STD` | Standard deviation of displacement differences |

The displacement time series itself is processed separately by the BiLSTM branch.

---

## 12. A/IA/N classifier architecture

The classifier is a hybrid neural network with two input branches.

### Sequence branch

The displacement time series is converted into two channels:

1. Log-transformed raw displacement.
2. Maximum-normalized displacement waveform.

This two-channel sequence is processed by a two-layer bidirectional LSTM with attention pooling.

### Scalar branch

The six scalar features are processed by a fully connected branch.

### Final classifier

The sequence features and scalar features are concatenated and passed through fully connected layers to classify the sample as:

```text
INACTIVE
ACTIVE
NOISE
```

---

## 13. Training the A/IA/N classifier

### Script

```bash
python Train_MyoCAS_A_IA_N_Classifier.py
```

### Dataset structure

The training dataset should be organized as:

```text
data/Train_folder/
└── Train_datasets/
    ├── TRAIN/
    │   ├── ACTIVE/
    │   ├── INACTIVE/
    │   └── NOISE/
    ├── VAL/
    │   ├── ACTIVE/
    │   ├── INACTIVE/
    │   └── NOISE/
    └── TEST/
        ├── ACTIVE/
        ├── INACTIVE/
        └── NOISE/
```

Set the root path to the folder that directly contains `TRAIN`, `VAL`, and `TEST`:

```python
REFERENCE_FILE_PATH = r"path/to/Train_folder/Train_datasets"
```

The script expects the following class names:

```python
CLASS_NAMES = ['INACTIVE', 'ACTIVE', 'NOISE']
```

---

## 14. Training script modes

Set the execution mode:

```python
MODE = 'TRAIN'
```

Available modes:

### `TRAIN`

Trains the MyoCAS A/IA/N classifier using the `TRAIN` and `VAL` datasets.

### `TEST`

Evaluates saved checkpoints using the `TEST` dataset. This mode is useful for evaluating saved checkpoints on the independent test dataset after model training and checkpoint selection.

### `PREDICTION`

Loads `BEST_CHECKPOINT_FILE` and evaluates prediction performance on the `TEST` dataset without training a new model.

### `CLASSIFICATION`

Loads `BEST_CHECKPOINT_FILE` and classifies samples into `ACTIVE`, `INACTIVE`, or `NOISE` output folders.

---

## 15. Training parameters

Important settings:

```python
EPOCHS 
BATCH_SIZE 
LEARNING_RATE
KEEP_TRAINING 
GPU_MODE 
```

### Continue training

```python
KEEP_TRAINING = 'on'
```

If enabled, the script attempts to load `BEST_CHECKPOINT_FILE` or a checkpoint from the checkpoint directory.

Set to `"off"` to train from scratch.

---

## 16. Training outputs

Training outputs are saved under:

```text
result/
├── checkpoints/
│   ├── epoch_0001_val_acc_..._loss_....pth
│   ├── epoch_0002_val_acc_..._loss_....pth
│   └── ...
└── logs/
```

The best model can be copied to:

```text
weights/MyoCAS_A_IA_N_classification.pth
```

and then used by `Run_MyoCAS_Analysis.py`.

---

## 17. Analysis outputs

For each analyzed video/sample, MyoCAS creates a `result` folder containing outputs such as:

```text
result/
├── Optical_Flow_Displacement.png
├── Optical_Flow_Displacement_FFT.png
├── Analysis_ROI_and_Tracked_Points.png
├── Moved_Pixel_Count_Time_Series.png
├── Moved_Region_Area_Time_Series.png
├── Moved_Area_vs_Displacement.png
├── Segmentation_Result_Contraction.png
├── Analysis_Video/
└── *_MyoCAS_Composite_Video.avi
```

The exact output files depend on the selected modes:

```python
VIDEO_MODE
EXPORT_EXTRA_FIGURES
CLASSIFICATION_MODE
```

---

## 18. Excel output files

The analysis script exports classification-based Excel summary files, such as:

```text
ACTIVE_SAMPLE.xlsx
INACTIVE_SAMPLE.xlsx
INACTIVE_DOUBLE_SAMPLE.xlsx
NOISE_SAMPLE.xlsx
OPTICAL_NOISE_SAMPLE.xlsx
DOUBLE_CHECK.xlsx
```

The main columns include:

```text
Folder name
Maximum Intensity of Pixels
Myotube Area
Maximum of Displacement
MAX BAND ENERGY
MAX Apeak
Coordinate X
Coordinate Y
Hz
Displacement List
MAX ABS DISP
ABS NET CHANGE RATIO
NUM SIGN CHANGES DIFF
DIFF STD
```

The six scalar features should match the `COLUMNS` variable used in both analysis and training scripts.

---

## 19. Reproducibility notes

For reproducible analysis, keep the following settings consistent between training and analysis:

```python
PIXEL_UM
AREA_MOVED
NUM_FRAMES
Hz
VIDEO_FPS
COLUMNS
NB_CLASSES
CLASS_NAMES
```

The trained classifier checkpoint must match the model architecture used in the analysis script.

When reporting results, record:

```text
Python version
PyTorch version
CUDA version
OpenCV version
Microscope magnification
PIXEL_UM calibration
Input video FPS
Checkpoint file name
Analysis settings
```

---

## 20. Citation

If you use MyoCAS in your research, please cite the associated manuscript once available.
