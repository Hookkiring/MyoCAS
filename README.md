# MyoCAS

MyoCAS enables multi-dimensional quantification of myotube contractility and reveals structure–function coupling in vitro.

MyoCAS is an image-analysis platform for quantitative analysis of myotube contractility, active-region detection, myotube area, sarcomere organization, and sarcomere maturation patterns from microscopy images and time-lapse videos.

This repository contains the main MyoCAS analysis scripts, model training scripts, lightweight example data, documentation, and example outputs used for the MyoCAS workflow. Large trained model weight files are provided separately as GitHub Release assets.

This repository is intended for academic, educational, and non-commercial research use. Commercial use is not permitted without prior permission from the author. See the [License](#license) section for details.

---

## Overview

MyoCAS integrates functional and structural analysis of skeletal muscle myotubes.

The platform includes:

```text
1. Initial active/inactive region classification
2. Optical flow-based myotube contraction analysis
3. Active / Inactive / Noise region classification
4. Myotube area segmentation
5. Sarcomere segmentation
6. Sarcomere maturation classification
7. Grad-CAM / Grad-CAM++ visualization
```

The main functional workflow is:

```text
Time-lapse microscopy video
↓
Initial active/inactive region screening
↓
Candidate contraction/relaxation frame detection
↓
Optical flow-derived displacement analysis
↓
FFT-derived feature extraction
↓
A/IA/N classification
↓
Contractile displacement and active-region quantification
```

The structural workflow is:

```text
α-actinin-stained fluorescence image
↓
HRNet + DeepLabV3-based segmentation
↓
Myotube area or sarcomere mask generation
↓
Sarcomere maturation classification
↓
Grad-CAM / Grad-CAM++ visualization
```

---

## Detailed documentation and example files

Each module folder includes its own `README.md` file with detailed instructions, required input folder structure, parameter settings, example files, and example output results.

Please refer to the README file inside each folder before running the scripts.

```text
01_A_IA_Initial_Classification/README.md
02_MyoCAS_Analysis/README.md
03_HRNet_DeepLabV3_Segmentation_Training/README.md
04_Sarcomere_Segmentation_Classification_GradCAM/README.md
```

Small example data and corresponding output results are included to help users understand the expected input/output structure and to test whether the scripts run correctly.

The example files are provided for demonstration and reproducibility checks only. Users should replace them with their own microscopy images, time-lapse videos, calibration values, and trained checkpoints for actual analysis.

---

## Downloading the prepared MyoCAS package

For reproducible use of MyoCAS, please download the prepared release assets from the GitHub Releases page.

Do not use the GitHub auto-generated "Source code (zip)" file as a replacement for the prepared MyoCAS package, because it may not contain the prepared example folders, example outputs, or trained model weight files.

Please download the following files:

Please download the following files:

```text
MyoCAS_repository_files_without_weights.zip
MyoCAS_trained_weights.zip

After downloading:

1. Extract MyoCAS_repository_files_without_weights.zip.
2. Extract MyoCAS_trained_weights.zip.
3. Place each .pth file in the corresponding module folder or weights/ folder according to the module-specific README.

The trained weight files are not committed directly to the main repository because of file size limitations.

```text
MyoCAS/
├── README.md
│
├── 01_A_IA_Initial_Classification/
│   ├── README.md
│   ├── train_A_IA_classifier.py
│   ├── split_and_classify_A_IA_regions.py
│   ├── weights/
│   ├── Run_folder/
│   └── Train_folder/
│
├── 02_MyoCAS_Analysis/
│   ├── README.md
│   ├── Run_MyoCAS_Analysis.py
│   ├── Train_MyoCAS_A_IA_N_Classifier.py
│   ├── weights/
│   ├── data/
│   └── example_outputs/
│
├── 03_HRNet_DeepLabV3_Segmentation_Training/
│   ├── README.md
│   ├── Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
│   └── Train_folder/
│       ├── Myotube_area/
│       └── Sarcomere/
│
└── 04_Sarcomere_Segmentation_Classification_GradCAM/
    ├── README.md
    ├── MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
    ├── Run_folder/
    ├── Train_folder/
    ├── efficient_net_sarcomere_best.pth
    └── HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
```

---

## Modules

## 1. Initial active/inactive region classification

Folder:

```text
01_A_IA_Initial_Classification/
```

Main scripts:

```text
train_A_IA_classifier.py
split_and_classify_A_IA_regions.py
```

This module performs the first-stage active/inactive region screening.

It can divide original time-lapse videos into spatial subregions and classify each region as:

```text
A  : Active region
IA : Inactive region
```

The classifier uses temporal motion features extracted from pixel-intensity changes relative to a median reference image.

This module is used before optical flow-derived displacement analysis to select candidate active regions.

Main functions:

```text
Train active/inactive classifier
Split original videos into 12 or 48 subregions
Classify subregions as active or inactive
Export active/inactive region videos
```

See the module-specific README for details:

```text
01_A_IA_Initial_Classification/README.md
```

---

## 2. MyoCAS displacement analysis and A/IA/N classification

Folder:

```text
02_MyoCAS_Analysis/
```

Main scripts:

```text
Run_MyoCAS_Analysis.py
Train_MyoCAS_A_IA_N_Classifier.py
```

This is the main MyoCAS contractility analysis module.

It performs:

```text
Frame extraction
Pixel-intensity-based candidate contraction/relaxation frame detection
Optical flow-derived displacement analysis
Moved-pixel and moved-area calculation
FFT-derived feature extraction
A/IA/N classification
Excel output
Figure output
Composite video generation
```

The final region-level classes are:

```text
ACTIVE
INACTIVE
NOISE
```

The A/IA/N classifier uses:

```text
1. Two-channel displacement time series
2. Six scalar displacement/FFT-derived features
3. BiLSTM-based temporal branch
4. Fully connected scalar feature branch
```

Representative output files include:

```text
Optical_Flow_Displacement.png
Optical_Flow_Displacement_FFT.png
Analysis_ROI_and_Tracked_Points.png
Moved_Pixel_Count_Time_Series.png
Moved_Region_Area_Time_Series.png
Moved_Area_vs_Displacement.png
MyoCAS_Composite_Video.avi
ACTIVE_SAMPLE.xlsx
INACTIVE_SAMPLE.xlsx
NOISE_SAMPLE.xlsx
```

See the module-specific README for details:

```text
02_MyoCAS_Analysis/README.md
```

---

## 3. HRNet + DeepLabV3 segmentation training

Folder:

```text
03_HRNet_DeepLabV3_Segmentation_Training/
```

Main script:

```text
Train_MyoCAS_HRNet_DeepLabV3_Segmentation.py
```

This module trains a custom dual-branch segmentation model combining:

```text
CustomHRNet
DeepLabV3-ResNet101
Fusion head
```

The model performs binary segmentation for:

```text
1. Myotube area segmentation
2. Sarcomere segmentation
```

The training dataset uses paired α-actinin fluorescence images and binary mask images.

Expected dataset structure:

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

The trained myotube area segmentation model can be used for myotube area quantification.

The trained sarcomere segmentation checkpoint is used in the downstream sarcomere maturation classification and Grad-CAM module.

See the module-specific README for details:

```text
03_HRNet_DeepLabV3_Segmentation_Training/README.md
```

---

## 4. Sarcomere segmentation, classification, and Grad-CAM

Folder:

```text
04_Sarcomere_Segmentation_Classification_GradCAM/
```

Main script:

```text
MyoCAS_Sarcomere_Segmentation_Classification_GradCAM.py
```

This module performs sarcomere mask-based maturation classification.

Workflow:

```text
α-actinin-stained image
↓
Sarcomere segmentation using HRNet + DeepLabV3 checkpoint
↓
Binary sarcomere mask
↓
Mask converted to 3-channel image
↓
EfficientNet-B4 classifier
↓
Pre / Nascent / Mature classification
↓
Grad-CAM / Grad-CAM++ visualization
```

The classifier predicts three sarcomere maturation classes:

```text
0 : Pre
1 : Nascent
2 : Mature
```

Required weight files:

```text
HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
efficient_net_sarcomere_best.pth
```

This module supports:

```text
EfficientNet-B4 classifier training
Checkpoint evaluation
Confusion matrix generation
Folder-level classification
Grad-CAM visualization
Grad-CAM++ visualization
```

See the module-specific README for details:

```text
04_Sarcomere_Segmentation_Classification_GradCAM/README.md
```

---

## Installation

## Recommended environment

MyoCAS was developed and tested mainly with:

```text
Python 3.9
PyTorch 2.0.1 + CUDA 11.8
Torchvision 0.15.2 + CUDA 11.8
NumPy 1.26.4
OpenCV 4.9.0
```

Using a virtual environment is recommended.
The scripts were mainly developed and executed in Visual Studio Code on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

---

## Windows C++ Build Tools

Some Python packages may require Microsoft C++ build tools during installation.

On Windows, install:

```text
Build Tools for Visual Studio
```

During installation, select:

```text
Desktop development with C++
MSVC C++ build tools
Windows 10/11 SDK
C++ CMake tools for Windows
```

This cannot be installed by `pip`.

---

## Install PyTorch

## GPU version, CUDA 11.8

Use this if you have an NVIDIA GPU and want to run MyoCAS with CUDA acceleration.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output:

```text
2.0.1+cu118
True
11.8
```

## CPU-only version

Use this if you do not have an NVIDIA GPU or do not want to use CUDA.

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

---

## Install dependencies

Install the main dependencies:

```bash
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
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
pip install tensorboard==2.20.0
pip install efficientnet-pytorch==0.7.1
pip install seaborn==0.13.2
pip install tifffile==2024.8.30
```

One-line version:

```bash
pip install numpy==1.26.4 opencv-python==4.9.0.80 matplotlib==3.9.4 openpyxl==3.1.5 natsort==8.4.0 pandas==2.2.3 scikit-learn==1.6.1 scipy==1.13.1 plotly==6.0.0 PyWavelets==1.6.0 albumentations==2.0.8 albucore==0.0.24 tqdm==4.67.1 moviepy==1.0.1 imageio==2.37.0 imageio-ffmpeg==0.6.0 pillow==10.4.0 decorator==4.4.2 tensorboard==2.20.0 efficientnet-pytorch==0.7.1 seaborn==0.13.2 tifffile==2024.8.30
```

---

## Installation check

After installation, run:

```bash
python -c "import cv2, numpy, pandas, torch, torchvision, albumentations, pywt, moviepy, efficientnet_pytorch; print('MyoCAS environment check passed'); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:

```text
MyoCAS environment check passed
Torch: 2.0.1+cu118
CUDA available: True
```

If CUDA is not available, MyoCAS can still run on CPU, but processing will be slower.

---

## Example data and outputs

This repository includes small example data and example output results for demonstration.

The example files are intended to help users understand:

```text
Input folder structure
Expected output files
Model checkpoint usage
Analysis workflow
```

The example outputs were generated by running the corresponding scripts on the example inputs.

Users should refer to the README file inside each module folder for detailed instructions and example-specific explanations.

The example data are not intended to replace full experimental datasets.

For actual analysis, users should replace the example data with their own microscopy images, time-lapse videos, calibration values, and trained checkpoints.

---

## Required input types

Depending on the module, MyoCAS uses:

```text
Time-lapse microscopy videos
α-actinin-stained fluorescence images
Binary mask images
Trained PyTorch checkpoint files
```

Example video files may be provided as:

```text
.avi
.mp4
```

Example image files are provided mainly as:

```text
.png
```

---

## Required weight files

Different modules require different trained weight files.

Examples:

```text
MyoCAS_A_IA_N_classification.pth
MyoCAS_Segmentation.pth
HRNet_DeepLabV3_best_Sarcomere_segmentation.pth
efficient_net_sarcomere_best.pth
```

Place the required weight files in the corresponding module folder or in the `weights/` folder specified by each script.

---

## Reproducibility notes

For reproducible analysis, record the following information:

```text
Python version
PyTorch version
CUDA version
OpenCV version
Microscope model
Magnification
Pixel-to-micrometer calibration
Input video FPS
Checkpoint file names
Analysis mode settings
```

Important settings include:

```text
PIXEL_UM
AREA_MOVED
VIDEO_FPS
Hz
NUM_FRAMES
CLASS_NAMES
COLUMNS
```

For segmentation and classification workflows, record:

```text
Segmentation checkpoint
Classification checkpoint
Input image preprocessing
Normalization settings
Class folder names
Mask generation method
```

---

## Notes on interpretation

MyoCAS provides quantitative image-analysis outputs.

For biological interpretation, users should consider:

```text
Experimental design
Imaging conditions
Cell culture condition
Stimulation condition
Drug treatment condition
Segmentation accuracy
Classification accuracy
Biological replicate structure
```

Optical flow-derived displacement represents image-based contractile movement and should be interpreted together with experimental context.

Sarcomere classification results should be interpreted as image-based maturation pattern classification, not as direct molecular mechanism inference.

Grad-CAM and Grad-CAM++ visualizations are provided to support model interpretation, but they should not be treated as independent biological evidence.

---

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2026 Hyeon Jun Choe

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

You are free to:

```text
Share — copy and redistribute the material in any medium or format
Adapt — remix, transform, and build upon the material
```

Under the following terms:

```text
Attribution — You must give appropriate credit to the original author.
NonCommercial — You may not use the material for commercial purposes.
```

Academic, educational, and non-commercial research use is permitted under the terms of CC BY-NC 4.0.

Commercial use, including use in commercial products, paid services, proprietary platforms, contract research, or company-internal commercial development, is not permitted without prior permission from the author.

For commercial use, collaboration, licensing, or other inquiries, please contact the author.

To view a copy of this license, visit:

```text
https://creativecommons.org/licenses/by-nc/4.0/
```

---

## Citation

If you use MyoCAS in your research, please cite the associated manuscript once available.
