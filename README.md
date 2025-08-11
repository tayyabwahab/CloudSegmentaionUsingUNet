# Cloud Segmentation using UNet

A deep learning project for semantic segmentation of clouds in satellite imagery using the UNet architecture. This project implements a complete pipeline from data preprocessing to model training, validation, and inference.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Architecture Details](#model-architecture-details)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements cloud segmentation using the UNet architecture, a deep learning architecture popular for image segmentation tasks. The goal is to accurately identify and segment cloud regions in satellite imagery, which later used for weather forecast and climate research.

### Key Applications
- **Meteorology**: Cloud cover analysis and weather prediction
- **Climate Research**: Cloud pattern analysis and climate change studies
- **Remote Sensing**: Satellite image analysis and interpretation
- **Agriculture**: Weather monitoring for crop management

## 🏗️ Architecture

The project uses a modified UNet architecture with the following components:

- **Encoder**: Convolutional blocks with max pooling for feature extraction
- **Decoder**: Transposed convolutions with skip connections
- **Skip Connections**: Preserve spatial information from encoder to decoder
- **Output Head**: Final convolutional layer for binary segmentation

### Model Specifications
- **Input Channels**: 3 (RGB)
- **Output Classes**: 1 (Binary: cloud vs. non-cloud)
- **Image Dimensions**: 300x300 pixels
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Loss Function**: Binary Cross-Entropy with Logits

## ✨ Features

- **Complete Training Pipeline**: End-to-end training from data loading to model saving
- **Data Augmentation**: Image resizing and normalization
- **Validation Metrics**: Comprehensive evaluation including accuracy, precision, recall, and F1-score
- **Inference Pipeline**: Ready-to-use prediction script
- **Configurable Parameters**: Easy-to-modify hyperparameters and model settings
- **GPU Support**: CUDA acceleration for faster training
- **Visualization**: Training plots and prediction visualization

## 🚀 Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm

### Install Dependencies
```bash
pip install -r requirements.txt
```


## 📊 Dataset

The project uses the SwinySeg dataset for cloud segmentation:

- **Training Images**: RGB satellite images
- **Ground Truth Masks**: Binary masks indicating cloud regions
- **Image Format**: Various formats supported (converted to RGB)
- **Mask Format**: Grayscale binary masks (0 = non-cloud, 255 = cloud)
- **Dataset Split**: 85% training, 15% testing

### Dataset Structure
```
dataset/
├── train/
│   ├── images/          # RGB satellite images
│   └── masks/           # Binary cloud masks
```

## 💻 Usage

### 1. Configuration

Edit `utils/model_configure.py` to set your dataset paths and hyperparameters:

```python
# Dataset paths
DATASET_PATH = "path/to/your/dataset"
Real_Images_Path = "path/to/images"
GT_Path = "path/to/masks"

# Training parameters
n_epochs = 50
BatchSize = 128
l_rate = 0.001
img_rows = 300
img_cols = 300
```

### 2. Training

Run the training script:

```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Split data into training and testing sets
- Initialize the UNet model
- Train for specified epochs
- Save training metrics and plots
- Save the trained model

### 3. Prediction

Use the trained model for inference:

```bash
python predict.py
```

This will:
- Load the trained model
- Process test images
- Generate cloud segmentation masks
- Compare predictions with ground truth

### 4. Evaluation

Run evaluation metrics:

```bash
python Evaluation_Metrics.py
```

## 🎓 Training

### Training Process

1. **Data Loading**: Images and masks are loaded from specified paths
2. **Preprocessing**: Images are resized to 300x300 and normalized
3. **Data Splitting**: 85% training, 15% testing split
4. **Model Initialization**: UNet with specified architecture
5. **Training Loop**: 
   - Forward pass through UNet
   - Loss calculation using BCEWithLogitsLoss
   - Backpropagation and optimization
   - Validation on test set
6. **Metrics Tracking**: Loss and accuracy for both training and testing
7. **Model Saving**: Trained model saved to disk

### Model Configuration (`utils/model_configure.py`)

```python
# Model parameters
n_channels = 1
n_classes = 1
n_levels = 3

# Training parameters
l_rate = 0.001
n_epochs = 50
BatchSize = 128

# Image parameters
img_cols = 300
img_rows = 300
THRESHOLD = 0.5
```


## 📈 Evaluation

### Metrics Calculated

1. **Accuracy**: Pixel-wise accuracy between predictions and ground truth
2. **Precision**: True positive rate for cloud pixels
3. **Recall**: Sensitivity for cloud detection
4. **F1-Score**: Harmonic mean of precision and recall
5. **True Positives (TP)**: Correctly identified cloud pixels
6. **False Positives (FP)**: Incorrectly identified cloud pixels
7. **False Negatives (FN)**: Missed cloud pixels

### Evaluation Process

- Load trained model and test images
- Generate predictions for each test image
- Compare predictions with ground truth masks
- Calculate pixel-wise metrics
- Aggregate results across all test images

## 🏆 Results

The model achieves:
- **Binary Segmentation**: Clear distinction between cloud and non-cloud regions
- **Spatial Accuracy**: Maintains cloud boundaries and shapes
- **Robust Performance**: Consistent results across different image types

### Sample Outputs

The prediction script generates visualizations showing:
- Original satellite image
- Ground truth mask
- Predicted segmentation mask

## 🔧 Model Architecture Details

### UNet Components

#### Encoder
- Progressive downsampling with max pooling
- Feature extraction at multiple scales
- Skip connection storage for decoder

#### Decoder
- Upsampling with transposed convolutions
- Skip connections from encoder
- Progressive feature refinement

#### Final Output
- 1x1 convolution for binary classification
- Sigmoid activation for probability output


## 🚧 Project Structure

```
CloudSegmentaionUsingUNet/
├── train.py                 # Main training script
├── predict.py               # Inference and prediction
├── Evaluation_Metrics.py    # Model evaluation metrics
├── model.py                 # UNet model implementation
├── utils/
│   ├── dataset.py           # Dataset loading and preprocessing
│   └── model_configure.py   # Configuration parameters
├── unet_swinyseg_ful.pth   # Trained model weights
├── plot.png                 # Training visualization
└── README.md                # This file
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.