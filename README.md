### Environment Setup

Python Version
    Use Python 3.9+
    Install PyTorch

Follow the official instructions for your OS/GPU:
https://pytorch.org/get-started/locally/

Example (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verify CUDA
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

 If running locally
You must have an NVIDIA GPU
Install NVIDIA Drivers + CUDA Toolkit
PyTorch must be installed with CUDA support

 If running on Kaggle

Go to Settings → Accelerator → GPU
Install dependencies (if needed):
pip install torch torchvision torchaudio --upgrade

### Dataset Preparation

This project uses the Kaggle dataset:
Cats, Dogs & Pandas Images Dataset
https://www.kaggle.com/datasets/gpiosenka/cats-dogs-pandas-images
Download dataset inside your notebook:
kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data --unzip

Data Loading
    Using torchvision.datasets.ImageFolder with transforms:
    Resize → 224×224
    Normalize using ImageNet mean/std
    Augmentations for training:
    Horizontal flip
    Random rotation
    Random crop

### Model Design (Transfer Learning)

We use ResNet18 (pretrained on ImageNet).

Steps:

Load pretrained model
Freeze feature extractor
Replace final classification head:
[ResNet Feature Extractor] → FC(256) → ReLU → Dropout(0.5) → FC(3 classes)

The model runs on GPU if available:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

### Training
Training Configuration
Item	Value
Loss	CrossEntropyLoss
Optimizer	Adam
Learning Rate	0.001
Epochs	10–15
Batch Size	32
Seeds	Fixed for reproducibility
Model Checkpointing

Best model is saved using validation accuracy.

### Evaluation

After training, we compute:
    Test Metrics
    Test Loss
    Test Accuracy
Plots

    Confusion Matrix
    Example Predictions

These outputs are included inside the Jupyter notebook.

### Results

The model achieves:
High classification accuracy
Clear separation in confusion matrix
Reliable performance even with small dataset sizes

### Author

KESAVAN S
212224230121