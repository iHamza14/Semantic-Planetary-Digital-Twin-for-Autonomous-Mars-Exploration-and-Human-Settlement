# Duality AI Semantic Segmentation - Hackathon Submission

## Team Information

**Team Name:** PixelRaiders
**Project:** Offroad Semantic Scene Segmentation  
**Challenge:** Duality AI Code Sprint Hackathon

---

## Project Overview

This project implements a robust semantic segmentation model for desert environments using synthetic data from Duality AI's Falcon platform. The model accurately segments 10 different classes including vegetation, terrain features, and sky.

### Key Features

- ✅ U-Net architecture for precise segmentation
- ✅ PyTorch implementation with CUDA support
- ✅ Comprehensive training and testing pipeline
- ✅ Visual analysis and metrics tracking
- ✅ High-contrast visualization tools

---

## Quick Start Guide

### 1. Prerequisites

- Miniconda or Anaconda installed
- NVIDIA GPU (recommended) or CPU
- Downloaded dataset from Falcon platform

### 2. Environment Setup

#### For Windows:

```batch
cd ENV_SETUP
setup_env.bat
```

#### For Mac/Linux:

```bash
cd ENV_SETUP
chmod +x setup_env.sh
./setup_env.sh
```

### 3. Directory Structure

Organize your project as follows:

```
project/
├── data/
│   ├── train/
│   │   ├── rgb/        # Training RGB images
│   │   └── seg/        # Training segmentation masks
│   ├── val/
│   │   ├── rgb/        # Validation RGB images
│   │   └── seg/        # Validation segmentation masks
│   └── testImages/     # Test RGB images (no masks)
├── ENV_SETUP/
│   ├── setup_env.bat
│   └── setup_env.sh
├── runs/               # Output directory (created automatically)
├── train.py
├── test.py
├── visualize_segmentation.py
├── config.yaml
└── requirements.txt
```

### 4. Running the Model

#### Step 1: Activate Environment

```bash
conda activate EDU
```

#### Step 2: Train the Model

```bash
python train.py
```

This will:

- Train the U-Net model for 50 epochs
- Save checkpoints to `runs/best_model.pth`
- Generate training metrics and graphs
- Display progress with IoU scores

**Expected Output:**

- Training and validation loss curves
- Validation IoU scores (target: > 0.60)
- Best model checkpoint

#### Step 3: Test on Unseen Data

```bash
python test.py
```

This will:

- Load the best trained model
- Run inference on test images
- Save predicted segmentation masks
- Generate class distribution analysis

**Expected Output:**

- Prediction visualizations in `runs/predictions/`
- Class distribution chart
- Test results summary

#### Step 4: Visualize Training Data (Optional)

```bash
# Create legend
python visualize_segmentation.py --mode legend

# Visualize training samples
python visualize_segmentation.py --mode batch --rgb data/train/rgb --seg data/train/seg --num-samples 10

# Visualize single image
python visualize_segmentation.py --mode single --rgb path/to/image.png --seg path/to/mask.png --output output.png
```

---

## Model Architecture

### U-Net Segmentation Network

**Architecture Details:**

- **Encoder:** 4 down-sampling blocks (64, 128, 256, 512 channels)
- **Bottleneck:** 1024 channels
- **Decoder:** 4 up-sampling blocks with skip connections
- **Output:** 10-class pixel-wise classification

**Key Components:**

- Batch normalization for stable training
- ReLU activation functions
- Skip connections for preserving spatial information
- Max pooling for down-sampling
- Transposed convolutions for up-sampling

---

## Class Information

| ID    | Class Name     | Color        | Description          |
| ----- | -------------- | ------------ | -------------------- |
| 100   | Trees          | Forest Green | Tall vegetation      |
| 200   | Lush Bushes    | Lime Green   | Dense shrubs         |
| 300   | Dry Grass      | Tan          | Short vegetation     |
| 500   | Dry Bushes     | Saddle Brown | Sparse shrubs        |
| 550   | Ground Clutter | Sienna       | Debris/small objects |
| 600   | Flowers        | Deep Pink    | Flowering plants     |
| 700   | Logs           | Dark Brown   | Fallen trees         |
| 800   | Rocks          | Gray         | Stones/boulders      |
| 7100  | Landscape      | Burlywood    | General ground       |
| 10000 | Sky            | Sky Blue     | Sky regions          |

---

## Training Configuration

### Hyperparameters

- **Batch Size:** 8
- **Epochs:** 50
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Image Size:** 256x256

### Data Preprocessing

- Resize to 256x256 pixels
- Normalization (ImageNet mean/std)
- Channel-wise standardization

---

## Performance Metrics

### Evaluation Criteria

1. **IoU (Intersection over Union):** Primary metric for segmentation accuracy
   - Formula: IoU = (Intersection) / (Union)
   - Target: > 0.60 for competitive performance

2. **Training Loss:** Should steadily decrease
3. **Validation Loss:** Should decrease without overfitting

### Expected Results

- **Training Loss:** < 0.5 after 50 epochs
- **Validation IoU:** 0.55 - 0.75 (depends on dataset quality)
- **Test mAP50:** 0.60 - 0.80 (mean Average Precision at IoU 0.5)
- **Inference Time:** < 50ms per image (on GPU)

---

## Output Files

After running the complete pipeline, you'll have:

### In `runs/` directory:

- `best_model.pth` - Best model checkpoint
- `final_model.pth` - Final epoch weights
- `training_metrics.png` - Loss and IoU curves
- `metrics.json` - Numerical results
- `test_results.json` - Test summary

### In `runs/predictions/` directory:

- `pred_[filename].png` - Visualization for each test image
- Side-by-side comparison of input and segmentation

### In project root:

- `segmentation_legend.png` - Color-coded class legend

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:** Reduce batch size in `train.py`

```python
BATCH_SIZE = 4  # or even 2
```

#### 2. Slow Training

**Solutions:**

- Reduce image size to 128x128
- Use fewer epochs initially (20-30)
- Close other GPU applications

#### 3. Low IoU Score

**Solutions:**

- Train for more epochs (70-100)
- Adjust learning rate (try 0.0005)
- Check data quality and class balance
- Add data augmentation

#### 4. Module Not Found Errors

**Solution:** Reinstall environment

```bash
conda deactivate
conda remove -n EDU --all
# Then run setup script again
```

### Mac/Linux Specific

If `setup_env.sh` doesn't work:

```bash
# Manual setup
conda create -n EDU python=3.9 -y
conda activate EDU
pip install -r requirements.txt
```

---

## Advanced Usage

### Custom Training

Edit hyperparameters in `train.py`:

```python
BATCH_SIZE = 16          # Increase for faster training
NUM_EPOCHS = 100         # Train longer
LEARNING_RATE = 0.0005   # Fine-tune learning rate
```

### Resume Training

To continue from a checkpoint:

```python
# In train.py, before training loop:
checkpoint = torch.load('runs/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Custom Model

Replace UNet with your own architecture in `train.py`.

---

## File Descriptions

| File                        | Purpose                                       |
| --------------------------- | --------------------------------------------- |
| `train.py`                  | Main training script with UNet implementation |
| `test.py`                   | Testing and inference on unseen data          |
| `visualize_segmentation.py` | Visualization utilities                       |
| `config.yaml`               | Configuration parameters                      |
| `requirements.txt`          | Python dependencies                           |
| `setup_env.bat`             | Windows environment setup                     |
| `setup_env.sh`              | Mac/Linux environment setup                   |
| `README.md`                 | This documentation                            |

---

---

## Performance Report Template

### Methodology

We implemented a U-Net architecture trained on synthetic desert environment data. The model was trained for 50 epochs using Adam optimizer with a learning rate of 0.001.
# Offroad-Semantic-Scene-Segmentation
