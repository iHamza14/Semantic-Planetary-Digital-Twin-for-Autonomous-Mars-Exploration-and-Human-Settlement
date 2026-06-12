import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import UNet from train.py
from train import UNet, CLASS_MAPPING, NUM_CLASSES, calculate_iou

TEST_DIR = "/Users/hamza/Desktop/Arihant/Testing/rgb"
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.test_dir / self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]


def create_color_map():
    """Create a color map for visualization"""
    colors = [
        [34, 139, 34],    # Trees - Forest Green
        [50, 205, 50],    # Lush Bushes - Lime Green
        [210, 180, 140],  # Dry Grass - Tan
        [139, 69, 19],    # Dry Bushes - Saddle Brown
        [160, 82, 45],    # Ground Clutter - Sienna
        [255, 192, 203],  # Flowers - Pink
        [101, 67, 33],    # Logs - Dark Brown
        [128, 128, 128],  # Rocks - Gray
        [222, 184, 135],  # Landscape - Burlywood
        [135, 206, 235]   # Sky - Sky Blue
    ]
    return np.array(colors, dtype=np.uint8)


def mask_to_rgb(mask, color_map):
    """Convert mask to RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_idx in range(NUM_CLASSES):
        rgb[mask == cls_idx] = color_map[cls_idx]
    
    return rgb


def visualize_predictions(images, predictions, filenames, color_map, output_dir='runs/predictions'):
    """Save prediction visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    for img, pred, filename in zip(images, predictions, filenames):
        # Denormalize image
        img = img.cpu()
        pred = pred.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # Convert prediction to RGB
        pred_rgb = mask_to_rgb(pred.cpu().numpy(), color_map)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_rgb)
        axes[1].set_title('Segmentation Prediction')
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'pred_{filename}')
        plt.savefig(output_path)
        plt.close()


def test_model(model, test_loader, device='cuda', visualize=True):
    """Test the model on test dataset"""
    model.eval()
    color_map = create_color_map()
    
    all_predictions = []
    all_filenames = []
    
    print('Running inference on test images...')
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions)
            all_filenames.extend(filenames)
            
            # Visualize first 10 images
            if visualize and batch_idx < 2:
                visualize_predictions(images, predictions, filenames, color_map)
    
    print(f'Predictions saved to runs/predictions/')
    return all_predictions, all_filenames


def create_class_distribution(predictions):
    """Analyze class distribution in predictions"""
    class_names = [
        'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 
        'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
    ]
    
    all_preds = torch.cat([p.flatten() for p in predictions])
    class_counts = torch.bincount(all_preds, minlength=NUM_CLASSES)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(class_names, class_counts.cpu().numpy())
    ax.set_xlabel('Class')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Predicted Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('runs/class_distribution.png')
    print('Class distribution saved to runs/class_distribution.png')

import segmentation_models_pytorch as smp

def get_model(num_classes):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 4
    MODEL_PATH = 'runs/best_model.pth'
    TEST_DIR = "/Users/hamza/Desktop/Arihant/Testing/rgb"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = TestDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Load model
    model= get_model(num_classes=NUM_CLASSES)
    
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=False))
    model.to(device)
    # model.eval()
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device,weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Loaded model from {MODEL_PATH}')
            print(f'Model was trained to IoU: {checkpoint.get("val_iou", "N/A")}')
        else:
            model.load_state_dict(checkpoint)
            print(f'Loaded model weights from {MODEL_PATH}')
    else:
        print(f'Error: Model file not found at {MODEL_PATH}')
        print('Please train the model first using train.py')
        exit(1)
    
    # Run testing
    predictions, filenames = test_model(model, test_loader, device=device, visualize=True)
    
    # Create class distribution analysis
    create_class_distribution(predictions)
    
    # Save results summary
    results = {
        'num_test_images': len(test_dataset),
        'model_path': MODEL_PATH,
        'predictions_saved': 'runs/predictions/',
    }
    
    with open('runs/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print('\nTesting complete!')
    print(f'Processed {len(test_dataset)} test images')
    print('Results saved to runs/ directory')