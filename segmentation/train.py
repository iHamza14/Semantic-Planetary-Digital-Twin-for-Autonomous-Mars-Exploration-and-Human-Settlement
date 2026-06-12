import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Class mapping
CLASS_MAPPING = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}

NUM_CLASSES = 10

train_path="/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_Training_Dataset"

class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir, seg_dir, transform=None):
        self.rgb_dir = Path(rgb_dir)
        self.seg_dir = Path(seg_dir)
        self.transform = transform
        
        self.rgb_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_dir / self.rgb_images[idx]
        seg_path = self.seg_dir / self.rgb_images[idx]
        
        rgb_image = Image.open(rgb_path).convert('RGB')
        seg_image = Image.open(seg_path)
        seg_image = seg_image.resize((256, 256), Image.NEAREST)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Convert segmentation to class indices
        seg_array = np.array(seg_image)
        seg_tensor = torch.zeros((seg_array.shape[0], seg_array.shape[1]), dtype=torch.long)
        
        for original_class, new_class in CLASS_MAPPING.items():
            mask = seg_array == original_class
            seg_tensor[mask] = new_class
        
        return rgb_image, seg_tensor

import segmentation_models_pytorch as smp

def get_model(num_classes):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=NUM_CLASSES):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(n_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, n_classes, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        return out


def calculate_iou(pred, target, num_classes=NUM_CLASSES):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, device='cuda'):
    train_losses = []
    val_losses = []
    val_ious = []
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                total_iou += calculate_iou(preds, masks)
        
        val_loss = val_loss / len(val_loader)
        val_iou = total_iou / len(val_loader)
        
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val IoU: {val_iou:.4f}')
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, 'runs/best_model.pth')
            print(f'  Best model saved with IoU: {val_iou:.4f}')
    
    return train_losses, val_losses, val_ious


def plot_metrics(train_losses, val_losses, val_ious):
    os.makedirs('runs', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU plot
    ax2.plot(val_ious, label='Val IoU', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU Score')
    ax2.set_title('Validation IoU Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('runs/training_metrics.png')
    print('Metrics saved to runs/training_metrics.png')


if __name__ == '__main__':
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SegmentationDataset(os.path.join(train_path, "train", "rgb"), os.path.join(train_path, "train", "seg"), transform=transform)
    val_dataset = SegmentationDataset(os.path.join(train_path, "val", "rgb"), os.path.join(train_path, "val", "seg"), transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Initialize model
    # model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
    model= get_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train_losses, val_losses, val_ious = train_model(
        train_loader, val_loader, model, criterion, optimizer, 
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, val_ious)
    
    # Save final model
    torch.save(model.state_dict(), 'runs/final_model.pth')
    
    # Save metrics to JSON
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'best_iou': max(val_ious)
    }
    
    with open('runs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print('\nTraining complete!')
    print(f'Best validation IoU: {max(val_ious):.4f}')