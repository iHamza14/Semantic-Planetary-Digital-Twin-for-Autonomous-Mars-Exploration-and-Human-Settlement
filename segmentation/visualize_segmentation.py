import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

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

CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def create_color_map():
    """Create a high-contrast color map for visualization"""
    colors = [
        [34, 139, 34],    # Trees - Forest Green
        [50, 205, 50],    # Lush Bushes - Lime Green
        [210, 180, 140],  # Dry Grass - Tan
        [139, 69, 19],    # Dry Bushes - Saddle Brown
        [160, 82, 45],    # Ground Clutter - Sienna
        [255, 20, 147],   # Flowers - Deep Pink
        [101, 67, 33],    # Logs - Dark Brown
        [128, 128, 128],  # Rocks - Gray
        [222, 184, 135],  # Landscape - Burlywood
        [135, 206, 235]   # Sky - Sky Blue
    ]
    return np.array(colors, dtype=np.uint8)


def convert_segmentation_to_rgb(seg_image_path, output_path=None):
    """Convert a segmentation mask to RGB using high-contrast colors"""
    # Load segmentation image
    seg_img = Image.open(seg_image_path)
    seg_array = np.array(seg_img)
    
    # Create RGB image
    h, w = seg_array.shape[:2]
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get color map
    color_map = create_color_map()
    
    # Convert each class to its color
    for original_class, new_class in CLASS_MAPPING.items():
        mask = seg_array == original_class
        rgb_img[mask] = color_map[new_class]
    
    # Save or display
    if output_path:
        Image.fromarray(rgb_img).save(output_path)
        print(f'Saved colored segmentation to {output_path}')
    
    return rgb_img


def visualize_with_rgb(rgb_image_path, seg_image_path, output_path=None):
    """Visualize RGB image alongside its segmentation"""
    # Load images
    rgb_img = Image.open(rgb_image_path)
    seg_rgb = convert_segmentation_to_rgb(seg_image_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(seg_rgb)
    axes[1].set_title('Segmentation Mask', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {output_path}')
    else:
        plt.show()
    
    plt.close()


def create_legend():
    """Create a legend showing all classes and their colors"""
    color_map = create_color_map()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create color patches
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, color_map)):
        ax.add_patch(plt.Rectangle((0, i), 1, 0.8, 
                                   facecolor=color/255.0, 
                                   edgecolor='black'))
        ax.text(1.2, i+0.4, class_name, va='center', fontsize=12)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(CLASS_NAMES))
    ax.axis('off')
    ax.set_title('Segmentation Class Legend', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('segmentation_legend.png', dpi=150, bbox_inches='tight')
    print('Saved legend to segmentation_legend.png')
    plt.show()


def batch_visualize(rgb_dir, seg_dir, output_dir, num_samples=5):
    """Visualize multiple samples from the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for i, rgb_file in enumerate(rgb_files[:num_samples]):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        seg_path = os.path.join(seg_dir, rgb_file)
        output_path = os.path.join(output_dir, f'viz_{rgb_file}')
        
        if os.path.exists(seg_path):
            visualize_with_rgb(rgb_path, seg_path, output_path)
        else:
            print(f'Warning: Segmentation not found for {rgb_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize segmentation masks')
    parser.add_argument('--mode', type=str, default='batch', 
                       choices=['single', 'batch', 'legend'],
                       help='Visualization mode')
    parser.add_argument('--rgb', type=str, help='Path to RGB image or directory')
    parser.add_argument('--seg', type=str, help='Path to segmentation image or directory')
    parser.add_argument('--output', type=str, default='visualizations', 
                       help='Output path or directory')
    parser.add_argument('--num-samples', type=int, default=5, 
                       help='Number of samples to visualize in batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'legend':
        create_legend()
    elif args.mode == 'single':
        if not args.rgb or not args.seg:
            print('Error: --rgb and --seg required for single mode')
        else:
            visualize_with_rgb(args.rgb, args.seg, args.output)
    elif args.mode == 'batch':
        if not args.rgb or not args.seg:
            # Use default paths
            rgb_dir = 'data/train/rgb'
            seg_dir = 'data/train/seg'
        else:
            rgb_dir = args.rgb
            seg_dir = args.seg
        
        batch_visualize(rgb_dir, seg_dir, args.output, args.num_samples)
        print(f'\nVisualizations saved to {args.output}/')