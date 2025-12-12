#!/usr/bin/env python
"""
Script to generate dataset statistics and example visualizations
for classification and segmentation datasets.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import torch

# Add the path to the datasets.py file
sys.path.append('/data1/vanderbc/nandas1/PostProc')
from datasets import (
    MHISTDataset, CRCDataset, PCamDataset, BRACSDataset, 
    MiDOGDataset, PanNukeDataset, MonuSegDataset
)

# Dataset paths
DATASET_PATHS = {
    'MHIST': "/data1/vanderbc/nandas1/Benchmarks/MHIST_patches_unnormalized",
    'CRC': "/data1/vanderbc/nandas1/Benchmarks/CRC_unnormalized",
    'PCam': "/data1/vanderbc/nandas1/Benchmarks/PatchCamelyon_unnormalized",
    'BRACS': "/data1/vanderbc/nandas1/Benchmarks/BRACS",
    'MiDOG': "/data1/vanderbc/nandas1/Benchmarks/MiDOG++/classification/",
    'PanNuke': "/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized",
    'MonuSeg': "/data1/vanderbc/nandas1/Benchmarks/MonuSeg_patches_unnormalized"
}

# Output directory
OUTPUT_DIR = Path("/data1/vanderbc/nandas1/PostProc/dataset_examples")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def style_subplot(ax, img_shape):
    """Apply consistent styling to a subplot."""
    # Set only 0 and max pixel ticks
    ax.set_xticks([0, img_shape[1]-1])
    ax.set_yticks([0, img_shape[0]-1])
    ax.set_xlabel('[px]', fontsize=9)
    ax.set_ylabel('[px]', fontsize=9)
    
    # Make borders bold and visible
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # Ensure tick marks are visible
    ax.tick_params(width=1.5, length=4, colors='black')


def get_dataset_stats(dataset_class, dataset_path, dataset_name, is_segmentation=False):
    """Get statistics for a dataset."""
    stats = {
        'name': dataset_name,
        'path': dataset_path,
        'train_count': 0,
        'test_count': 0,
        'image_size': None,
        'num_classes': 0,
        'class_names': {}
    }
    
    try:
        if is_segmentation:
            # Load with 40x magnification for segmentation datasets
            train_dataset = dataset_class(dataset_path, split='Training', magnification='40x')
            test_dataset = dataset_class(dataset_path, split='Test', magnification='40x')
            
            stats['train_count'] = len(train_dataset)
            stats['test_count'] = len(test_dataset)
            
            # Get image size from first sample
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                if isinstance(sample[0], torch.Tensor):
                    img_shape = sample[0].shape
                    stats['image_size'] = (img_shape[-2], img_shape[-1])
                else:
                    stats['image_size'] = sample[0].shape[:2]
            
            return stats, train_dataset
        else:
            # Load classification datasets
            train_dataset = dataset_class(dataset_path, split='train')
            test_dataset = dataset_class(dataset_path, split='test')
            
            stats['train_count'] = len(train_dataset)
            stats['test_count'] = len(test_dataset)
            
            # Get image size from first sample
            if len(train_dataset) > 0:
                img, _ = train_dataset[0]
                if isinstance(img, torch.Tensor):
                    stats['image_size'] = (img.shape[1], img.shape[2])
                else:
                    stats['image_size'] = img.size
                    stats['image_size'] = (stats['image_size'][1], stats['image_size'][0])
            
            # Get class information
            stats['num_classes'] = len(train_dataset.class_to_idx)
            stats['class_names'] = {v: k for k, v in train_dataset.class_to_idx.items()}
            
            return stats, train_dataset
            
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return stats, None


def create_classification_figure(dataset, stats, dataset_name):
    """Create figure for classification dataset."""
    plt.style.use('ggplot')
    
    num_classes = stats['num_classes']
    class_names = stats['class_names']
    
    # Special handling for 2-class datasets (PCam, MiDOG)
    if num_classes == 2:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        
        for class_idx in range(2):
            # Find samples for this class
            class_samples = [i for i, (_, target) in enumerate(dataset.samples) if target == class_idx]
            
            # Show 3 examples for this class
            for col in range(3):
                ax = axes[class_idx, col]
                
                if col < len(class_samples):
                    sample_idx = class_samples[col]
                    img_path, _ = dataset.samples[sample_idx]
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    
                    ax.imshow(img_array)
                    
                    # Add class label only to first image in row
                    if col == 0:
                        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                                   edgecolor='black', linewidth=1)
                        ax.text(0.05, 0.95, class_names[class_idx], transform=ax.transAxes,
                               fontsize=9, weight='bold', verticalalignment='top',
                               bbox=props)
                
                # Apply styling
                style_subplot(ax, img_array.shape)
    
    else:
        # Regular layout for datasets with more than 2 classes
        cols = min(3, num_classes)
        rows = (num_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx in range(num_classes):
            row = class_idx // cols
            col = class_idx % cols
            ax = axes[row, col]
            
            # Find first sample of this class
            for img_path, target in dataset.samples:
                if target == class_idx:
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    
                    ax.imshow(img_array)
                    
                    # Add class label
                    props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                               edgecolor='black', linewidth=1)
                    ax.text(0.05, 0.95, class_names[class_idx], transform=ax.transAxes,
                           fontsize=9, weight='bold', verticalalignment='top',
                           bbox=props)
                    
                    # Apply styling
                    style_subplot(ax, img_array.shape)
                    break
        
        # Hide extra subplots if any
        for idx in range(num_classes, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
    
    plt.tight_layout(pad=0.5)
    save_path = OUTPUT_DIR / f"{dataset_name}_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved figure for {dataset_name} at {save_path}")


def create_segmentation_figure(dataset, stats, dataset_name):
    """Create figure for segmentation dataset (PanNuke/MonuSeg)."""
    plt.style.use('ggplot')
    
    # Show 3 examples with images on top, masks on bottom
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    
    for col in range(3):
        if col < len(dataset):
            # Get sample
            sample = dataset[col]
            img = sample[0]  # Image
            instance_mask = sample[3]  # Instance mask
            
            # Convert tensors to numpy if needed
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                # Denormalize if needed
                if img.min() < 0:
                    img = (img - img.min()) / (img.max() - img.min())
            
            if isinstance(instance_mask, torch.Tensor):
                instance_mask = instance_mask.squeeze().numpy()
            
            # Top row: original image
            ax_img = axes[0, col]
            ax_img.imshow(img)
            style_subplot(ax_img, img.shape)
            
            # Add label for first column only
            if col == 0:
                props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                           edgecolor='black', linewidth=1)
                ax_img.text(0.05, 0.95, 'Image', transform=ax_img.transAxes,
                           fontsize=9, weight='bold', verticalalignment='top',
                           bbox=props)
            
            # Bottom row: instance mask
            ax_mask = axes[1, col]
            # Use a colormap that shows instances clearly
            mask_display = ax_mask.imshow(instance_mask, cmap='tab20', interpolation='nearest')
            style_subplot(ax_mask, instance_mask.shape)
            
            # Add label for first column only
            if col == 0:
                props = dict(boxstyle='round', facecolor='white', alpha=0.9,
                           edgecolor='black', linewidth=1)
                ax_mask.text(0.05, 0.95, 'Nuclei', transform=ax_mask.transAxes,
                            fontsize=9, weight='bold', verticalalignment='top',
                            bbox=props)
    
    plt.tight_layout(pad=0.5)
    save_path = OUTPUT_DIR / f"{dataset_name}_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved figure for {dataset_name} at {save_path}")


def print_stats_table(all_stats):
    """Print a formatted table of dataset statistics."""
    print("\n" + "="*70)
    print(" DATASET STATISTICS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<12} {'Train':<10} {'Test':<10} {'Image Size':<12} {'Classes':<8}")
    print("-"*70)
    
    for stats in all_stats:
        if stats['image_size']:
            size_str = f"{stats['image_size'][0]}×{stats['image_size'][1]}"
        else:
            size_str = "N/A"
        
        classes_str = str(stats['num_classes']) if stats['num_classes'] > 0 else "N/A"
        
        print(f"{stats['name']:<12} {stats['train_count']:<10} {stats['test_count']:<10} "
              f"{size_str:<12} {classes_str:<8}")
    
    print("="*70)


def main():
    """Main function to process all datasets."""
    
    # Classification datasets
    classification_datasets = [
        ('MHIST', MHISTDataset),
        ('CRC', CRCDataset),
        ('PCam', PCamDataset),
        ('BRACS', BRACSDataset),
        ('MiDOG', MiDOGDataset)
    ]
    
    # Segmentation datasets
    segmentation_datasets = [
        ('PanNuke', PanNukeDataset),
        ('MonuSeg', MonuSegDataset)
    ]
    
    all_stats = []
    
    # Process classification datasets
    print("\nProcessing classification datasets...")
    for dataset_name, dataset_class in classification_datasets:
        print(f"  Loading {dataset_name}...")
        
        stats, dataset = get_dataset_stats(
            dataset_class, 
            DATASET_PATHS[dataset_name], 
            dataset_name,
            is_segmentation=False
        )
        
        if dataset is not None:
            all_stats.append(stats)
            create_classification_figure(dataset, stats, dataset_name)
    
    # Process segmentation datasets
    print("\nProcessing segmentation datasets...")
    for dataset_name, dataset_class in segmentation_datasets:
        print(f"  Loading {dataset_name}...")
        
        stats, dataset = get_dataset_stats(
            dataset_class,
            DATASET_PATHS[dataset_name],
            dataset_name,
            is_segmentation=True
        )
        
        if dataset is not None:
            all_stats.append(stats)
            create_segmentation_figure(dataset, stats, dataset_name)
    
    # Print summary table
    print_stats_table(all_stats)
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
