import time, gc
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import json
from typing import Dict, List, Tuple, Optional, Union
from torchvision import transforms
from PIL import Image
import pandas as pd
import cv2
from datetime import datetime
import pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, precision_recall_curve, average_precision_score
from scipy.optimize import minimize_scalar
import scipy

from skimage.morphology import remove_small_objects, label
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes



# Import custom modules
from utils import set_seed, load_dino_backbone, WarmupDecayScheduler

from datasets import (
    MHISTDataset, CRCDataset, PCamDataset,
    PanNukeDataset, MonuSegDataset,
    BRACSDataset, MiDOGDataset,
    GLASDataset, BCSSDataset
)

from datasets import (SynchronizedTransform, DinoTransforms)
from models import CellViT, CellViTMultiClass
from utils import (calculate_rankme_metric, calculate_clid_metric, 
                  calculate_lidar_metric, calculate_alphareq_metric)


import torch.multiprocessing as mp


#############################################################
# Utility Functions
#############################################################

def optimize_class_threshold_youdens_j_index(y_true_binary, y_pred_proba):
    """
    Optimize classification threshold using Youden's J-index
    
    Args:
        y_true_binary: Binary ground truth labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        optimal_threshold: Threshold that maximizes Youden's J-index
    """
    def objective(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        j_index = sensitivity + specificity - 1
        return -j_index  # We negate because we want to maximize J index

    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x

#############################################################
# Feature Extraction Classes and Functions
#############################################################

class FeatureExtractor:
    """
    Extracts features from images using a backbone model.
    """
    def __init__(self, backbone, device, feature_dim=1024):
        self.backbone = backbone
        self.backbone.eval()
        self.device = device
        self.feature_dim = feature_dim
        
    @torch.no_grad()
    def extract_features(self, images):
        """Extract features from a batch of images."""
        if isinstance(images, list):
            images = torch.stack(images)
        
        images = images.to(self.device)
        features = self.backbone(images)
        return features
    
    @torch.no_grad()
    def extract_multi_scale_features(self, images):
        """Extract intermediate features at multiple scales."""
        if isinstance(images, list):
            images = torch.stack(images)
            
        images = images.to(self.device)
        features = self.backbone.get_intermediate_layers(images)
        return features



class AugmentationDataset(torch.utils.data.Dataset):
    """Dataset that applies multiple augmentations to each image for representation metrics."""
    def __init__(
        self, 
        dataset, 
        augmentations_per_image,
        global_size,
        local_size,
        n_local_crops,
        global_crop_scale,
        local_crop_scale,
        mean,
        std
    ):
        self.dataset = dataset
        self.augmentations_per_image = augmentations_per_image
        
        # Calculate augmentations per type to get total augmentations_per_image
        types_count = 2 + n_local_crops  # global_1, global_2, and n_local_crops
        augmentations_per_type = max(1, augmentations_per_image // types_count)
        
        self.transforms = DinoTransforms(
            local_size=local_size,
            global_size=global_size,
            local_crop_scale=local_crop_scale,
            global_crop_scale=global_crop_scale,
            n_local_crops=n_local_crops,
            mean=mean,
            std=std,
            augmentations_per_type=augmentations_per_type,
        )
        
        total_augs = augmentations_per_type * (2 + n_local_crops)
        print(f"Generating {total_augs} augmentations per image")
        print(f"- {augmentations_per_type} variations of global_1")
        print(f"- {augmentations_per_type} variations of global_2")
        print(f"- {augmentations_per_type} variations of each of the {n_local_crops} local crops")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, labels = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image if needed
            image = transforms.ToPILImage()(image)
        
        # Get all augmentations for this image
        augmentations = self.transforms(image)
        
        # Debug print to verify we're getting multiple augmentations
        if idx == 0:  # Only for the first image
            print(f"DEBUG: Generated {len(augmentations)} augmentations for image {idx}")
            
        return augmentations,labels

def extract_dataset_features(
    backbone, 
    dataset, 
    batch_size, 
    num_workers, 
    device,
    save_path=None,
    desc="Extracting features",
    large_dataset_threshold=10000  # Parameter to detect large datasets
):
    """
    Extract features from a dataset using the backbone model.
    For large datasets, save features locally to avoid OOM by using chunked processing.
    """
    # Skip if features already exist
    if save_path and os.path.exists(save_path):
        print(f"Features already exist at {save_path}, skipping extraction")
        # Load features from file
        data = torch.load(save_path)
        return data['features'], data['labels']
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    extractor = FeatureExtractor(backbone, device)
    
    # Determine if this is a large dataset that needs special handling
    is_large_dataset = len(dataset) > large_dataset_threshold
    
    # For large datasets, create temp directory for feature chunks
    if is_large_dataset:
        temp_dir = os.path.join(os.path.dirname(save_path) if save_path else '.', 'temp_features')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Keep track of all saved chunk files
        chunk_files = []
        
        all_features = []
        all_labels = []
        chunk_counter = 0
    else:
        # For smaller datasets, store in memory
        features = []
        labels = []
    
    # Use progress bar for tracking
    progress_bar = tqdm(total=len(loader), desc=desc)
    
    # Extract features
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            batch_features = extractor.extract_features(images)
            
            if is_large_dataset:
                # For large datasets, append to list and periodically save to disk
                all_features.append(batch_features.cpu())
                all_labels.append(targets)
                
                # Save to disk every 100 batches to avoid OOM
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                    chunk_counter += 1
                    temp_features_file = os.path.join(temp_dir, f'features_chunk_{chunk_counter}.pt')
                    temp_labels_file = os.path.join(temp_dir, f'labels_chunk_{chunk_counter}.pt')
                    
                    temp_features = torch.cat(all_features, dim=0)
                    temp_labels = torch.cat(all_labels, dim=0)
                    
                    # Save this chunk with unique filenames
                    torch.save(temp_features, temp_features_file)
                    torch.save(temp_labels, temp_labels_file)
                    
                    # Add to list of chunk files
                    chunk_files.append((temp_features_file, temp_labels_file))
                    
                    print(f"Saved chunk {chunk_counter} with {temp_features.shape[0]} features to disk (batch {batch_idx+1}/{len(loader)})")
                    
                    # Clear memory
                    all_features = []
                    all_labels = []
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                # For smaller datasets, keep in memory
                features.append(batch_features.cpu())
                labels.append(targets)
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Process final batches for large datasets if there are any left
    if is_large_dataset and all_features:
        chunk_counter += 1
        temp_features_file = os.path.join(temp_dir, f'features_chunk_{chunk_counter}.pt')
        temp_labels_file = os.path.join(temp_dir, f'labels_chunk_{chunk_counter}.pt')
        
        temp_features = torch.cat(all_features, dim=0)
        temp_labels = torch.cat(all_labels, dim=0)
        
        torch.save(temp_features, temp_features_file)
        torch.save(temp_labels, temp_labels_file)
        
        # Add to list of chunk files
        chunk_files.append((temp_features_file, temp_labels_file))
        
        print(f"Saved final chunk {chunk_counter} with {temp_features.shape[0]} features to disk")
        
        # Clear memory
        all_features = []
        all_labels = []
        gc.collect()
        torch.cuda.empty_cache()
    
    # For large datasets, load and combine all chunks
    if is_large_dataset:
        print(f"Combining {len(chunk_files)} chunks of features...")
        
        # First pass: count total number of samples to pre-allocate tensor
        total_samples = 0
        feature_dim = None
        
        for feat_file, _ in chunk_files:
            chunk_data = torch.load(feat_file, map_location='cpu')
            total_samples += chunk_data.shape[0]
            if feature_dim is None:
                feature_dim = chunk_data.shape[1]
        
        # Pre-allocate tensors for efficiency
        combined_features = torch.zeros((total_samples, feature_dim), dtype=torch.float32)
        combined_labels = torch.zeros(total_samples, dtype=torch.long)
        
        # Second pass: fill the pre-allocated tensors
        start_idx = 0
        for feat_file, label_file in chunk_files:
            chunk_features = torch.load(feat_file, map_location='cpu')
            chunk_labels = torch.load(label_file, map_location='cpu')
            
            end_idx = start_idx + chunk_features.shape[0]
            combined_features[start_idx:end_idx] = chunk_features
            combined_labels[start_idx:end_idx] = chunk_labels
            
            start_idx = end_idx
            
            # Delete chunk files after loading
            os.remove(feat_file)
            os.remove(label_file)
        
        features = combined_features
        labels = combined_labels
    else:
        # For smaller datasets, concatenate in-memory features
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
    
    # Print statistics about the extracted features
    print(f"Extracted features statistics:")
    print(f"  Total samples: {features.shape[0]}")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Label distribution: ", end="")
    
    # Count unique labels and their frequencies
    unique_labels, counts = torch.unique(labels, return_counts=True)
    label_distribution = {int(l.item()): int(c.item()) for l, c in zip(unique_labels, counts)}
    print(label_distribution)
    
    # Save features and labels if path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'features': features, 'labels': labels}, save_path)
    
    # Clean up temporary directory if it exists
    if is_large_dataset and os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    return features, labels


def custom_collate(batch):
    """Custom collate function to properly handle lists of augmentations."""
    # Each item in batch is a tuple (augmentations_list, label) or just augmentations_list
    augmentations_lists = [item[0] if isinstance(item, tuple) else item for item in batch]
    # Return the list of augmentations for each sample
    return augmentations_lists



def extract_augmentation_features(
    backbone, 
    dataset, 
    sample_size,
    augmentations_per_image,
    global_size,
    local_size,
    n_local_crops,
    global_crop_scale,
    local_crop_scale,
    normalize_mean,
    normalize_std,
    batch_size, 
    num_workers, 
    device,
    save_path=None
):
    """
    Extract features for LiDAR metrics with multiple augmentations per image.
    
    Args:
        backbone: The backbone model
        dataset: Base dataset to extract samples from
        sample_size: Number of images to sample
        augmentations_per_image: Target number of total augmentations per image
        global_size: Size of global crops
        local_size: Size of local crops
        n_local_crops: Number of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        normalize_mean: Mean for normalization
        normalize_std: Std for normalization
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        device: Device to use
        save_path: Path to save features
    
    Returns:
        Tensor of features shaped [n_samples, n_augmentations, feature_dim]
    """
    # Skip if features already exist
    if save_path and os.path.exists(save_path):
        print(f"Augmentation features already exist at {save_path}, loading...")
        features = torch.load(save_path)
        print(f"Loaded features shape: {features.shape}")
        
        # Check if loaded features have at least 2 augmentations per image
        # This is required for LiDAR calculation
        if features.shape[1] < 2:
            print(f"WARNING: Loaded features have only {features.shape[1]} augmentations per image.")
            print(f"LiDAR requires at least 2 augmentations per image. Regenerating features...")
        else:
            return features

    # Ensure sample_size is at least 1
    sample_size = max(1, sample_size)
    
    # Ensure augmentations_per_image is at least 2 (LiDAR requirement)
    augmentations_per_image = max(2, augmentations_per_image)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Warning: Empty dataset provided to extract_augmentation_features")
        # Return empty tensor with correct shape
        return torch.zeros((0, augmentations_per_image, backbone.embed_dim), device='cpu')
    
    # Sample a subset of the dataset
    actual_sample_size = min(sample_size, len(dataset))
    indices = random.sample(range(len(dataset)), actual_sample_size)
    subset = torch.utils.data.Subset(dataset, indices)
    
    print(f"Creating augmentation dataset with {len(subset)} images")
    print(f"Target: {augmentations_per_image} augmentations per image")
    
    # Create dataset with multiple augmentations per image
    aug_dataset = AugmentationDataset(
        subset, 
        augmentations_per_image=augmentations_per_image,
        global_size=global_size,
        local_size=local_size,
        n_local_crops=n_local_crops,
        global_crop_scale=global_crop_scale,
        local_crop_scale=local_crop_scale,
        mean=normalize_mean,
        std=normalize_std,
    )
    
    # Create data loader - we process one sample at a time to maintain the sample/augmentation structure
    loader = torch.utils.data.DataLoader(
        aug_dataset,
        batch_size=1,  # Process one image at a time
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    extractor = FeatureExtractor(backbone, device)
    
    # Initialize feature storage in correct format: [n_samples, n_augmentations, feature_dim]
    all_features = []
    
    # Use progress bar
    progress_bar = tqdm(total=len(loader), desc="Extracting augmentation features")
    
    # Extract features preserving sample/augmentation structure
    with torch.no_grad():
        for batch_idx, augmentations_list in enumerate(loader):
            # Since batch_size=1, we get a list with a single item, which is itself a list of augmentations
            augmentations = augmentations_list[0]  # Get the augmentations for this sample
            
            # Debug info for first batch
            if batch_idx == 0:
                print(f"Sample {batch_idx}: Found {len(augmentations)} augmentations")
                if len(augmentations) < augmentations_per_image:
                    print(f"WARNING: Expected {augmentations_per_image} augmentations but got {len(augmentations)}")
                    print("Check the AugmentationDataset and DinoTransforms implementation")
            
            # Check if we have any augmentations
            if len(augmentations) == 0:
                print(f"WARNING: No augmentations for sample {batch_idx}, skipping")
                progress_bar.update(1)
                continue
                
            # Process augmentations in smaller batches to avoid OOM
            sample_features = []
            
            # Handle different types of augmentations (tensors or PIL images)
            try:
                # Group tensors by size to avoid stacking tensors of different dimensions
                size_groups = {}
                
                # Sort augmentations into groups by spatial size
                for aug in augmentations:
                    if isinstance(aug, torch.Tensor):
                        # If it's a tensor, get its shape
                        shape_key = (aug.shape[1], aug.shape[2])  # Using (H, W) as key
                        aug_tensor = aug
                    else:
                        # If it's a PIL image, convert to tensor to get shape
                        aug_tensor = transforms.ToTensor()(aug)
                        shape_key = (aug_tensor.shape[1], aug_tensor.shape[2])  # Using (H, W) as key
                    
                    # Add to appropriate size group
                    if shape_key not in size_groups:
                        size_groups[shape_key] = []
                    size_groups[shape_key].append(aug_tensor)
                
                # Debug info for first batch
                if batch_idx == 0:
                    print(f"Grouped augmentations by size: {', '.join([f'{k[0]}x{k[1]}: {len(v)}' for k, v in size_groups.items()])}")
                
                # Process each size group separately
                for size, group in size_groups.items():
                    # Process in batches
                    for i in range(0, len(group), batch_size):
                        end_idx = min(i + batch_size, len(group))
                        # Now all tensors in this group have the same spatial dimensions
                        sub_batch = torch.stack(group[i:end_idx]).to(device)
                        # Extract features
                        batch_features = extractor.extract_features(sub_batch)
                        # Add features to this sample
                        sample_features.append(batch_features.cpu())
            except Exception as e:
                print(f"Error processing augmentations for sample {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                progress_bar.update(1)
                continue
            
            # Combine all feature tensors for this sample
            if sample_features:
                try:
                    # Concatenate along batch dimension (first dimension)
                    if all(isinstance(f, torch.Tensor) for f in sample_features):
                        if len(sample_features) == 1:
                            sample_tensor = sample_features[0]
                        else:
                            sample_tensor = torch.cat(sample_features, dim=0)
                    else:
                        # Handle case where we have a mix of tensors and other types
                        tensor_features = [f for f in sample_features if isinstance(f, torch.Tensor)]
                        if tensor_features:
                            sample_tensor = torch.cat(tensor_features, dim=0)
                        else:
                            print(f"WARNING: No valid features for sample {batch_idx}, skipping")
                            progress_bar.update(1)
                            continue
                    
                    # Debug for first sample
                    if batch_idx == 0:
                        print(f"First sample features shape: {sample_tensor.shape}")
                    
                    # Add sample dimension and append to all_features
                    all_features.append(sample_tensor.unsqueeze(0))  # Add sample dimension
                except Exception as e:
                    print(f"Error combining features for sample {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Combine all samples into a single tensor
    if all_features:
        try:
            combined_features = torch.cat(all_features, dim=0)  # [n_samples, n_augmentations, feature_dim]
            print(f"Extracted augmentation features shape: {combined_features.shape}")
        except Exception as e:
            print(f"Error combining all features: {e}")
            import traceback
            traceback.print_exc()
            # Return empty tensor as fallback
            combined_features = torch.zeros((0, augmentations_per_image, backbone.embed_dim), device='cpu')
    else:
        print("WARNING: No features were extracted. Check for errors in the extraction process.")
        combined_features = torch.zeros((0, augmentations_per_image, backbone.embed_dim), device='cpu')
    

    
    # Save features if path is provided
    if save_path is not None and combined_features.shape[0] > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(combined_features, save_path)
        print(f"Saved augmentation features to {save_path}")
    
    return combined_features

#############################################################
# Classification Benchmark Functions
#############################################################

def evaluate_classification_dataset_monte_carlo(
    features, 
    labels, 
    num_classes, 
    device,
    seed,
    n_iterations=20,  # Number of Monte Carlo iterations
    test_size=0.2,    # Fixed at 20% test size
    weight_decay_values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    learning_rate=0.1,
    early_stop_patience=10,
    max_epochs=100,
    batch_size=64,
    metrics_file=None
):
    """
    Evaluate classification performance using Monte Carlo cross-validation with class-balanced splits.
    
    Args:
        features: Feature tensor of shape [n_samples, feature_dim]
        labels: Labels tensor of shape [n_samples]
        num_classes: Number of classes
        device: Device to run computations on
        seed: Random seed for reproducibility
        n_iterations: Number of Monte Carlo iterations
        test_size: Fraction of data to use for testing (fixed at 0.2 for 80/20 split)
        weight_decay_values: List of weight decay values to try
        learning_rate: Learning rate for optimizer
        early_stop_patience: Number of epochs with no improvement before stopping
        max_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        metrics_file: Path to save metrics
        
    Returns:
        metrics: Dictionary of metrics with error bars
    """
    # Check if metrics already exist
    if metrics_file and os.path.exists(metrics_file):
        print(f"Monte Carlo metrics already exist at {metrics_file}, loading...")
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize metrics storage
    all_metrics = []
    
    # For logging/tracking progress
    iteration_progress = tqdm(total=n_iterations, desc="Monte Carlo iterations")
    
    # Get class distribution for balanced sampling
    class_indices = {}
    for c in range(num_classes):
        class_indices[c] = (labels == c).nonzero().squeeze(1).cpu().numpy()
    
    # Run Monte Carlo iterations
    for iteration in range(n_iterations):
        # Create class-balanced train/test split indices
        test_indices = []
        train_indices = []
        
        # For each class, take test_size % for test set
        for c in range(num_classes):
            if c in class_indices and len(class_indices[c]) > 0:
                # Shuffle indices for this class
                indices = np.random.permutation(class_indices[c])
                # Calculate test size for this class
                n_test = int(len(indices) * test_size)
                # Ensure at least one sample per class in both sets if possible
                if len(indices) > 1:
                    n_test = max(1, min(n_test, len(indices) - 1))
                    test_indices.extend(indices[:n_test])
                    train_indices.extend(indices[n_test:])
                else:
                    # If only one sample, put it in train set
                    train_indices.extend(indices)
        
        # Convert to tensors for indexing
        test_indices = torch.tensor(test_indices, device='cpu', dtype=torch.long)
        train_indices = torch.tensor(train_indices, device='cpu', dtype=torch.long)
        
        # Skip iteration if not enough samples or classes in either set
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        train_classes = torch.unique(train_labels)
        test_classes = torch.unique(test_labels)
        
        if len(train_classes) < 2 or len(test_classes) < 2:
            print(f"  Warning: Not enough classes in split (train: {len(train_classes)}, test: {len(test_classes)})")
            iteration_progress.update(1)
            continue
        
        # Create train and test sets
        X_train, y_train = features[train_indices], labels[train_indices]
        X_test, y_test = features[test_indices], labels[test_indices]
        
        # Print split statistics
        print(f"\nIteration {iteration+1}/{n_iterations}")
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        unique_train, counts_train = torch.unique(y_train, return_counts=True)
        unique_test, counts_test = torch.unique(y_test, return_counts=True)
        
        print(f"  Train class distribution: {dict(zip(unique_train.cpu().numpy(), counts_train.cpu().numpy()))}")
        print(f"  Test class distribution: {dict(zip(unique_test.cpu().numpy(), counts_test.cpu().numpy()))}")
        
        # Move to device
        X_train, X_test = X_train.to(device), X_test.to(device)
        y_train, y_test = y_train.to(device), y_test.to(device)
        
        # Split train into train/validation for early stopping
        val_size = int(len(X_train) * 0.2)  # 20% of train for validation
        # Ensure balanced validation set
        val_indices = []
        final_train_indices = []
        
        # Get class indices in the training set
        train_class_indices = {}
        for c in range(num_classes):
            train_class_indices[c] = (y_train == c).nonzero().squeeze(1).cpu().numpy()
        
        # For each class, take 20% for validation
        for c in range(num_classes):
            if c in train_class_indices and len(train_class_indices[c]) > 0:
                indices = np.random.permutation(train_class_indices[c])
                n_val = int(len(indices) * 0.2)
                # Ensure at least one sample per class in both sets if possible
                if len(indices) > 1:
                    n_val = max(1, min(n_val, len(indices) - 1))
                    val_indices.extend(indices[:n_val])
                    final_train_indices.extend(indices[n_val:])
                else:
                    # If only one sample, put it in final train set
                    final_train_indices.extend(indices)
        
        # Create final train and validation sets
        if len(val_indices) > 0 and len(final_train_indices) > 0:
            X_train_final = X_train[final_train_indices]
            y_train_final = y_train[final_train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
        else:
            # Fallback to random split if balanced split fails
            indices = torch.randperm(len(X_train))
            X_train_final = X_train[indices[val_size:]]
            y_train_final = y_train[indices[val_size:]]
            X_val = X_train[indices[:val_size]]
            y_val = y_train[indices[:val_size]]
        
        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train_final, y_train_final)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )
        
        # Try different weight decay values
        best_val_metrics = None
        best_model_state = None
        best_weight_decay = None
        
        for weight_decay in weight_decay_values:
            print(f"  Trying weight decay: {weight_decay}")
            
            # Initialize model
            model = nn.Linear(features.shape[1], num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
            # Initialize early stopping
            best_val_acc = 0.0
            patience_counter = 0
            best_model_state_for_wd = None
            
            # Training loop
            for epoch in range(max_epochs):
                # Train
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Validate
                model.eval()
                all_val_outputs = []
                all_val_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        all_val_outputs.append(outputs)
                        all_val_labels.append(batch_y)
                
                all_val_outputs = torch.cat(all_val_outputs)
                all_val_labels = torch.cat(all_val_labels)
                _, predicted = torch.max(all_val_outputs, 1)
                
                # Calculate metrics
                val_acc = (predicted == all_val_labels).float().mean().item()
                
                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state_for_wd = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for this weight decay
            model.load_state_dict(best_model_state_for_wd)
            model.to(device)
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                outputs = model(X_val)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Calculate metrics
                accuracy = (predicted == y_val).float().mean().item()
                
                # Calculate other metrics with error handling
                try:
                    if num_classes == 2:
                        auc = roc_auc_score(y_val.cpu().numpy(), probs[:, 1].cpu().numpy())
                    else:
                        auc = roc_auc_score(
                            y_val.cpu().numpy(), 
                            probs.cpu().numpy(), 
                            multi_class='ovr', 
                            average='weighted'
                        )
                except Exception as e:
                    print(f"    Warning: AUC calculation failed: {e}")
                    auc = None
                
                try:
                    weighted_f1 = f1_score(
                        y_val.cpu().numpy(), 
                        predicted.cpu().numpy(), 
                        average='weighted'
                    )
                except Exception as e:
                    print(f"    Warning: F1 calculation failed: {e}")
                    weighted_f1 = None
                
                print(f"    Validation accuracy: {accuracy:.4f}")
            
            # Update best model across weight decays
            if best_val_metrics is None or accuracy > best_val_acc:
                best_val_metrics = {'val_accuracy': accuracy, 'val_auc': auc, 'val_f1': weighted_f1}
                best_model_state = best_model_state_for_wd
                best_weight_decay = weight_decay
            
            # Clean up
            del model, optimizer
            torch.cuda.empty_cache()
        
        # Skip iteration if no valid model was found
        if best_model_state is None:
            print("  No valid model found for this iteration")
            iteration_progress.update(1)
            continue
        
        print(f"  Best weight decay: {best_weight_decay}, Val Accuracy: {best_val_metrics['val_accuracy']:.4f}")
        
        # Evaluate on test set using best model
        model = nn.Linear(features.shape[1], num_classes).to(device)
        model.load_state_dict(best_model_state)
        model.eval()
        
        # Test set evaluation
        with torch.no_grad():
            all_outputs = []
            all_labels = []
            
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                all_outputs.append(outputs)
                all_labels.append(batch_y)
            
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            probs = F.softmax(all_outputs, dim=1)
            _, predicted = torch.max(all_outputs, 1)
            
            # Calculate metrics
            accuracy = (predicted == all_labels).float().mean().item()
            
            # Calculate other metrics with error handling
            try:
                if num_classes == 2:
                    auc = roc_auc_score(all_labels.cpu().numpy(), probs[:, 1].cpu().numpy())
                else:
                    auc = roc_auc_score(
                        all_labels.cpu().numpy(), 
                        probs.cpu().numpy(), 
                        multi_class='ovr', 
                        average='weighted'
                    )
            except Exception as e:
                print(f"  Warning: Test AUC calculation failed: {e}")
                auc = None
            
            try:
                weighted_f1 = f1_score(
                    all_labels.cpu().numpy(), 
                    predicted.cpu().numpy(), 
                    average='weighted'
                )
            except Exception as e:
                print(f"  Warning: Test F1 calculation failed: {e}")
                weighted_f1 = None
            
            # Calculate confusion matrix
            try:
                cm = confusion_matrix(all_labels.cpu().numpy(), predicted.cpu().numpy())
                cm_list = cm.tolist()
            except Exception as e:
                print(f"  Warning: Confusion matrix calculation failed: {e}")
                cm_list = []
            
            print(f"  Test accuracy: {accuracy:.4f}")
            if auc is not None:
                print(f"  Test AUC: {auc:.4f}")
            if weighted_f1 is not None:
                print(f"  Test F1: {weighted_f1:.4f}")
        
        # Store metrics for this iteration
        iteration_metrics = {
            'iteration': iteration,
            'weight_decay': float(best_weight_decay),
            'val_accuracy': float(best_val_metrics['val_accuracy']),
            'val_auc': float(best_val_metrics['val_auc']) if best_val_metrics['val_auc'] is not None else None,
            'val_f1': float(best_val_metrics['val_f1']) if best_val_metrics['val_f1'] is not None else None,
            'test_accuracy': float(accuracy),
            'test_auc': float(auc) if auc is not None else None,
            'test_f1': float(weighted_f1) if weighted_f1 is not None else None,
            'confusion_matrix': cm_list,
            'n_train': int(len(X_train_final)),
            'n_val': int(len(X_val)),
            'n_test': int(len(X_test))
        }
        
        all_metrics.append(iteration_metrics)
        
        # Clean up
        del model, X_train, y_train, X_test, y_test, X_train_final, y_train_final, X_val, y_val
        del train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save intermediate results
        if metrics_file is not None:
            interim_file = metrics_file.replace('.json', f'_interim_{iteration}.json')
            os.makedirs(os.path.dirname(interim_file), exist_ok=True)
            with open(interim_file, 'w') as f:
                interim_metrics = calculate_aggregate_metrics(all_metrics)
                interim_metrics['completed_iterations'] = iteration + 1
                interim_metrics['all_iterations'] = all_metrics
                json.dump(interim_metrics, f, indent=2)
        
        iteration_progress.update(1)
    
    iteration_progress.close()
    
    # Calculate aggregate metrics
    final_metrics = calculate_aggregate_metrics(all_metrics)
    final_metrics['completed_iterations'] = len(all_metrics)
    final_metrics['attempted_iterations'] = n_iterations
    final_metrics['all_iterations'] = all_metrics
    
    # Save final metrics
    if metrics_file is not None:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    
    return final_metrics


def calculate_aggregate_metrics(all_metrics):
    """
    Calculate aggregate metrics from multiple Monte Carlo iterations.
    
    Args:
        all_metrics: List of metrics dictionaries from Monte Carlo iterations
        
    Returns:
        aggregate_metrics: Dictionary of aggregate metrics with error bars
    """
    if not all_metrics:
        return {}
    
    # Extract metrics
    accuracies = [m['test_accuracy'] for m in all_metrics]
    aucs = [m['test_auc'] for m in all_metrics if m['test_auc'] is not None]
    f1s = [m['test_f1'] for m in all_metrics if m['test_f1'] is not None]
    
    # Calculate statistics
    aggregate_metrics = {
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'ci_95': [float(np.percentile(accuracies, 2.5)), 
                     float(np.percentile(accuracies, 97.5))],
            'values': accuracies
        },
        'auc': {
            'mean': float(np.mean(aucs)) if aucs else None,
            'std': float(np.std(aucs)) if aucs else None,
            'ci_95': [float(np.percentile(aucs, 2.5)), 
                     float(np.percentile(aucs, 97.5))] if aucs else None,
            'values': aucs
        },
        'f1': {
            'mean': float(np.mean(f1s)) if f1s else None,
            'std': float(np.std(f1s)) if f1s else None,
            'ci_95': [float(np.percentile(f1s, 2.5)), 
                     float(np.percentile(f1s, 97.5))] if f1s else None,
            'values': f1s
        }
    }
    
    return aggregate_metrics


def evaluate_classification_dataset_test(
    train_features, 
    train_labels,
    test_features,
    test_labels,
    num_classes, 
    device,
    weight_decay_values=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    learning_rate=0.1,
    early_stop_patience=10,
    max_epochs=100,
    batch_size=64,
    val_split=0.15,
    metrics_file=None
):
    """
    Evaluate classification performance on a test set with early stopping.
    
    Args:
        train_features: Training features
        train_labels: Training labels
        test_features: Test features
        test_labels: Test labels
        num_classes: Number of classes
        device: Device to use
        weight_decay_values: List of weight decay values to try
        learning_rate: Learning rate for optimizer
        early_stop_patience: Number of epochs with no improvement before stopping
        max_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        val_split: Percentage of train data to use for validation
        metrics_file: Path to save metrics
        
    Returns:
        metrics: Dictionary of test metrics
    """
    # Check if metrics already exist
    if metrics_file and os.path.exists(metrics_file):
        print(f"Classification test metrics already exist at {metrics_file}, skipping evaluation")
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Move data to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Split training data into train and validation
    val_size = int(len(train_features) * val_split)
    
    # Use stratified sampling for validation split
    stratified_indices = {}
    for label in torch.unique(train_labels):
        label_indices = (train_labels == label).nonzero().squeeze(1)
        stratified_indices[label.item()] = label_indices
    
    val_indices = []
    train_indices = []
    
    # Get stratified validation indices
    for label, indices in stratified_indices.items():
        n_val = int(len(indices) * val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        val_indices.extend(val_idx.tolist())
        train_indices.extend(train_idx.tolist())
    
    # Create train and validation sets
    X_train_final = train_features[train_indices]
    y_train_final = train_labels[train_indices]
    X_val = train_features[val_indices]
    y_val = train_labels[val_indices]
    
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_final, y_train_final)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    # Try different weight decay values
    best_val_metrics = None
    best_model_state = None
    best_weight_decay = None
    
    for weight_decay in weight_decay_values:
        print(f"Trying weight decay: {weight_decay}")
        
        # Initialize model
        model = nn.Linear(train_features.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Initialize early stopping
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state_for_wd = None
        
        # Training loop
        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            all_val_outputs = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    all_val_outputs.append(outputs)
                    all_val_labels.append(batch_y)
            
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_labels = torch.cat(all_val_labels)
            _, predicted = torch.max(all_val_outputs, 1)
            
            # Calculate metrics
            val_acc = (predicted == all_val_labels).float().mean().item()
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state_for_wd = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for this weight decay
        model.load_state_dict(best_model_state_for_wd)
        model.to(device)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Calculate metrics
            accuracy = (predicted == y_val).float().mean().item()
            
            print(f"  Validation accuracy: {accuracy:.4f}")
        
        # Update best model across weight decays
        if best_val_metrics is None or accuracy > best_val_acc:
            best_val_metrics = {'val_accuracy': accuracy}
            best_model_state = best_model_state_for_wd
            best_weight_decay = weight_decay
    
    print(f"Best weight decay: {best_weight_decay}, Val Accuracy: {best_val_metrics['val_accuracy']:.4f}")
    
    # Load best model
    model = nn.Linear(train_features.shape[1], num_classes).to(device)
    model.load_state_dict(best_model_state)
    
    # Train on full training set with best hyperparameters 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=best_weight_decay
    )
    
    # Create DataLoader for full training set
    full_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    full_train_loader = torch.utils.data.DataLoader(
        full_train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Initialize early stopping for final training
    best_val_acc = best_val_metrics['val_accuracy']
    patience_counter = 0
    best_final_model_state = best_model_state
    
    # Training loop for full training set
    for epoch in range(max_epochs):
        # Train
        model.train()
        for batch_X, batch_y in full_train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val).float().mean().item()
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_final_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping final training at epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(best_final_model_state)
    model.to(device)
    
    # Evaluate best model on test set
    model.eval()
    with torch.no_grad():
        outputs = model(test_features)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate metrics
        accuracy = (predicted == test_labels).float().mean().item()
        
        # For binary classification, find optimal threshold using Youden's J-index
        if num_classes == 2:
            test_labels_np = test_labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()
            auc = roc_auc_score(test_labels_np, probs_np)
            
            # Calculate optimal threshold using Youden's J-index
            optimal_threshold = optimize_class_threshold_youdens_j_index(test_labels_np, probs_np)
            binary_preds = (probs_np >= optimal_threshold).astype(int)
            
            # Calculate metrics with optimal threshold
            tn, fp, fn, tp = confusion_matrix(test_labels_np, binary_preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # PR curve metrics
            precision, recall, _ = precision_recall_curve(test_labels_np, probs_np)
            ap = average_precision_score(test_labels_np, probs_np)
            
            # Calculate F1 with optimal threshold
            weighted_f1 = f1_score(test_labels_np, binary_preds, average='weighted')
            
            # Calculate confusion matrix with optimal threshold
            cm = confusion_matrix(test_labels_np, binary_preds)
            
            metrics = {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'weighted_f1': float(weighted_f1),
                'confusion_matrix': cm.tolist(),
                'best_weight_decay': float(best_weight_decay),
                'val_accuracy': float(best_val_acc),
                'optimal_threshold': float(optimal_threshold),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'average_precision': float(ap)
            }
        else:
            # For multiclass classification
            auc = roc_auc_score(
                test_labels.cpu().numpy(), 
                probs.cpu().numpy(), 
                multi_class='ovr', 
                average='weighted'
            )
            
            # Calculate F1 scores
            weighted_f1 = f1_score(
                test_labels.cpu().numpy(), 
                predicted.cpu().numpy(), 
                average='weighted'
            )
            
            # Calculate confusion matrix
            cm = confusion_matrix(test_labels.cpu().numpy(), predicted.cpu().numpy())
            
            metrics = {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'weighted_f1': float(weighted_f1),
                'confusion_matrix': cm.tolist(),
                'best_weight_decay': float(best_weight_decay),
                'val_accuracy': float(best_val_acc)
            }
    
    # Save metrics if path is provided
    if metrics_file is not None:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


#############################################################
# Segmentation Benchmark Functions
#############################################################

def __proc_np_hv(pred, mask_threshold=0.5, overall_threshold=0.4):
    """
    Process Nuclei Prediction with XY Coordinate Map.
    
    Args:
        pred: Prediction array (mask, h_map, v_map)
        mask_threshold: Threshold for binary mask creation
        overall_threshold: Threshold for overall map processing
    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    v_dir_raw = pred[..., 1]
    h_dir_raw = pred[..., 2]
    
    # Processing
    blb = np.array(blb_raw >= 0.95, dtype=np.int32)  # Threshold for binary mask

    blb = label(blb)
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # Background is 0 already

    # Normalize direction maps
    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # Calculate gradients
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    # Combine gradient maps
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    # Create distance map for watershed
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # Threshold overall map
    overall = np.array(overall >= overall_threshold, dtype=np.int32)

    # Create markers for watershed
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = label(marker)
    marker = remove_small_objects(marker, min_size=3)
    
    # Apply watershed
    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


def __proc_np_dt(pred, mask_threshold=0.5, min_size=10, max_hole_size=50):
    """
    Process predictions using distance transform approach for instance segmentation.

    Args:
        pred: Prediction array (mask_logits, distance_map_1, distance_map_2)
        mask_threshold: Threshold for binary mask creation
        min_size: Minimum object size to keep
        max_hole_size: Maximum hole size to fill

    Returns:
        Instance segmentation map
    """
    from scipy.ndimage import distance_transform_edt, label as scipy_label
    from scipy.ndimage import maximum_filter, binary_fill_holes
    from skimage.morphology import remove_small_objects, label, remove_small_holes
    from skimage.segmentation import watershed

    # Extract predictions
    blb_raw = pred[..., 0]  # Foreground probability
    distance_raw = pred[..., 1]  # Distance transform

    # Create binary mask
    blb = (blb_raw >= mask_threshold).astype(np.uint8)

    # Fill small holes
    blb = remove_small_holes(blb.astype(bool), area_threshold=max_hole_size).astype(np.uint8)

    # Remove small objects
    blb_labeled = label(blb)
    blb_labeled = remove_small_objects(blb_labeled, min_size=min_size)
    blb = (blb_labeled > 0).astype(np.uint8)

    if blb.sum() == 0:
        # No foreground detected
        return np.zeros_like(blb, dtype=np.int32)

    # Use distance transform to find seeds for watershed
    distance_map = distance_raw * blb  # Mask out background

    # Find local maxima in distance map as markers
    # Use a larger footprint for glands (they're bigger than nuclei)
    footprint_size = 15  # Adjust based on typical gland size
    local_max = maximum_filter(distance_map, size=footprint_size)
    markers = (distance_map == local_max) & (distance_map > 0.3)  # Threshold to avoid noise

    # Label markers
    markers_labeled = scipy_label(markers)[0]

    if markers_labeled.max() == 0:
        # No markers found, return the binary mask as single instance
        return blb.astype(np.int32)

    # Apply watershed using negative distance as elevation
    # Watershed treats the image as a topographic surface
    instance_map = watershed(-distance_map, markers_labeled, mask=blb)

    return instance_map.astype(np.int32)


def aggregated_jaccard_index(true, pred):
    """
    Calculate Aggregated Jaccard Index for nuclear segmentation.
    
    Args:
        true: Ground truth instance segmentation
        pred: Predicted instance segmentation
        
    Returns:
        aji: Aggregated Jaccard Index
    """
    gt_list = np.unique(true)
    pred_list = np.unique(pred)

    gt_list = gt_list[1:]  # Exclude 0, the background
    pred_list = pred_list[1:]  # Exclude 0, the background

    n_gt_nuclei = len(gt_list)
    n_pred_nuclei = len(pred_list)

    # Initialize intersection and union pixel counts
    overall_correct_count = 0
    union_pixel_count = 0

    # Create a list to track used predicted nuclei
    pred_used = np.zeros(n_pred_nuclei)

    while len(gt_list) > 0:
        # Ground truth nuclei
        gt = (true == gt_list[-1]).astype(float)
        
        # Compute JI of each gt with matched predicted nuclei
        predicted_match = gt * pred
        
        if np.sum(predicted_match) == 0:
            # No predicted nuclei for this gt (false negative)
            union_pixel_count += np.sum(gt)
            gt_list = gt_list[:-1]
        else:
            # Find best matching predicted nuclei
            predicted_nuc_index = np.unique(predicted_match)
            predicted_nuc_index = predicted_nuc_index[1:]  # exclude 0
            
            best_match = None
            best_ji = 0
            
            for idx in predicted_nuc_index:
                matched = (pred == idx).astype(float)
                nJI = np.sum(np.logical_and(gt, matched)) / np.sum(np.logical_or(gt, matched))
                if nJI > best_ji:
                    best_match = idx
                    best_ji = nJI
            
            predicted_nuclei = (pred == best_match).astype(float)
            
            # Update intersection and union pixel counts
            overall_correct_count += np.sum(np.logical_and(gt, predicted_nuclei))
            union_pixel_count += np.sum(np.logical_or(gt, predicted_nuclei))
            
            # Remove used gt from the list
            gt_list = gt_list[:-1]
            
            # Mark predicted nuclei as used
            idx_in_list = np.where(pred_list == best_match)[0]
            if idx_in_list.size > 0:
                pred_used[idx_in_list[0]] += 1

    # Add all unmatched pixels left in the predicted map to union set
    for idx in pred_list[pred_used == 0]:
        unused_nuclei = (pred == idx).astype(float)
        union_pixel_count += np.sum(unused_nuclei)

    if union_pixel_count > 0:  # If this is not an empty image
        aji = overall_correct_count / union_pixel_count
    else:
        aji = 0  # or np.nan if preferred

    return aji


def evaluate_segmentation_dataset(
    backbone, 
    train_dataset,
    test_dataset,
    batch_size, 
    num_workers, 
    device,
    feature_dim,
    magnification,
    save_path=None,
    val_split=0.2,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stop_patience=10,
    max_epochs=50,
    metrics_file=None
):
    """
    Train CellViT on training data and evaluate segmentation performance on test data.
    Now using CombinedLoss for better training stability and performance.
    """
    
    # First, define the CombinedLoss class
    class CombinedLoss(nn.Module):
        def __init__(self, w_xentropy=1.0, w_dice=1.0, w_mse=1.0, w_msge=1.0, w_ftversky=1.0):
            super(CombinedLoss, self).__init__()
            self.w_dice = w_dice
            self.w_mse = w_mse
            self.w_msge = w_msge
            self.w_ftversky = w_ftversky
            self.w_xentropy = w_xentropy

        def forward(self, true_mask, pred_mask, true_dist, pred_dist):
            # Cross-entropy loss
            xentropy = self.xentropy_loss(true_mask, pred_mask)
            # Dice loss
            dice = self.dice_loss(true_mask, pred_mask)
            # Mean squared error loss
            mse = self.mse_loss(true_dist, pred_dist)
            # Mean squared gradient error loss
            msge = self.msge_loss(true_dist, pred_dist, true_mask)
            # Focal Tversky loss
            focal_tversky = self.focal_tversky_loss(true_mask, pred_mask)
            # Channel consistency loss
            consistency = self.channel_consistency_loss(pred_mask)

            # Weighted sum of losses
            loss = (
                self.w_xentropy * xentropy
                + self.w_ftversky * focal_tversky
                + self.w_dice * dice
                + self.w_mse * mse
                + self.w_msge * msge
                + 1.0 * consistency
            )

            return loss

        def xentropy_loss(self, true, pred, reduction="mean"):
            epsilon = 1e-7
            
            # Ensure pred and true are in the correct format
            if pred.dim() == 4:
                pred = pred.permute(0, 2, 3, 1)  # NCHW to NHWC
                true = true.permute(0, 2, 3, 1)  # NCHW to NHWC
            
            # Apply softmax to pred
            pred = F.softmax(pred, dim=-1)
            
            # Clamp pred to avoid log(0)
            pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
            
            # Compute cross entropy
            loss = -torch.sum(true * torch.log(pred), dim=-1)
            
            # Apply reduction
            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss

        def dice_loss(self, true, pred, smooth=1e-5):
            loss = 0
            weights = [0.7, 0.3]  # More weight to foreground
            for channel in range(true.shape[1]):
                inse = torch.sum(pred[:, channel] * true[:, channel], (1, 2))
                l = torch.sum(pred[:, channel], (1, 2))
                r = torch.sum(true[:, channel], (1, 2))
                loss += weights[channel] * (1.0 - (2.0 * inse + smooth) / (l + r + smooth))
            return loss.mean()
        
        def channel_consistency_loss(self, pred):
            pred_softmax = F.softmax(pred, dim=1)
            return F.mse_loss(pred_softmax.sum(dim=1), torch.ones_like(pred_softmax[:, 0]))

        def mse_loss(self, true, pred):
            loss = pred - true
            loss = (loss * loss).mean()
            return loss

        def focal_tversky_loss(self, true, pred):
            alpha_t = 0.7
            beta_t = 0.3
            gamma_f = 4 / 3
            smooth = 1e-6
            loss = 0
            
            for channel in range(true.shape[1]):
                target = true[:, channel].reshape(-1)
                inp = F.softmax(pred[:, channel], dim=1).reshape(-1)
                tp = (inp * target).sum()
                fp = ((1 - target) * inp).sum()
                fn = (target * (1 - inp)).sum()
                
                Tversky = (tp + smooth) / (tp + alpha_t * fn + beta_t * fp + smooth)
                loss += (1 - Tversky) ** gamma_f
            
            return loss / true.shape[1]

        def msge_loss(self, true, pred, focus):
            def get_sobel_kernel(size):
                assert size % 2 == 1, "Must be odd, get size=%d" % size
                h_range = torch.arange(
                    -size // 2 + 1,
                    size // 2 + 1,
                    dtype=torch.float32,
                    device=true.device,
                    requires_grad=False,
                )
                v_range = torch.arange(
                    -size // 2 + 1,
                    size // 2 + 1,
                    dtype=torch.float32,
                    device=true.device,
                    requires_grad=False,
                )
                h, v = torch.meshgrid(h_range, v_range, indexing='ij')
                kernel_h = h / (h * h + v * v + 1.0e-15)
                kernel_v = v / (h * h + v * v + 1.0e-15)
                return kernel_h, kernel_v

            def get_gradient_hv(hv):
                kernel_h, kernel_v = get_sobel_kernel(5)
                kernel_h = kernel_h.view(1, 1, 5, 5)
                kernel_v = kernel_v.view(1, 1, 5, 5)
                h_ch = hv[..., 0].unsqueeze(1)
                v_ch = hv[..., 1].unsqueeze(1)
                h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
                v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
                dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
                dhv = dhv.permute(0, 2, 3, 1).contiguous()
                return dhv
            
            # Rearrange tensors for gradient computation
            pred = pred.permute(0, 2, 3, 1)
            true = true.permute(0, 2, 3, 1)
            focus = focus.permute(0, 2, 3, 1)
            focus = focus[..., 0]  # Pick nuclei channel for focus region
            
            focus = (focus[..., None]).float()
            focus = torch.cat([focus, focus], axis=-1)
            true_grad = get_gradient_hv(true)
            pred_grad = get_gradient_hv(pred)
            loss = pred_grad - true_grad
            loss = focus * (loss * loss)

            # Avoid division by very small numbers
            denominator = focus.sum() + 1e-8
            if denominator.item() < 1e-7:
                return torch.tensor(0.0, device=loss.device)

            loss = loss.sum() / denominator
            return loss

    # Check if metrics already exist
    if metrics_file and os.path.exists(metrics_file):
        print(f"Segmentation metrics already exist at {metrics_file}, skipping evaluation")
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Split training dataset into train and validation sets
    train_size = len(train_dataset)
    val_size = int(train_size * val_split)
    
    # Use indices to split dataset
    indices = list(range(train_size))
    
    # Shuffle indices 
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create DataLoaders with larger batch size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,  # Fixed batch size as in working version
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,  # Fixed batch size
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,  # Fixed batch size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize CellViT model
    model = CellViT(backbone, encoder_dim=feature_dim, drop_rate=0.2).to(device)
    
    # Use CombinedLoss with proper weights
    criterion = CombinedLoss(
        w_xentropy=1.0,
        w_dice=1.0,
        w_mse=2.5,      # Higher weight for distance MSE
        w_msge=8.0,     # Much higher weight for gradient error
        w_ftversky=0.0  # Disabled in working version
    ).to(device)
    
    # Use AdamW optimizer with lower learning rate
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-8,  # Start with very low LR for warmup
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    # Define scheduler
    warmup_epochs = 5
    base_lr = learning_rate      # Peak learning rate after warmup (use function parameter)
    final_lr = learning_rate / 10 # Final LR is 1/10th of base (so if base is 5e-5, final is 5e-6)
    warmup_start_lr = learning_rate / 100  # Start warmup at 1/100th of base

    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs, 
        total_epochs=max_epochs, 
        base_lr=base_lr, 
        final_lr=final_lr,
        warmup_start_lr=warmup_start_lr
    )
    
    # Initialize early stopping
    best_val_aji = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(max_epochs):
        print(f"  Epoch {epoch+1}/{max_epochs}")
        
        # Train - set model to train mode but keep encoder in eval mode
        model.train()
        model.encoder.eval()
        
        train_progress = tqdm(total=len(train_loader), desc="  Training")
        running_loss = 0.0
        
        for batch_idx, (images, masks, distance_maps, instance_masks) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            masks = masks.float().to(device)
            distance_maps = distance_maps.float().to(device)
            
            # Forward pass
            outputs = model(images, magnification)
            
            # Calculate loss using CombinedLoss
            loss = criterion(
                masks,                # true_mask
                outputs['masks'],     # pred_mask
                distance_maps,        # true_dist
                outputs['distances']  # pred_dist
            )
            
            running_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=0.3
            )
            
            optimizer.step()
            
            train_progress.update(1)
            train_progress.set_postfix({'loss': loss.item()})
            
            # Log individual loss components periodically
            if batch_idx % 50 == 0:
                print(f"\n  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        train_progress.close()
        print(f"  Training loss: {avg_loss:.4f}")
        
        # Validate
        model.eval()
        
        val_progress = tqdm(total=len(val_loader), desc="  Validating")
        
        val_metrics = {
            'aji': []
        }
        
        with torch.no_grad():
            for batch_idx, (images, masks, distance_maps, instance_masks) in enumerate(val_loader):
                # Move to device
                images = images.to(device)
                
                # Forward pass
                outputs = model(images, magnification)
                
                # Get predictions
                pred_masks = outputs['masks']
                pred_distances = outputs['distances']
                
                # Calculate AJI
                for i in range(images.shape[0]):
                    # Convert to numpy
                    pred_mask_np = pred_masks[i].cpu().numpy()
                    pred_distance_np = pred_distances[i].cpu().numpy()
                    
                    # Process predictions
                    pred_np = np.stack([
                        pred_mask_np[0], 
                        pred_distance_np[0], 
                        pred_distance_np[1]
                    ], axis=-1)
                    
                    # Convert instance masks to numpy
                    instance_mask_np = instance_masks[i].squeeze().cpu().numpy()
                    
                    # Invert the ground truth instance map
                    # Set background pixel value to 0
                    instance_mask_np = np.max(instance_mask_np) - instance_mask_np
                    
                    if dataset_name == 'GLAS':
                        pred_instance_map = __proc_np_dt(pred_np)
                    else:
                        pred_instance_map = __proc_np_hv(pred_np)
                    
                    # Calculate AJI
                    aji = aggregated_jaccard_index(instance_mask_np, pred_instance_map)
                    val_metrics['aji'].append(aji)
                
                val_progress.update(1)
        
        val_progress.close()
        
        # Calculate mean AJI for validation
        val_aji_mean = np.mean(val_metrics['aji'])
        
        print(f"  Validation AJI: {val_aji_mean:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping check
        if val_aji_mean > best_val_aji:
            best_val_aji = val_aji_mean
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    # Evaluate trained model on test set
    print("Evaluating best model on test set...")
    
    # Switch to evaluation mode for the entire model
    model.eval()
    
    # Initialize metrics
    metrics = {
        'aji': []
    }
    
    # Process batches with tqdm progress bar
    progress_bar = tqdm(total=len(test_loader), desc="Evaluating on test set")
    
    # Process batches
    with torch.no_grad():
        for batch_idx, (images, masks, distance_maps, instance_masks) in enumerate(test_loader):
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images, magnification)
            
            # Get predictions
            pred_masks = outputs['masks']
            pred_distances = outputs['distances']
            
            # Calculate AJI
            for i in range(images.shape[0]):
                # Convert to numpy
                pred_mask_np = pred_masks[i].cpu().numpy()
                pred_distance_np = pred_distances[i].cpu().numpy()
                
                # Process predictions
                pred_np = np.stack([
                    pred_mask_np[0], 
                    pred_distance_np[0], 
                    pred_distance_np[1]
                ], axis=-1)
                
                # Convert instance masks to numpy
                instance_mask_np = instance_masks[i].squeeze().cpu().numpy()

                # Invert the ground truth instance map
                # Set background pixel value to 0
                instance_mask_np = np.max(instance_mask_np) - instance_mask_np
                
                # Generate instance segmentation
                if dataset_name == 'GLAS':
                    pred_instance_map = __proc_np_dt(pred_np)
                else:
                    pred_instance_map = __proc_np_hv(pred_np)
                
                # Calculate AJI
                aji = aggregated_jaccard_index(instance_mask_np, pred_instance_map)
                metrics['aji'].append(aji)
                
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate final statistics
    metrics['weight_decay'] = float(weight_decay)
    metrics['val_aji'] = float(best_val_aji)
    metrics['aji_mean'] = float(np.mean(metrics['aji']))
    metrics['aji_std'] = float(np.std(metrics['aji']))
    metrics['aji_median'] = float(np.median(metrics['aji']))
    metrics['aji_q1'] = float(np.percentile(metrics['aji'], 25))
    metrics['aji_q3'] = float(np.percentile(metrics['aji'], 75))
    metrics['aji_min'] = float(np.min(metrics['aji']))
    metrics['aji_max'] = float(np.max(metrics['aji']))
    metrics['aji_ci_lower'] = np.percentile(metrics['aji'], 2.5)
    metrics['aji_ci_upper'] = np.percentile(metrics['aji'], 97.5)
    
    # Save metrics
    if metrics_file is not None:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics



#############################################################
# Task-Agnostic Representation Metrics
#############################################################


class AdvancedConvergenceTesting:
    """
    Class for advanced convergence testing of representation metrics.
    Uses consistent bootstrapping for all statistics to avoid CI/mean inconsistency.
    """
    def __init__(
        self,
        metric_name,
        start_size=5000,
        step_size=1000,
        min_steps=5,
        convergence_threshold=1e-3,
        confidence_level=0.95,
        bootstrap_samples=100,
        max_features=None
    ):
        self.metric_name = metric_name
        self.start_size = start_size
        self.step_size = step_size
        self.min_steps = min_steps
        self.convergence_threshold = convergence_threshold
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.max_features = max_features
        
        # Initialize storage for results
        self.convergence_results = []
        self.bootstrap_statistics = []  # Will store all bootstrap stats for each step
        self.is_converged = False
        self.converged_at_size = 0
        
        # Set seed for reproducibility
        self.rng = np.random.RandomState(42)
    
    def _run_bootstrap(self, metric_fn, features, sample_size, n_samples):
        """Run bootstrap to calculate mean, std, and CIs all from the same distribution."""
        from joblib import Parallel, delayed
        
        # Define standalone functions for parallelization
        def sample_and_compute_regular(features, metric_fn, random_seed, sample_size):
            rng = np.random.RandomState(random_seed)
            indices = rng.choice(sample_size, size=sample_size, replace=True)
            bootstrap_sampled = features[:sample_size][indices]
            return metric_fn(bootstrap_sampled)
        
        def sample_and_compute_dict(features, metric_fn, random_seed, sample_size):
            rng = np.random.RandomState(random_seed)
            bootstrap_sampled = {}
            for key, feats in features.items():
                effective_size = min(sample_size, len(feats))
                if effective_size > 0:
                    indices = rng.choice(effective_size, size=effective_size, replace=True)
                    bootstrap_sampled[key] = feats[:effective_size][indices]
                else:
                    bootstrap_sampled[key] = feats
            return metric_fn(bootstrap_sampled)
        
        # Generate random seeds for reproducibility
        random_seeds = [self.rng.randint(0, 2**32-1) for _ in range(n_samples)]
        
        # Run bootstrap samples in parallel
        if isinstance(features, dict):
            bootstrap_values = Parallel(n_jobs=-1)(
                delayed(sample_and_compute_dict)(features, metric_fn, seed, sample_size) 
                for seed in random_seeds
            )
        else:
            bootstrap_values = Parallel(n_jobs=-1)(
                delayed(sample_and_compute_regular)(features, metric_fn, seed, sample_size) 
                for seed in random_seeds
            )
        
        # Calculate statistics from bootstrap distribution
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_mean = np.mean(bootstrap_values)
        bootstrap_std = np.std(bootstrap_values)
        
        # Calculate percentile-based confidence intervals
        sorted_values = np.sort(bootstrap_values)
        lower_idx = int((1 - self.confidence_level) / 2 * len(sorted_values))
        upper_idx = int((1 - (1 - self.confidence_level) / 2) * len(sorted_values)) - 1
        ci_lower = sorted_values[lower_idx]
        ci_upper = sorted_values[upper_idx]
        
        return {
            'mean': bootstrap_mean,
            'std': bootstrap_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'values': bootstrap_values
        }
    
    def _check_convergence(self):
        """Check if bootstrap means and CIs have converged across steps."""
        if len(self.bootstrap_statistics) <= self.min_steps:
            return False
        
        # Check stability of bootstrap means
        recent_means = [stats['mean'] for stats in self.bootstrap_statistics[-self.min_steps:]]
        mean_stability = np.std(recent_means)
        
        # Check stability of confidence intervals
        recent_ci_lowers = [stats['ci_lower'] for stats in self.bootstrap_statistics[-self.min_steps:]]
        recent_ci_uppers = [stats['ci_upper'] for stats in self.bootstrap_statistics[-self.min_steps:]]
        
        ci_range_stability = 0
        if len(recent_ci_lowers) > 1 and len(recent_ci_uppers) > 1:
            ci_lower_stability = max(abs(np.diff(recent_ci_lowers)))
            ci_upper_stability = max(abs(np.diff(recent_ci_uppers)))
            ci_range_stability = ci_lower_stability + ci_upper_stability
        
        return mean_stability < self.convergence_threshold and ci_range_stability < self.convergence_threshold
    
    def test_convergence(self, metric_fn, features):
        """Test for convergence with consistent bootstrap statistics at each step."""
        # Limit features if needed
        if self.max_features is not None and isinstance(features, torch.Tensor) and len(features) > self.max_features:
            indices = torch.randperm(len(features))[:self.max_features]
            features = features[indices]
        
        # Generate sample sizes
        max_features = len(features)
        sample_sizes = range(self.start_size, max_features + self.step_size, self.step_size)
        
        # Ensure we have at least one sample size
        if not list(sample_sizes):
            sample_sizes = [min(self.start_size, max_features)]
        
        progress_bar = tqdm(total=len(sample_sizes), desc=f"Testing convergence for {self.metric_name}")
        
        for n in sample_sizes:
            # Ensure we don't exceed available features
            n = min(n, max_features)
            
            # Run bootstrap to get statistics from same distribution
            bootstrap_stats = self._run_bootstrap(metric_fn, features, n, self.bootstrap_samples)
            
            # Store non-bootstrapped result for backward compatibility and direct value
            direct_value = metric_fn(features[:n])
            self.convergence_results.append((n, direct_value))
            
            # Store bootstrap statistics
            self.bootstrap_statistics.append(bootstrap_stats)
            
            progress_bar.update(1)
            
            # Check for convergence
            if self._check_convergence():
                self.is_converged = True
                self.converged_at_size = n
                print(f"{self.metric_name} converged after {len(self.convergence_results)} steps (sample size {n})")
                print(f"  Final bootstrap value: {bootstrap_stats['mean']:.4f}, {self.confidence_level*100:.0f}% CI: "
                      f"[{bootstrap_stats['ci_lower']:.4f}, {bootstrap_stats['ci_upper']:.4f}]")
                
                # Exit the loop if converged
                break
        
        progress_bar.close()
        
        # Use the final bootstrap statistics for the result
        final_stats = self.bootstrap_statistics[-1]
        
        # Return results - both direct and bootstrap values for comparison
        return {
            'final_value': final_stats['mean'],  # Use bootstrap mean for consistency with CIs
            'direct_final_value': self.convergence_results[-1][1],  # Original direct calculation
            'convergence': self.convergence_results,
            'bootstrap_statistics': [{
                    'size': self.convergence_results[i][0],
                    'mean': stats['mean'],
                    'std': stats['std'], 
                    'ci_lower': stats['ci_lower'],
                    'ci_upper': stats['ci_upper'],
                    'values': stats['values'].tolist()  # ADD THIS LINE
                } for i, stats in enumerate(self.bootstrap_statistics)],
            'mean': final_stats['mean'],
            'std': final_stats['std'],
            'ci_lower': final_stats['ci_lower'],
            'ci_upper': final_stats['ci_upper'],
            'converged': self.is_converged,
            'converged_at_size': self.converged_at_size if self.is_converged else self.convergence_results[-1][0],
            'final_sample_size': self.convergence_results[-1][0]
        }



class AdvancedLiDARConvergenceTesting:
    """
    Class for advanced LiDAR metric convergence testing with consistent statistics.
    Tests convergence across two dimensions: number of images and augmentations per image.
    """
    def __init__(
        self,
        step_size_images=50,
        step_size_augs=5,
        min_steps=3,
        convergence_threshold=1e-3,
        confidence_level=0.95,
        bootstrap_samples=50,
        max_images=None,
        max_augs_per_image=None
    ):
        self.step_size_images = step_size_images
        self.step_size_augs = step_size_augs
        self.min_steps = min_steps
        self.convergence_threshold = convergence_threshold
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.max_images = max_images
        self.max_augs_per_image = max_augs_per_image
        
        # Initialize storage for results
        self.convergence_results = []  # (n_images, n_augs_per_image, direct_value)
        self.bootstrap_statistics = []  # (n_images, n_augs_per_image, stats_dict)
        self.is_converged = False
        self.converged_at = (0, 0)  # (n_images, n_augs_per_image)
        
        # Set seed for reproducibility
        self.rng = np.random.RandomState(42)
        
    def _run_bootstrap(self, metric_fn, features, n_images, n_augs, n_samples):
        """Run bootstrap for LiDAR to calculate mean, std, and CIs from same distribution."""
        from joblib import Parallel, delayed
        
        # Define standalone function for parallelization
        def sample_and_compute(features, metric_fn, random_seed, n_images, n_augs):
            rng = np.random.RandomState(random_seed)
            
            # Sample images with replacement
            image_indices = rng.choice(min(n_images, features.shape[0]), size=n_images, replace=True)
            
            # Create bootstrap sample tensor
            bootstrap_tensor = features[image_indices, :n_augs, :]
            
            # Calculate metric on bootstrap sample
            if bootstrap_tensor.shape[0] > 0:
                return metric_fn(bootstrap_tensor)
            else:
                return 0.0
        
        # Generate random seeds for reproducibility
        random_seeds = [self.rng.randint(0, 2**32-1) for _ in range(n_samples)]
        
        # Run bootstrap samples in parallel
        bootstrap_values = Parallel(n_jobs=-1)(
            delayed(sample_and_compute)(features, metric_fn, seed, n_images, n_augs) 
            for seed in random_seeds
        )
        
        # If no valid bootstrap values, return defaults
        if not bootstrap_values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': None,
                'ci_upper': None,
                'values': []
            }
        
        # Calculate statistics from bootstrap distribution
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_mean = np.mean(bootstrap_values)
        bootstrap_std = np.std(bootstrap_values)
        
        # Calculate percentile-based confidence intervals
        sorted_values = np.sort(bootstrap_values)
        lower_idx = max(0, int((1 - self.confidence_level) / 2 * len(sorted_values)))
        upper_idx = min(len(sorted_values)-1, int((1 - (1 - self.confidence_level) / 2) * len(sorted_values)))
        ci_lower = sorted_values[lower_idx]
        ci_upper = sorted_values[upper_idx]
        
        return {
            'mean': bootstrap_mean,
            'std': bootstrap_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'values': bootstrap_values
        }
    
    def _check_convergence(self):
        """Check if bootstrap means and CIs have converged across steps."""
        if len(self.bootstrap_statistics) <= self.min_steps:
            return False
        
        # Check stability of bootstrap means
        recent_means = [stats['mean'] for stats in self.bootstrap_statistics[-self.min_steps:]]
        mean_stability = np.std(recent_means)
        
        # Check stability of confidence intervals
        recent_ci_lowers = [stats['ci_lower'] for stats in self.bootstrap_statistics[-self.min_steps:] 
                           if stats['ci_lower'] is not None]
        recent_ci_uppers = [stats['ci_upper'] for stats in self.bootstrap_statistics[-self.min_steps:] 
                           if stats['ci_upper'] is not None]
        
        ci_range_stability = 0
        if len(recent_ci_lowers) > 1 and len(recent_ci_uppers) > 1:
            ci_lower_stability = max(abs(np.diff(recent_ci_lowers)))
            ci_upper_stability = max(abs(np.diff(recent_ci_uppers)))
            ci_range_stability = ci_lower_stability + ci_upper_stability
        
        return mean_stability < self.convergence_threshold and ci_range_stability < self.convergence_threshold
    
    def test_convergence(self, metric_fn, features):
        """Test LiDAR convergence with consistent bootstrap statistics across two dimensions."""
        # Convert PyTorch tensor to NumPy if needed
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # Determine dimensions
        if features.shape[0] == 0:
            print("Warning: Empty features tensor for LiDAR calculation")
            return {
                'final_value': 0.0,
                'convergence': [],
                'bootstrap_statistics': [],
                'grid_results': {},
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': None,
                'ci_upper': None,
                'converged': False,
                'converged_at': (0, 0),
                'final_config': (0, 0)
            }
        
        n_total_images, max_augs_per_image, feature_dim = features.shape
        
        # Apply limits if needed
        if self.max_images is not None:
            n_total_images = min(n_total_images, self.max_images)
        
        if self.max_augs_per_image is not None:
            max_augs_per_image = min(max_augs_per_image, self.max_augs_per_image)
        
        # Generate testing grid
        image_steps = range(self.step_size_images, n_total_images + 1, self.step_size_images)
        image_steps = list(image_steps) if image_steps else [min(n_total_images, self.step_size_images)]
        
        aug_steps = range(self.step_size_augs, max_augs_per_image + 1, self.step_size_augs)
        aug_steps = list(aug_steps) if aug_steps else [min(max_augs_per_image, self.step_size_augs)]
        
        total_steps = len(image_steps) * len(aug_steps)
        progress_bar = tqdm(total=total_steps, desc=f"Testing convergence for LiDAR")
        
        # Test different configurations
        grid_results = {}
        for n_images in image_steps:
            grid_results[n_images] = {}
            
            for n_augs in aug_steps:
                # Calculate direct LiDAR value
                if n_images <= features.shape[0]:
                    subset = features[:n_images, :n_augs, :]
                    direct_value = metric_fn(subset)
                else:
                    direct_value = 0.0
                
                # Run bootstrap for consistent statistics
                bootstrap_stats = self._run_bootstrap(
                    metric_fn, features, n_images, n_augs, self.bootstrap_samples
                )
                
                # Store results
                self.convergence_results.append((n_images, n_augs, direct_value))
                self.bootstrap_statistics.append(bootstrap_stats)
                grid_results[n_images][n_augs] = bootstrap_stats['mean']
                
                progress_bar.update(1)
                
                # Check for convergence after each full set of augmentations
                if n_augs == aug_steps[-1] and self._check_convergence():
                    self.is_converged = True
                    self.converged_at = (n_images, n_augs)
                    
                    print(f"LiDAR converged at {n_images} images with {n_augs} augs per image")
                    print(f"  Final bootstrap value: {bootstrap_stats['mean']:.4f}, {self.confidence_level*100:.0f}% CI: "
                          f"[{bootstrap_stats['ci_lower']:.4f}, {bootstrap_stats['ci_upper']:.4f}]")
                    
                    # Exit the loops if converged
                    break
            
            # Exit outer loop if converged
            if self.is_converged:
                break
        
        progress_bar.close()
        
        # Get final statistics
        final_stats = self.bootstrap_statistics[-1] if self.bootstrap_statistics else {
            'mean': 0.0, 'std': 0.0, 'ci_lower': None, 'ci_upper': None
        }
        final_config = self.convergence_results[-1][:2] if self.convergence_results else (0, 0)
        
        # Return results with both direct values and bootstrap statistics
        return {
            'final_value': final_stats['mean'],  # Use bootstrap mean
            'direct_final_value': self.convergence_results[-1][2] if self.convergence_results else 0.0,
            'convergence': self.convergence_results,
            'bootstrap_statistics': [{
                    'images': self.convergence_results[i][0],
                    'augs': self.convergence_results[i][1],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'ci_lower': stats['ci_lower'],
                    'ci_upper': stats['ci_upper'],
                    'values': stats['values'].tolist()
                } for i, stats in enumerate(self.bootstrap_statistics)],            
            'grid_results': grid_results,
            'mean': final_stats['mean'],
            'std': final_stats['std'],
            'ci_lower': final_stats['ci_lower'],
            'ci_upper': final_stats['ci_upper'],
            'converged': self.is_converged,
            'converged_at': self.converged_at if self.is_converged else final_config,
            'final_config': final_config
        }



def calculate_combined_task_agnostic_metrics(
    all_features,
    all_aug_features,
    output_dir,
    start_size=10000,
    step_size=1000,
    convergence_threshold=1e-3,
    convergence_min_steps=5,
    confidence_level=0.95,
    bootstrap_samples=100,
    max_features=50000,
    lidar_step_size_images=10,
    lidar_step_size_augs=5,
    metrics_file=None
):
    """
    Calculate all task-agnostic metrics using combined features from all datasets,
    with improved bootstrap-based convergence testing.
    """
    # Create metrics directory if it doesn't exist
    ta_metrics_dir = os.path.join(output_dir, "task_agnostic_metrics")
    os.makedirs(ta_metrics_dir, exist_ok=True)
    
    # Initialize final metrics structure
    metrics = {}
    
    # Check if complete metrics file already exists
    if metrics_file and os.path.exists(metrics_file):
        print(f"Task-agnostic metrics already exist at {metrics_file}, loading...")
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    print("Calculating combined task-agnostic metrics from all datasets...")
    
    # Combine features for standard metrics
    combined_features = []
    for dataset, features in all_features.items():
        if isinstance(features, torch.Tensor):
            combined_features.append(features)
        elif isinstance(features, dict) and 'features' in features:
            combined_features.append(features['features'])
        else:
            print(f"Warning: Could not process features from {dataset}")
    
    if combined_features:
        combined_features = torch.cat(combined_features, dim=0)
        print(f"Combined features shape: {combined_features.shape}")
    else:
        combined_features = torch.zeros((0, 1024))
        print("Warning: No valid features found")
    
    # 1. Check for existing RankMe metric and calculate if needed
    rankme_file = os.path.join(ta_metrics_dir, "rankme_metric.json")
    if os.path.exists(rankme_file):
        print(f"Loading existing RankMe metric from {rankme_file}")
        with open(rankme_file, 'r') as f:
            rankme_results = json.load(f)
    else:
        # Calculate RankMe with convergence testing
        print("Calculating RankMe metric...")
        try:
            def rankme_wrapper(x):
                return calculate_rankme_metric(x).item()
            
            rankme_tester = AdvancedConvergenceTesting(
                metric_name="RankMe",
                start_size=start_size,   
                step_size=step_size,
                min_steps=convergence_min_steps,
                convergence_threshold=convergence_threshold,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                max_features=max_features
            )
            
            rankme_results = rankme_tester.test_convergence(rankme_wrapper, combined_features)
            
            # Save RankMe results immediately
            with open(rankme_file, 'w') as f:
                json.dump(rankme_results, f, indent=2)
            print(f"RankMe metric saved to {rankme_file}")
            
            # Visualize RankMe convergence
            plt.figure(figsize=(10, 6))
            x = [entry['size'] for entry in rankme_results['bootstrap_statistics']]
            y = [entry['mean'] for entry in rankme_results['bootstrap_statistics']]
            plt.plot(x, y, 'o-', label='RankMe Value (Bootstrap Mean)')
            
            # Add confidence intervals from bootstrap statistics
            ci_lower = [entry['ci_lower'] for entry in rankme_results['bootstrap_statistics']]
            ci_upper = [entry['ci_upper'] for entry in rankme_results['bootstrap_statistics']]
            
            if all(x is not None for x in ci_lower) and all(x is not None for x in ci_upper):
                plt.fill_between(x, ci_lower, ci_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% Confidence Interval')
            
            plt.xlabel('Number of Features')
            plt.ylabel('RankMe Value')
            plt.title('RankMe Convergence Testing')
            if rankme_results['converged']:
                plt.axvline(x=rankme_results['converged_at_size'], color='r', linestyle='--', 
                            label=f'Converged at {rankme_results["converged_at_size"]} features')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(ta_metrics_dir, 'rankme_convergence.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error calculating RankMe: {e}")
            rankme_results = {"error": str(e), "mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "converged": False}
        
        # Clear memory after RankMe calculation
        gc.collect()
        torch.cuda.empty_cache()
    
    # Add to final metrics
    metrics['rankme'] = rankme_results
    
    # 2. Check for existing CLID metric and calculate if needed
    clid_file = os.path.join(ta_metrics_dir, "clid_metric.json")
    if os.path.exists(clid_file):
        print(f"Loading existing CLID metric from {clid_file}")
        with open(clid_file, 'r') as f:
            clid_results = json.load(f)
    else:
        # Calculate CLID with convergence testing
        print("Calculating CLID metric...")
        try:
            def clid_wrapper(x):
                # This wrapper returns a single combined CLID value
                return calculate_clid_metric(x)
            
            clid_tester = AdvancedConvergenceTesting(
                metric_name="CLID",
                start_size=start_size,
                step_size=step_size,
                min_steps=convergence_min_steps,
                convergence_threshold=convergence_threshold,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                max_features=max_features
            )
            
            clid_results = clid_tester.test_convergence(clid_wrapper, combined_features)
            
            # Save CLID results immediately
            with open(clid_file, 'w') as f:
                json.dump(clid_results, f, indent=2)
            print(f"CLID metric saved to {clid_file}")
            
            # Visualize CLID convergence
            plt.figure(figsize=(10, 6))
            x = [entry['size'] for entry in clid_results['bootstrap_statistics']]
            y = [entry['mean'] for entry in clid_results['bootstrap_statistics']]
            plt.plot(x, y, 'o-', label='CLID Value (Bootstrap Mean)')
            
            # Add confidence intervals from bootstrap statistics
            ci_lower = [entry['ci_lower'] for entry in clid_results['bootstrap_statistics']]
            ci_upper = [entry['ci_upper'] for entry in clid_results['bootstrap_statistics']]
            
            if all(x is not None for x in ci_lower) and all(x is not None for x in ci_upper):
                plt.fill_between(x, ci_lower, ci_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% Confidence Interval')
            
            plt.xlabel('Number of Features')
            plt.ylabel('CLID Value')
            plt.title('CLID Convergence Testing')
            if clid_results['converged']:
                plt.axvline(x=clid_results['converged_at_size'], color='r', linestyle='--', 
                            label=f'Converged at {clid_results["converged_at_size"]} features')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(ta_metrics_dir, 'clid_convergence.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error calculating CLID: {e}")
            clid_results = {"error": str(e), "mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "converged": False}

        # Clear memory after CLID calculation
        gc.collect()
        torch.cuda.empty_cache()
    
    # Add to final metrics
    metrics['clid'] = clid_results
    
    # 3. Check for existing Alpha-ReQ metric and calculate if needed
    alphareq_file = os.path.join(ta_metrics_dir, "alphareq_metric.json")
    if os.path.exists(alphareq_file):
        print(f"Loading existing Alpha-ReQ metric from {alphareq_file}")
        with open(alphareq_file, 'r') as f:
            alphareq_results = json.load(f)
    else:
        # Calculate Alpha-ReQ with convergence testing
        print("Calculating Alpha-ReQ metric...")
        try:
            def alphareq_wrapper(x):
                results = calculate_alphareq_metric(x)
                return results['alpha']  # Return only alpha for convergence testing
            
            alphareq_tester = AdvancedConvergenceTesting(
                metric_name="Alpha-ReQ",
                start_size=start_size,
                step_size=step_size,
                min_steps=convergence_min_steps,
                convergence_threshold=convergence_threshold,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                max_features=max_features
            )
            
            alphareq_results = alphareq_tester.test_convergence(alphareq_wrapper, combined_features)
            
            # Calculate final full Alpha-ReQ values on converged sample size
            final_sample_size = alphareq_results['final_sample_size']
            final_results = calculate_alphareq_metric(combined_features[:final_sample_size])
            
            # Merge additional alpha-req metrics with convergence results
            for key, value in final_results.items():
                if key != 'alpha':  # Skip alpha as it's already in alphareq_results
                    alphareq_results[key] = value
            
            # Save Alpha-ReQ results immediately
            with open(alphareq_file, 'w') as f:
                json.dump(alphareq_results, f, indent=2)
            print(f"Alpha-ReQ metric saved to {alphareq_file}")
            
            # Visualize Alpha-ReQ convergence
            plt.figure(figsize=(10, 6))
            x = [entry['size'] for entry in alphareq_results['bootstrap_statistics']]
            y = [entry['mean'] for entry in alphareq_results['bootstrap_statistics']]
            plt.plot(x, y, 'o-', label='Alpha Value (Bootstrap Mean)')
            
            # Add confidence intervals from bootstrap statistics
            ci_lower = [entry['ci_lower'] for entry in alphareq_results['bootstrap_statistics']]
            ci_upper = [entry['ci_upper'] for entry in alphareq_results['bootstrap_statistics']]
            
            if all(x is not None for x in ci_lower) and all(x is not None for x in ci_upper):
                plt.fill_between(x, ci_lower, ci_upper, alpha=0.2, label=f'{confidence_level*100:.0f}% Confidence Interval')
            
            plt.xlabel('Number of Features')
            plt.ylabel('Alpha Value')
            plt.title('Alpha-ReQ Convergence Testing')
            if alphareq_results['converged']:
                plt.axvline(x=alphareq_results['converged_at_size'], color='r', linestyle='--', 
                            label=f'Converged at {alphareq_results["converged_at_size"]} features')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(ta_metrics_dir, 'alphareq_convergence.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error calculating Alpha-ReQ: {e}")
            alphareq_results = {"error": str(e), "mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "converged": False, "r_squared": 0.0}
            
        # Clear memory after Alpha-ReQ calculation
        gc.collect()
        torch.cuda.empty_cache()
    
    # Add to final metrics
    metrics['alphareq'] = alphareq_results
    
    # We no longer need combined_features - clear it to free memory
    del combined_features
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Check for existing LiDAR metric and calculate if needed
    lidar_file = os.path.join(ta_metrics_dir, "lidar_metric.json")
    if os.path.exists(lidar_file):
        print(f"Loading existing LiDAR metric from {lidar_file}")
        with open(lidar_file, 'r') as f:
            lidar_results = json.load(f)
    else:
        # Calculate LiDAR with advanced convergence testing
        print("Calculating LiDAR metric...")
        try:
            # UPDATED: Combine augmentation features for LiDAR
            # Now handling the new format where each feature is a tensor of shape [n_samples, n_augmentations, feature_dim]
            combined_aug_features = []
            for dataset, aug_features in all_aug_features.items():
                # Now aug_features should be a tensor of shape [n_samples, n_augmentations, feature_dim]
                if isinstance(aug_features, torch.Tensor) and aug_features.dim() == 3:
                    combined_aug_features.append(aug_features)
            
            # Concatenate all features along the first dimension (n_samples)
            if combined_aug_features:
                combined_aug_features = torch.cat(combined_aug_features, dim=0)
            else:
                # Create empty tensor with correct shape if no features
                combined_aug_features = torch.zeros((0, 10, 1024), device='cpu')
            
            print(f"Combined features shape for LiDAR: {combined_aug_features.shape}")
            
            # Limit the number of images if needed to avoid memory issues
            max_images = min(500, combined_aug_features.shape[0])
            if max_images < combined_aug_features.shape[0]:
                print(f"Limiting to {max_images} images for LiDAR calculation to avoid memory issues")
                combined_aug_features = combined_aug_features[:max_images]
            
            # Free up some memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Use the new convergence testing class with try-except for safety
            lidar_tester = AdvancedLiDARConvergenceTesting(
                step_size_images=lidar_step_size_images,
                step_size_augs=lidar_step_size_augs,
                min_steps=convergence_min_steps,
                convergence_threshold=convergence_threshold,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                max_images=max_images,  # Add this to limit maximum images
                max_augs_per_image=min(20, combined_aug_features.shape[1])  # Limit augmentations too
            )
            
            lidar_results = lidar_tester.test_convergence(calculate_lidar_metric, combined_aug_features)
            
            # Convert all keys in grid_results to strings to ensure serialization works properly
            if 'grid_results' in lidar_results:
                string_grid_results = {}
                for img_key, img_dict in lidar_results['grid_results'].items():
                    # Ensure img_key is string
                    str_img_key = str(img_key)
                    string_grid_results[str_img_key] = {}
                    
                    for aug_key, value in img_dict.items():
                        # Ensure aug_key is string
                        str_aug_key = str(aug_key)
                        string_grid_results[str_img_key][str_aug_key] = value
                
                lidar_results['grid_results'] = string_grid_results
            
            # Ensure final_config is properly stored as list, not tuple
            if 'final_config' in lidar_results and isinstance(lidar_results['final_config'], tuple):
                lidar_results['final_config'] = list(lidar_results['final_config'])
                
            if 'converged_at' in lidar_results and isinstance(lidar_results['converged_at'], tuple):
                lidar_results['converged_at'] = list(lidar_results['converged_at'])
            
            # Save LiDAR results immediately
            with open(lidar_file, 'w') as f:
                json.dump(lidar_results, f, indent=2)
            print(f"LiDAR metric saved to {lidar_file}")
            
            # Visualize LiDAR convergence grid - FIX: Convert keys properly
            if 'grid_results' in lidar_results:
                grid_results = lidar_results['grid_results']
                
                # Create grid for heatmap using string keys
                image_steps = sorted([int(k) for k in grid_results.keys()])
                
                if image_steps:
                    # Ensure we have the first key as string 
                    first_key = str(image_steps[0])
                    if first_key in grid_results:
                        aug_steps = sorted([int(k) for k in grid_results[first_key].keys()])
                        
                        if aug_steps:
                            lidar_grid = np.zeros((len(image_steps), len(aug_steps)))
                            
                            for i, n_images in enumerate(image_steps):
                                img_key = str(n_images)
                                if img_key in grid_results:
                                    for j, n_augs in enumerate(aug_steps):
                                        aug_key = str(n_augs)
                                        if aug_key in grid_results[img_key]:
                                            lidar_grid[i, j] = grid_results[img_key][aug_key]
                            
                            plt.figure(figsize=(10, 8))
                            plt.imshow(lidar_grid, interpolation='nearest', aspect='auto')
                            plt.colorbar(label='LiDAR Value')
                            plt.xticks(range(len(aug_steps)), aug_steps)
                            plt.yticks(range(len(image_steps)), image_steps)
                            plt.xlabel('Number of Augmentations per Image')
                            plt.ylabel('Number of Images')
                            plt.title('LiDAR Convergence Testing Grid')
                            
                            if lidar_results['converged']:
                                # Find the indices of the convergence point
                                conv_img, conv_aug = lidar_results['converged_at']
                                try:
                                    img_idx = image_steps.index(conv_img)
                                    aug_idx = aug_steps.index(conv_aug)
                                    plt.plot(aug_idx, img_idx, 'rx', markersize=12, 
                                            label=f'Converged at ({conv_img} images, {conv_aug} augs)')
                                    plt.legend()
                                except (ValueError, TypeError) as e:
                                    print(f"Error plotting convergence point: {e}")
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(ta_metrics_dir, 'lidar_convergence_grid.png'), dpi=300, bbox_inches='tight')
                            plt.close()
        
        except Exception as e:
            print(f"Error calculating LiDAR: {e}")
            import traceback
            traceback.print_exc()
            lidar_results = {
                "error": str(e), 
                "mean": 0.0, 
                "std": 0.0, 
                "ci_lower": 0.0, 
                "ci_upper": 0.0, 
                "converged": False,
                "final_config": [0, 0],
                "final_value": 0.0
            }
        
        # Clear memory after LiDAR calculation
        if 'combined_aug_features' in locals():
            del combined_aug_features
        gc.collect()
        torch.cuda.empty_cache()
    
    # Add to final metrics
    metrics['lidar'] = lidar_results
    
    # Save full combined metrics
    if metrics_file is not None:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Create a serializable version of the metrics
        serializable_metrics = {}
        for metric_name, metric_data in metrics.items():
            serializable_metrics[metric_name] = {}
            for key, value in metric_data.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[metric_name][key] = value.tolist()
                elif isinstance(value, list) and any(isinstance(x, tuple) for x in value):
                    # Handle lists of tuples by converting to list of lists
                    serializable_metrics[metric_name][key] = [list(x) if isinstance(x, tuple) else x for x in value]
                elif isinstance(value, tuple):
                    serializable_metrics[metric_name][key] = list(value)
                elif isinstance(value, dict) and any(isinstance(x, torch.Tensor) for x in value.values()):
                    # Handle dictionaries with tensor values
                    serializable_dict = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            serializable_dict[k] = v.tolist()
                        else:
                            serializable_dict[k] = v
                    serializable_metrics[metric_name][key] = serializable_dict
                else:
                    serializable_metrics[metric_name][key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    # Create a summary table with final values - now using bootstrap means consistently
    summary = {
        'RankMe': {
            'Value': rankme_results['mean'],  
            'Converged': rankme_results['converged'],
            'Features Used': rankme_results.get('final_sample_size', 0),
        },
        'CLID': {
            'Value': clid_results['mean'],  
            'Converged': clid_results['converged'],
            'Features Used': clid_results.get('final_sample_size', 0),
        },
        'Alpha-ReQ': {
            'Alpha': alphareq_results['mean'],  
            'R²': alphareq_results.get('r_squared', 0.0),
            'Converged': alphareq_results['converged'],
            'Features Used': alphareq_results.get('final_sample_size', 0),
        },
        'LiDAR': {
            'Value': lidar_results.get('mean', 0.0),  # Use get() to avoid KeyError
            'Converged': lidar_results.get('converged', False),
            'Images Used': lidar_results.get('final_config', [0, 0])[0] if isinstance(lidar_results.get('final_config', [0, 0]), list) else 0,
            'Augmentations per Type': lidar_results.get('final_config', [0, 0])[1] if isinstance(lidar_results.get('final_config', [0, 0]), list) else 0,
        }
    }
    
    # Safeguard all values in summary to ensure they're serializable
    for metric_name, metric_data in summary.items():
        for key, value in metric_data.items():
            if isinstance(value, (np.integer, np.floating)):
                summary[metric_name][key] = float(value)
    
    # Save summary
    summary_path = os.path.join(ta_metrics_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nTask-Agnostic Metrics Summary:")
    print("------------------------------")
    print(f"RankMe: {summary['RankMe']['Value']:.4f} (Converged: {summary['RankMe']['Converged']})")
    print(f"CLID: {summary['CLID']['Value']:.4f} (Converged: {summary['CLID']['Converged']})")
    print(f"Alpha-ReQ: {summary['Alpha-ReQ']['Alpha']:.4f} (R²: {summary['Alpha-ReQ']['R²']:.4f}, Converged: {summary['Alpha-ReQ']['Converged']})")
    print(f"LiDAR: {summary['LiDAR']['Value']:.4f} (Converged: {summary['LiDAR']['Converged']})")
    print("------------------------------")
    
    return metrics



#############################################################
# Main Processing Functions
#############################################################

def extract_and_evaluate_features(
    checkpoint_path, 
    output_dir, 
    datasets_config, 
    device, 
    cfg
):
    """
    Extract features from a single checkpoint without using distributed processing.
    
    Args:
        checkpoint_path: Path to checkpoint
        output_dir: Directory to save outputs
        datasets_config: Configuration for datasets
        device: Device to use (now passed from worker)
        cfg: Configuration dictionary
        
    Returns:
        checkpoint_output_dir: Path to output directory for this checkpoint
    """
    # Extract checkpoint iteration
    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name == 'checkpoint.pth':
        iteration = 'final'
    else:
        iteration = checkpoint_name.split('_')[-1].split('.')[0]
    
    print(f"Processing checkpoint: {checkpoint_path} (Iteration: {iteration})")
    
    # Create output directory
    checkpoint_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(checkpoint_output_dir, "processing_log.txt")
    
    # Check if this checkpoint has been fully processed
    final_metrics_path = os.path.join(checkpoint_output_dir, "all_metrics.json")
    if os.path.exists(final_metrics_path):
        print(f"Checkpoint {iteration} already fully processed, skipping")
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Checkpoint {iteration} already fully processed\n")
        return checkpoint_output_dir
    
    # Load backbone
    try:
        backbone = load_dino_backbone(checkpoint_path, device)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Loaded backbone for iteration {iteration}\n")
    except Exception as e:
        print(f"Error loading backbone: {e}")
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Error loading backbone: {e}\n")
        return checkpoint_output_dir
    
    # Store features for task-agnostic metrics
    all_features = {}
    all_aug_features = {}
    
    # Process classification datasets
    for dataset_name, dataset_config in datasets_config['classification'].items():
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Starting processing of {dataset_name}\n")
        
        try:
            # Create transforms
            transform = transforms.Compose([
                transforms.Resize((cfg['global_size'], cfg['global_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg['normalize_mean'], std=cfg['normalize_std'])
            ])
            
            # Create dataset class mapping

            if dataset_name == 'MHIST':
                dataset_class = MHISTDataset
            elif dataset_name == 'CRC':
                dataset_class = CRCDataset
            elif dataset_name == 'PCam':
                dataset_class = PCamDataset
            elif dataset_name == 'BRACS':
                dataset_class = BRACSDataset
            elif dataset_name == 'MiDOG':
                dataset_class = MiDOGDataset
            else:
                print(f"Unknown classification dataset: {dataset_name}")
                continue
            
            # Load ALL data (train + test + val if available)
            full_dataset = dataset_class(
                root_dir=dataset_config['path'], 
                split='all',  # This will combine all available splits
                transform=transform
            )
            
            # Define features path
            features_path = os.path.join(checkpoint_output_dir, f"{dataset_name}_all_features.pt")
            aug_features_path = os.path.join(checkpoint_output_dir, f"{dataset_name}_augmentation_features.pt")
            
            # Extract features for all data
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Extracting {dataset_name} features (all splits combined)\n")
                
            features, labels = extract_dataset_features(
                backbone, 
                full_dataset, 
                batch_size=cfg['batch_size'], 
                num_workers=cfg['num_workers'], 
                device=device,
                save_path=features_path,
                desc=f"Extracting {dataset_name} features"
            )
            
            # Extract augmentation features for representation metrics
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Extracting {dataset_name} augmentation features\n")
                
            aug_features = extract_augmentation_features(
                backbone,
                full_dataset,
                sample_size=cfg['sample_size'],
                augmentations_per_image=cfg['augmentations_per_image'],
                global_size=cfg['global_size'],
                local_size=cfg['local_size'],
                n_local_crops=cfg['n_local_crops'],
                global_crop_scale=cfg['global_crop_scale'],
                local_crop_scale=cfg['local_crop_scale'],
                normalize_mean=cfg['normalize_mean'],
                normalize_std=cfg['normalize_std'],
                batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                device=device,
                save_path=aug_features_path
            )
            
            # Store for task-agnostic metrics
            all_features[dataset_name] = features
            all_aug_features[dataset_name] = aug_features
            
            # Evaluate using Monte Carlo cross-validation ONLY
            monte_carlo_metrics_file = os.path.join(checkpoint_output_dir, f"{dataset_name}_monte_carlo_metrics.json")
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Running {dataset_name} Monte Carlo evaluation\n")
                
            monte_carlo_metrics = evaluate_classification_dataset_monte_carlo(
                features, 
                labels, 
                num_classes=dataset_config['num_classes'], 
                device=device,
                seed=cfg['seed'],
                n_iterations=cfg.get('monte_carlo_iterations', 20),
                weight_decay_values=cfg['weight_decay_values'],
                learning_rate=cfg['learning_rate'],
                early_stop_patience=cfg['early_stop_patience'],
                max_epochs=cfg['max_epochs'],
                batch_size=cfg['batch_size'],
                metrics_file=monte_carlo_metrics_file
            )
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Completed processing of {dataset_name}\n")

            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Error processing {dataset_name}: {e}\n")
                import traceback
                f.write(traceback.format_exc())
                
            # Continue with next dataset
            continue
        
        # Force cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Process segmentation datasets
    for dataset_name, dataset_config in datasets_config['segmentation'].items():
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Starting processing of {dataset_name} segmentation\n")
        
        try:
            # Create transforms
            transform_settings = {
                "RandomRotate90": {"p": 0.5},
                "HorizontalFlip": {"p": 0.5},
                "VerticalFlip": {"p": 0.5},
                "Downscale": {"p": 0.15, "scale": 0.5},
                "Blur": {"p": 0.2, "blur_limit": 3},
                "ColorJitter": {"p": 0.2},
                "normalize": {"mean": cfg['normalize_mean'], "std": cfg['normalize_std']}
            }
            
            val_transform_settings = {
                "normalize": {"mean": cfg['normalize_mean'], "std": cfg['normalize_std']}
            }
            
            # Determine image size based on magnification
            if dataset_config['magnification'] == '20x':
                img_size = cfg['local_size']  # Use local_size for 20x
            else:  # '40x'
                img_size = cfg['global_size']  # Use global_size for 40x
            
            train_transform = SynchronizedTransform(transform_settings, input_shape=img_size)
            val_transform = SynchronizedTransform(val_transform_settings, input_shape=img_size)
            
            # Create datasets
            if dataset_name == 'PanNuke':
                dataset_class = PanNukeDataset
            elif dataset_name == 'MonuSeg':
                dataset_class = MonuSegDataset
            else:
                print(f"Unknown segmentation dataset: {dataset_name}")
                continue
            
            # Load test dataset
            test_dataset = dataset_class(
                data_dir=dataset_config['path'], 
                split='Test',
                magnification=dataset_config['magnification'],
                transform=val_transform
            )

            train_dataset = dataset_class(
                data_dir=dataset_config['path'], 
                split='Training',
                magnification=dataset_config['magnification'],
                transform=train_transform
            )
            
            # Evaluate segmentation performance
            seg_metrics_file = os.path.join(checkpoint_output_dir, f"{dataset_name}_segmentation_metrics.json")
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Running {dataset_name} segmentation evaluation\n")
                
            seg_metrics = evaluate_segmentation_dataset(
                backbone, 
                train_dataset,                        # Add training dataset parameter
                test_dataset,                         # Pass test dataset as second parameter
                batch_size=cfg['batch_size'] // 8,    # Use smaller batch size for segmentation 
                num_workers=cfg['num_workers'], 
                device=device,
                feature_dim=cfg['feature_dim'],
                magnification=dataset_config['magnification'],
                save_path=None,
                val_split=cfg['seg_val_split'],            # Keep validation split
                learning_rate=cfg['seg_learning_rate'],
                weight_decay=cfg.get('seg_weight_decay', 1e-4),  # Get weight decay or use default
                early_stop_patience=cfg['seg_early_stop_patience'],  # Keep early stopping patience
                max_epochs=cfg['seg_max_epochs'],
                metrics_file=seg_metrics_file
            )
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Completed processing of {dataset_name} segmentation\n")
            
        except Exception as e:
            print(f"Error processing {dataset_name} segmentation: {e}")
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()} - Error processing {dataset_name} segmentation: {e}\n")
                import traceback
                f.write(traceback.format_exc())
                
            # Continue with next dataset
            continue
        
        # Force cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate combined task-agnostic metrics
    ta_metrics_dir = os.path.join(checkpoint_output_dir, "task_agnostic_metrics")
    os.makedirs(ta_metrics_dir, exist_ok=True)
    
    ta_metrics_file = os.path.join(ta_metrics_dir, "task_agnostic_metrics.json")
    
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - Calculating task-agnostic metrics\n")
        
    try:
        ta_metrics = calculate_combined_task_agnostic_metrics(
            all_features=all_features,
            all_aug_features=all_aug_features,
            output_dir=checkpoint_output_dir,
            start_size=cfg['convergence_start_size'],
            step_size=cfg['convergence_step_size'],
            convergence_threshold=cfg['convergence_threshold'],
            convergence_min_steps=cfg['convergence_min_steps'],
            confidence_level=cfg['confidence_level'],
            bootstrap_samples=cfg['bootstrap_samples'],
            max_features=50000,
            lidar_step_size_images=10,
            lidar_step_size_augs=5,
            metrics_file=ta_metrics_file
        )
    except Exception as e:
        print(f"Error calculating task-agnostic metrics: {e}")
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Error calculating task-agnostic metrics: {e}\n")
            import traceback
            f.write(traceback.format_exc())
    
    # Collect and save all metrics
    try:
        all_metrics = {}
        
        # Collect classification metrics
        for dataset_name in datasets_config['classification']:
            all_metrics[dataset_name] = {}
            
            # Monte Carlo metrics
            monte_carlo_metrics_file = os.path.join(checkpoint_output_dir, f"{dataset_name}_monte_carlo_metrics.json")
            if os.path.exists(monte_carlo_metrics_file):
                with open(monte_carlo_metrics_file, 'r') as f:
                    all_metrics[dataset_name]['monte_carlo'] = json.load(f)
            
        
        # Collect segmentation metrics
        for dataset_name in datasets_config['segmentation']:
            all_metrics[dataset_name] = {}
            
            # Segmentation metrics
            seg_metrics_file = os.path.join(checkpoint_output_dir, f"{dataset_name}_segmentation_metrics.json")
            if os.path.exists(seg_metrics_file):
                with open(seg_metrics_file, 'r') as f:
                    all_metrics[dataset_name]['segmentation'] = json.load(f)
        
        # Task-agnostic metrics
        all_metrics['task_agnostic'] = {}
        if os.path.exists(ta_metrics_file):
            with open(ta_metrics_file, 'r') as f:
                all_metrics['task_agnostic'] = json.load(f)
        
        # Save all metrics
        with open(final_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
            
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Successfully saved all metrics\n")
            
    except Exception as e:
        print(f"Error saving all metrics: {e}")
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()} - Error saving all metrics: {e}\n")
            import traceback
            f.write(traceback.format_exc())
    
    # Clean up and return
    del backbone
    torch.cuda.empty_cache()
    gc.collect()
    
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - Completed processing of checkpoint {iteration}\n")
        
    return checkpoint_output_dir



def create_summary_tables(output_dir):
    """
    Create summary tables for all metrics across checkpoints.
    
    Args:
        output_dir: Directory containing checkpoint outputs
    """
    # Create summaries directory
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Find all checkpoint directories
    checkpoint_dirs = []
    for d in os.listdir(output_dir):
        if d.startswith('iteration_') and os.path.isdir(os.path.join(output_dir, d)):
            checkpoint_dirs.append(d)
    
    if not checkpoint_dirs:
        print("No checkpoint directories found for summaries")
        return
    
    # Extract and sort iterations
    iterations = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir == 'iteration_final':
            iterations.append(('final', float('inf'), checkpoint_dir))
        else:
            try:
                iter_num = int(checkpoint_dir.split('_')[1])
                # Keep the original directory name with leading zeros
                iterations.append((checkpoint_dir.split('_')[1], iter_num, checkpoint_dir))
            except:
                # Skip directories that don't match the pattern
                continue

    # Sort by numeric value but keep original string format
    sorted_iterations = [it[2] for it in sorted(iterations, key=lambda x: x[1])]
    
    # Check available metrics
    all_metrics = {}
    for iteration in sorted_iterations:
        metrics_file = os.path.join(output_dir, iteration, "all_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics[iteration] = json.load(f)
    
    if not all_metrics:
        print("No metrics found for summaries")
        return
    
    # Identify all datasets and metrics
    classification_datasets = set()
    segmentation_datasets = set()
    
    for metrics in all_metrics.values():
        for dataset in metrics:
            if dataset != 'task_agnostic':
                if 'monte_carlo' in metrics[dataset]:
                    classification_datasets.add(dataset)
                if 'segmentation' in metrics[dataset]:
                    segmentation_datasets.add(dataset)
    
    # Create classification summary (Monte Carlo only)
    classification_summary = {
        'iterations': sorted_iterations,
        'datasets': {}
    }
    
    for dataset in classification_datasets:
        classification_summary['datasets'][dataset] = {
            'monte_carlo_accuracy': [],
            'monte_carlo_accuracy_ci': [],
            'monte_carlo_auc': [],
            'monte_carlo_auc_ci': [],
            'monte_carlo_f1': [],
            'monte_carlo_f1_ci': []
        }
        
        for iteration in sorted_iterations:
            if iteration in all_metrics and dataset in all_metrics[iteration]:
                metrics = all_metrics[iteration][dataset]
                
                # Monte Carlo metrics only
                if 'monte_carlo' in metrics:
                    mc_metrics = metrics['monte_carlo']
                    
                    # Accuracy
                    if 'accuracy' in mc_metrics and 'mean' in mc_metrics['accuracy']:
                        classification_summary['datasets'][dataset]['monte_carlo_accuracy'].append(
                            mc_metrics['accuracy']['mean']
                        )
                        
                        if 'ci_95' in mc_metrics['accuracy']:
                            classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci'].append(
                                mc_metrics['accuracy']['ci_95']
                            )
                        else:
                            classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci'].append(None)
                    else:
                        classification_summary['datasets'][dataset]['monte_carlo_accuracy'].append(None)
                        classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci'].append(None)
                    
                    # AUC
                    if 'auc' in mc_metrics and 'mean' in mc_metrics['auc'] and mc_metrics['auc']['mean'] is not None:
                        classification_summary['datasets'][dataset]['monte_carlo_auc'].append(
                            mc_metrics['auc']['mean']
                        )
                        
                        if 'ci_95' in mc_metrics['auc'] and mc_metrics['auc']['ci_95'] is not None:
                            classification_summary['datasets'][dataset]['monte_carlo_auc_ci'].append(
                                mc_metrics['auc']['ci_95']
                            )
                        else:
                            classification_summary['datasets'][dataset]['monte_carlo_auc_ci'].append(None)
                    else:
                        classification_summary['datasets'][dataset]['monte_carlo_auc'].append(None)
                        classification_summary['datasets'][dataset]['monte_carlo_auc_ci'].append(None)
                    
                    # F1
                    if 'f1' in mc_metrics and 'mean' in mc_metrics['f1'] and mc_metrics['f1']['mean'] is not None:
                        classification_summary['datasets'][dataset]['monte_carlo_f1'].append(
                            mc_metrics['f1']['mean']
                        )
                        
                        if 'ci_95' in mc_metrics['f1'] and mc_metrics['f1']['ci_95'] is not None:
                            classification_summary['datasets'][dataset]['monte_carlo_f1_ci'].append(
                                mc_metrics['f1']['ci_95']
                            )
                        else:
                            classification_summary['datasets'][dataset]['monte_carlo_f1_ci'].append(None)
                    else:
                        classification_summary['datasets'][dataset]['monte_carlo_f1'].append(None)
                        classification_summary['datasets'][dataset]['monte_carlo_f1_ci'].append(None)
                else:
                    # No Monte Carlo metrics for this iteration
                    classification_summary['datasets'][dataset]['monte_carlo_accuracy'].append(None)
                    classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci'].append(None)
                    classification_summary['datasets'][dataset]['monte_carlo_auc'].append(None)
                    classification_summary['datasets'][dataset]['monte_carlo_auc_ci'].append(None)
                    classification_summary['datasets'][dataset]['monte_carlo_f1'].append(None)
                    classification_summary['datasets'][dataset]['monte_carlo_f1_ci'].append(None)
            else:
                # No metrics for this iteration
                classification_summary['datasets'][dataset]['monte_carlo_accuracy'].append(None)
                classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci'].append(None)
                classification_summary['datasets'][dataset]['monte_carlo_auc'].append(None)
                classification_summary['datasets'][dataset]['monte_carlo_auc_ci'].append(None)
                classification_summary['datasets'][dataset]['monte_carlo_f1'].append(None)
                classification_summary['datasets'][dataset]['monte_carlo_f1_ci'].append(None)
    
    # Save classification summary
    if classification_datasets:
        classification_summary_path = os.path.join(summaries_dir, "classification_summary.json")
        with open(classification_summary_path, 'w') as f:
            json.dump(classification_summary, f, indent=2)
        
        # Create classification summary plots (Monte Carlo only)
        for dataset in classification_datasets:
            # Monte Carlo accuracy plot
            plt.figure(figsize=(10, 6))
            y = classification_summary['datasets'][dataset]['monte_carlo_accuracy']
            x = list(range(len(y)))
            
            # Plot means with error bars
            valid_indices = [i for i, val in enumerate(y) if val is not None]
            
            if valid_indices:
                valid_x = [x[i] for i in valid_indices]
                valid_y = [y[i] for i in valid_indices]
                
                # Extract confidence intervals
                ci_values = classification_summary['datasets'][dataset]['monte_carlo_accuracy_ci']
                valid_ci = [ci_values[i] for i in valid_indices if ci_values[i] is not None]
                valid_ci_x = [valid_x[i] for i, ci in enumerate(valid_ci) if ci is not None]
                
                # Create error bars if we have CIs
                if valid_ci:
                    lower_err = [y - ci[0] for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    upper_err = [ci[1] - y for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    asymmetric_err = [lower_err, upper_err]
                    
                    plt.errorbar(valid_ci_x, 
                                [valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], 
                                yerr=asymmetric_err, 
                                fmt='o-', 
                                capsize=5, 
                                label=f'{dataset} Accuracy (n-fold CV with 95% CI)')
                else:
                    # Plot without error bars if no CIs
                    plt.plot(valid_x, valid_y, 'o-', label=f'{dataset} Accuracy')
            
            plt.xlabel('Checkpoint')
            plt.ylabel('Accuracy')
            plt.title(f'{dataset} Accuracy vs. Checkpoint (n-fold Cross-Validation)')
            plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summaries_dir, f"{dataset}_accuracy.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Monte Carlo AUC plot
            plt.figure(figsize=(10, 6))
            y = classification_summary['datasets'][dataset]['monte_carlo_auc']
            
            # Plot means with error bars
            valid_indices = [i for i, val in enumerate(y) if val is not None]
            
            if valid_indices:
                valid_x = [x[i] for i in valid_indices]
                valid_y = [y[i] for i in valid_indices]
                
                # Extract confidence intervals
                ci_values = classification_summary['datasets'][dataset]['monte_carlo_auc_ci']
                valid_ci = [ci_values[i] for i in valid_indices if ci_values[i] is not None]
                valid_ci_x = [valid_x[i] for i, ci in enumerate(valid_ci) if ci is not None]
                
                # Create error bars if we have CIs
                if valid_ci:
                    lower_err = [y - ci[0] for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    upper_err = [ci[1] - y for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    asymmetric_err = [lower_err, upper_err]
                    
                    plt.errorbar(valid_ci_x, 
                                [valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], 
                                yerr=asymmetric_err, 
                                fmt='o-', 
                                capsize=5, 
                                label=f'{dataset} AUC (n-fold CV with 95% CI)')
                else:
                    # Plot without error bars if no CIs
                    plt.plot(valid_x, valid_y, 'o-', label=f'{dataset} AUC')
            
            plt.xlabel('Checkpoint')
            plt.ylabel('AUC')
            plt.title(f'{dataset} AUC vs. Checkpoint (n-fold Cross-Validation)')
            plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summaries_dir, f"{dataset}_auc.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Monte Carlo F1 plot
            plt.figure(figsize=(10, 6))
            y = classification_summary['datasets'][dataset]['monte_carlo_f1']
            
            # Plot means with error bars
            valid_indices = [i for i, val in enumerate(y) if val is not None]
            
            if valid_indices:
                valid_x = [x[i] for i in valid_indices]
                valid_y = [y[i] for i in valid_indices]
                
                # Extract confidence intervals
                ci_values = classification_summary['datasets'][dataset]['monte_carlo_f1_ci']
                valid_ci = [ci_values[i] for i in valid_indices if ci_values[i] is not None]
                valid_ci_x = [valid_x[i] for i, ci in enumerate(valid_ci) if ci is not None]
                
                # Create error bars if we have CIs
                if valid_ci:
                    lower_err = [y - ci[0] for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    upper_err = [ci[1] - y for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                    asymmetric_err = [lower_err, upper_err]
                    
                    plt.errorbar(valid_ci_x, 
                                [valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], 
                                yerr=asymmetric_err, 
                                fmt='o-', 
                                capsize=5, 
                                label=f'{dataset} F1 Score (n-fold CV with 95% CI)')
                else:
                    # Plot without error bars if no CIs
                    plt.plot(valid_x, valid_y, 'o-', label=f'{dataset} F1 Score')
            
            plt.xlabel('Checkpoint')
            plt.ylabel('F1 Score')
            plt.title(f'{dataset} F1 Score vs. Checkpoint (n-fold Cross-Validation)')
            plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summaries_dir, f"{dataset}_f1.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create segmentation summary (unchanged)
    segmentation_summary = {
        'iterations': sorted_iterations,
        'datasets': {}
    }
    
    for dataset in segmentation_datasets:
        segmentation_summary['datasets'][dataset] = {
            'aji_mean': [],
            'aji_std': [],
            'aji_ci': []
        }
        
        for iteration in sorted_iterations:
            if iteration in all_metrics and dataset in all_metrics[iteration] and 'segmentation' in all_metrics[iteration][dataset]:
                seg_metrics = all_metrics[iteration][dataset]['segmentation']
                segmentation_summary['datasets'][dataset]['aji_mean'].append(
                    seg_metrics.get('aji_mean', None)
                )
                segmentation_summary['datasets'][dataset]['aji_std'].append(
                    seg_metrics.get('aji_std', None)
                )
                
                # Add 95% confidence interval if available
                if 'aji_ci_lower' in seg_metrics and 'aji_ci_upper' in seg_metrics:
                    segmentation_summary['datasets'][dataset]['aji_ci'].append(
                        [seg_metrics['aji_ci_lower'], seg_metrics['aji_ci_upper']]
                    )
                else:
                    segmentation_summary['datasets'][dataset]['aji_ci'].append(None)
            else:
                segmentation_summary['datasets'][dataset]['aji_mean'].append(None)
                segmentation_summary['datasets'][dataset]['aji_std'].append(None)
                segmentation_summary['datasets'][dataset]['aji_ci'].append(None)
    
    # Save segmentation summary
    if segmentation_datasets:
        segmentation_summary_path = os.path.join(summaries_dir, "segmentation_summary.json")
        with open(segmentation_summary_path, 'w') as f:
            json.dump(segmentation_summary, f, indent=2)
        
        # Create segmentation summary plots (unchanged)
        for dataset in segmentation_datasets:
            plt.figure(figsize=(10, 6))
            y = segmentation_summary['datasets'][dataset]['aji_mean']
            yerr = segmentation_summary['datasets'][dataset]['aji_std']
            ci_values = segmentation_summary['datasets'][dataset]['aji_ci']
            x = list(range(len(y)))
            
            # Plot with error bars only where we have valid values
            valid_indices = [i for i, val in enumerate(y) if val is not None]
            
            if valid_indices:
                valid_x = [x[i] for i in valid_indices]
                valid_y = [y[i] for i in valid_indices]
                
                # Use confidence intervals if available, otherwise use std for error bars
                if any(ci is not None for ci in ci_values):
                    valid_ci = [ci_values[i] for i in valid_indices if ci_values[i] is not None]
                    valid_ci_x = [valid_x[i] for i, ci in enumerate(valid_ci) if ci is not None]
                    
                    # Create error bars
                    if valid_ci:
                        lower_err = [y - ci[0] for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                        upper_err = [ci[1] - y for y, ci in zip([valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], valid_ci)]
                        asymmetric_err = [lower_err, upper_err]
                        
                        plt.errorbar(valid_ci_x, 
                                    [valid_y[i] for i, _ in enumerate(valid_ci) if valid_ci[i] is not None], 
                                    yerr=asymmetric_err, 
                                    fmt='o-', 
                                    capsize=5, 
                                    label=f'{dataset} AJI with 95% CI')
                else:
                    # Use standard deviation if no CIs
                    valid_yerr = [yerr[i] for i in valid_indices if yerr[i] is not None]
                    valid_err_x = [valid_x[i] for i, err in enumerate(valid_yerr) if err is not None]
                    
                    if valid_yerr:
                        plt.errorbar(valid_err_x, 
                                    [valid_y[i] for i, _ in enumerate(valid_yerr) if valid_yerr[i] is not None], 
                                    yerr=valid_yerr, 
                                    fmt='o-', 
                                    capsize=5, 
                                    label=f'{dataset} AJI with Std Dev')
                
                # Always plot the mean line
                plt.plot(valid_x, valid_y, 'o-', label=f'{dataset} AJI Mean' if not (any(ci is not None for ci in ci_values) or any(err is not None for err in yerr)) else None)
            
            plt.xlabel('Checkpoint')
            plt.ylabel('AJI (Mean)')
            plt.title(f'{dataset} AJI vs. Checkpoint')
            plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(summaries_dir, f"{dataset}_aji.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    
    task_agnostic_summary = {
        'iterations': sorted_iterations,
        'metrics': {
            'rankme': {'values': [], 'ci_lower': [], 'ci_upper': []},
            'clid': {'values': [], 'ci_lower': [], 'ci_upper': []},
            'alphareq': {'values': [], 'ci_lower': [], 'ci_upper': []},
            'lidar': {'values': [], 'ci_lower': [], 'ci_upper': []}
        }
    }
    
    for iteration in sorted_iterations:
        if iteration in all_metrics and 'task_agnostic' in all_metrics[iteration]:
            ta_metrics = all_metrics[iteration]['task_agnostic']
            
            # RankMe
            if 'rankme' in ta_metrics:
                task_agnostic_summary['metrics']['rankme']['values'].append(
                    ta_metrics['rankme'].get('mean', ta_metrics['rankme'].get('final_value', None))
                )
                task_agnostic_summary['metrics']['rankme']['ci_lower'].append(
                    ta_metrics['rankme'].get('ci_lower', None)
                )
                task_agnostic_summary['metrics']['rankme']['ci_upper'].append(
                    ta_metrics['rankme'].get('ci_upper', None)
                )
            else:
                task_agnostic_summary['metrics']['rankme']['values'].append(None)
                task_agnostic_summary['metrics']['rankme']['ci_lower'].append(None)
                task_agnostic_summary['metrics']['rankme']['ci_upper'].append(None)
            
            # CLID
            if 'clid' in ta_metrics:
                task_agnostic_summary['metrics']['clid']['values'].append(
                    ta_metrics['clid'].get('mean', ta_metrics['clid'].get('final_value', None))
                )
                task_agnostic_summary['metrics']['clid']['ci_lower'].append(
                    ta_metrics['clid'].get('ci_lower', None)
                )
                task_agnostic_summary['metrics']['clid']['ci_upper'].append(
                    ta_metrics['clid'].get('ci_upper', None)
                )
            else:
                task_agnostic_summary['metrics']['clid']['values'].append(None)
                task_agnostic_summary['metrics']['clid']['ci_lower'].append(None)
                task_agnostic_summary['metrics']['clid']['ci_upper'].append(None)
            
            # Alpha-ReQ
            if 'alphareq' in ta_metrics:
                task_agnostic_summary['metrics']['alphareq']['values'].append(
                    ta_metrics['alphareq'].get('mean', ta_metrics['alphareq'].get('final_value', None))
                )
                task_agnostic_summary['metrics']['alphareq']['ci_lower'].append(
                    ta_metrics['alphareq'].get('ci_lower', None)
                )
                task_agnostic_summary['metrics']['alphareq']['ci_upper'].append(
                    ta_metrics['alphareq'].get('ci_upper', None)
                )
            else:
                task_agnostic_summary['metrics']['alphareq']['values'].append(None)
                task_agnostic_summary['metrics']['alphareq']['ci_lower'].append(None)
                task_agnostic_summary['metrics']['alphareq']['ci_upper'].append(None)
            
            # LiDAR
            if 'lidar' in ta_metrics:
                task_agnostic_summary['metrics']['lidar']['values'].append(
                    ta_metrics['lidar'].get('mean', ta_metrics['lidar'].get('final_value', None))
                )
                task_agnostic_summary['metrics']['lidar']['ci_lower'].append(
                    ta_metrics['lidar'].get('ci_lower', None)
                )
                task_agnostic_summary['metrics']['lidar']['ci_upper'].append(
                    ta_metrics['lidar'].get('ci_upper', None)
                )
            else:
                task_agnostic_summary['metrics']['lidar']['values'].append(None)
                task_agnostic_summary['metrics']['lidar']['ci_lower'].append(None)
                task_agnostic_summary['metrics']['lidar']['ci_upper'].append(None)
        else:
            # No task-agnostic metrics for this iteration
            for metric in ['rankme', 'clid', 'alphareq', 'lidar']:
                task_agnostic_summary['metrics'][metric]['values'].append(None)
                task_agnostic_summary['metrics'][metric]['ci_lower'].append(None)
                task_agnostic_summary['metrics'][metric]['ci_upper'].append(None)
    
    # Save task-agnostic summary
    task_agnostic_summary_path = os.path.join(summaries_dir, "task_agnostic_summary.json")
    with open(task_agnostic_summary_path, 'w') as f:
        json.dump(task_agnostic_summary, f, indent=2)
    
    # Create task-agnostic summary plots with error bars
    x = list(range(len(sorted_iterations)))
    
    # RankMe plot
    plt.figure(figsize=(10, 6))
    y = task_agnostic_summary['metrics']['rankme']['values']
    ci_lower = task_agnostic_summary['metrics']['rankme']['ci_lower']
    ci_upper = task_agnostic_summary['metrics']['rankme']['ci_upper']
    
    valid_indices = [i for i, val in enumerate(y) if val is not None]
    
    if valid_indices:
        valid_x = [x[i] for i in valid_indices]
        valid_y = [y[i] for i in valid_indices]
        
        # Check if we have confidence intervals
        valid_ci_indices = [i for i in valid_indices if ci_lower[i] is not None and ci_upper[i] is not None]
        
        if valid_ci_indices:
            valid_ci_x = [x[i] for i in valid_ci_indices]
            valid_ci_y = [y[i] for i in valid_ci_indices]
            lower_err = [valid_ci_y[idx] - ci_lower[i] for idx, i in enumerate(valid_ci_indices)]
            upper_err = [ci_upper[i] - valid_ci_y[idx] for idx, i in enumerate(valid_ci_indices)]
            
            plt.errorbar(valid_ci_x, valid_ci_y, 
                        yerr=[lower_err, upper_err], 
                        fmt='o-', capsize=5, label='RankMe with 95% CI')
        else:
            plt.plot(valid_x, valid_y, 'o-', label='RankMe')
    
    plt.xlabel('Checkpoint')
    plt.ylabel('RankMe')
    plt.title('RankMe vs. Checkpoint')
    plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summaries_dir, "rankme.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CLID plot
    plt.figure(figsize=(10, 6))
    y = task_agnostic_summary['metrics']['clid']['values']
    ci_lower = task_agnostic_summary['metrics']['clid']['ci_lower']
    ci_upper = task_agnostic_summary['metrics']['clid']['ci_upper']
    
    valid_indices = [i for i, val in enumerate(y) if val is not None]
    
    if valid_indices:
        valid_x = [x[i] for i in valid_indices]
        valid_y = [y[i] for i in valid_indices]
        
        # Check if we have confidence intervals
        valid_ci_indices = [i for i in valid_indices if ci_lower[i] is not None and ci_upper[i] is not None]
        
        if valid_ci_indices:
            valid_ci_x = [x[i] for i in valid_ci_indices]
            valid_ci_y = [y[i] for i in valid_ci_indices]
            lower_err = [valid_ci_y[idx] - ci_lower[i] for idx, i in enumerate(valid_ci_indices)]
            upper_err = [ci_upper[i] - valid_ci_y[idx] for idx, i in enumerate(valid_ci_indices)]
            
            plt.errorbar(valid_ci_x, valid_ci_y, 
                        yerr=[lower_err, upper_err], 
                        fmt='o-', capsize=5, label='CLID with 95% CI')
        else:
            plt.plot(valid_x, valid_y, 'o-', label='CLID')
    
    plt.xlabel('Checkpoint')
    plt.ylabel('CLID Value')
    plt.title('CLID vs. Checkpoint')
    plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summaries_dir, "clid.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Alpha-ReQ plot
    plt.figure(figsize=(10, 6))
    y = task_agnostic_summary['metrics']['alphareq']['values']
    ci_lower = task_agnostic_summary['metrics']['alphareq']['ci_lower']
    ci_upper = task_agnostic_summary['metrics']['alphareq']['ci_upper']
    
    valid_indices = [i for i, val in enumerate(y) if val is not None]
    
    if valid_indices:
        valid_x = [x[i] for i in valid_indices]
        valid_y = [y[i] for i in valid_indices]
        
        # Check if we have confidence intervals
        valid_ci_indices = [i for i in valid_indices if ci_lower[i] is not None and ci_upper[i] is not None]
        
        if valid_ci_indices:
            valid_ci_x = [x[i] for i in valid_ci_indices]
            valid_ci_y = [y[i] for i in valid_ci_indices]
            lower_err = [valid_ci_y[idx] - ci_lower[i] for idx, i in enumerate(valid_ci_indices)]
            upper_err = [ci_upper[i] - valid_ci_y[idx] for idx, i in enumerate(valid_ci_indices)]
            
            plt.errorbar(valid_ci_x, valid_ci_y, 
                        yerr=[lower_err, upper_err], 
                        fmt='o-', capsize=5, label='Alpha-ReQ with 95% CI')
        else:
            plt.plot(valid_x, valid_y, 'o-', label='Alpha-ReQ')
    
    plt.xlabel('Checkpoint')
    plt.ylabel('Alpha')
    plt.title('Alpha-ReQ vs. Checkpoint')
    plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summaries_dir, "alphareq.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # LiDAR plot
    plt.figure(figsize=(10, 6))
    y = task_agnostic_summary['metrics']['lidar']['values']
    ci_lower = task_agnostic_summary['metrics']['lidar']['ci_lower']
    ci_upper = task_agnostic_summary['metrics']['lidar']['ci_upper']
    
    valid_indices = [i for i, val in enumerate(y) if val is not None]
    
    if valid_indices:
        valid_x = [x[i] for i in valid_indices]
        valid_y = [y[i] for i in valid_indices]
        
        # Check if we have confidence intervals
        valid_ci_indices = [i for i in valid_indices if ci_lower[i] is not None and ci_upper[i] is not None]
        
        if valid_ci_indices:
            valid_ci_x = [x[i] for i in valid_ci_indices]
            valid_ci_y = [y[i] for i in valid_ci_indices]
            lower_err = [valid_ci_y[idx] - ci_lower[i] for idx, i in enumerate(valid_ci_indices)]
            upper_err = [ci_upper[i] - valid_ci_y[idx] for idx, i in enumerate(valid_ci_indices)]
            
            plt.errorbar(valid_ci_x, valid_ci_y, 
                        yerr=[lower_err, upper_err], 
                        fmt='o-', capsize=5, label='LiDAR with 95% CI')
        else:
            plt.plot(valid_x, valid_y, 'o-', label='LiDAR')
    
    plt.xlabel('Checkpoint')
    plt.ylabel('LiDAR Value')
    plt.title('LiDAR vs. Checkpoint')
    plt.xticks(x, [it.replace('iteration_', '') for it in sorted_iterations], rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summaries_dir, "lidar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary tables and plots created successfully")

################################################################
# Worker process function
################################################################

def worker_process(gpu_id, checkpoints, output_dir, datasets_config, cfg):
    """
    Worker process to handle a subset of checkpoints on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use
        checkpoints: List of checkpoint paths to process
        output_dir: Directory to save outputs
        datasets_config: Configuration for datasets
        cfg: Configuration dictionary
    """
    # Import needed modules
    import logging
    import os
    import gc
    import torch
    import traceback
    from datetime import datetime
    
    # Set GPU device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    # Set up process-specific logger
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    worker_log_file = os.path.join(log_dir, f"gpu_{gpu_id}_processing.log")
    
    # Create file handler
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(logging.Formatter(
        f'%(asctime)s [GPU {gpu_id}] %(levelname)s: %(message)s'
    ))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        f'%(asctime)s [GPU {gpu_id}] %(levelname)s: %(message)s'
    ))
    
    # Configure logger
    worker_logger = logging.getLogger(f'worker_{gpu_id}')
    worker_logger.setLevel(logging.INFO)
    worker_logger.addHandler(file_handler)
    worker_logger.addHandler(console_handler)
    
    # Check and report GPU status
    worker_logger.info(f"Worker process starting on device: {device}")
    if torch.cuda.is_available():
        worker_logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")
        worker_logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB total")
    
    worker_logger.info(f"Processing {len(checkpoints)} checkpoints")
    
    # Process each checkpoint
    processed_checkpoints = []
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        try:
            checkpoint_name = os.path.basename(checkpoint)
            worker_logger.info(f"Starting processing of checkpoint {checkpoint_idx+1}/{len(checkpoints)}: {checkpoint_name}")
            
            # Extract checkpoint iteration
            if checkpoint_name == 'checkpoint.pth':
                iteration = 'final'
            else:
                iteration = checkpoint_name.split('_')[-1].split('.')[0]
            
            # Create output directory
            checkpoint_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
            os.makedirs(checkpoint_output_dir, exist_ok=True)
            
            # Create log file
            log_file = os.path.join(checkpoint_output_dir, f"processing_log_gpu_{gpu_id}.txt")
            
            # Check if this checkpoint has been fully processed
            final_metrics_path = os.path.join(checkpoint_output_dir, "all_metrics.json")
            if os.path.exists(final_metrics_path):
                worker_logger.info(f"Checkpoint {iteration} already fully processed, skipping")
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now()} - Checkpoint {iteration} already fully processed\n")
                processed_checkpoints.append(checkpoint_output_dir)
                continue
            
            # Extract and evaluate features
            checkpoint_dir = extract_and_evaluate_features(
                checkpoint,
                output_dir,
                datasets_config,
                device,
                cfg
            )
            
            processed_checkpoints.append(checkpoint_dir)
            
            # Force cleanup between checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory usage after cleanup
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                worker_logger.info(f"GPU memory after cleanup: Allocated {memory_allocated:.2f} GB, Reserved {memory_reserved:.2f} GB")
            
            worker_logger.info(f"Completed processing of checkpoint {checkpoint_idx+1}/{len(checkpoints)}: {checkpoint_name}")
            
        except Exception as e:
            worker_logger.error(f"Error processing checkpoint {checkpoint}: {e}")
            worker_logger.error(traceback.format_exc())
    
    # Mark worker as complete
    completion_file = os.path.join(output_dir, f"gpu_{gpu_id}_complete")
    with open(completion_file, "w") as f:
        f.write("done")
    
    worker_logger.info(f"Worker {gpu_id} has completed processing all assigned checkpoints")
    return processed_checkpoints

#############################################################
# Main Function
#############################################################

def main():
    """
    Main function to coordinate benchmark processing across multiple GPUs.
    """
    # Import logging at the top level to make it available in the main function
    import logging
    import torch.multiprocessing as mp
    
    parser = argparse.ArgumentParser("DINO Benchmarking")
    
    # Checkpoint and output directories
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    
    # Checkpoint processing options
    parser.add_argument('--interval', type=int, default=10000,
                        help='Interval between checkpoints to process')
    parser.add_argument('--include_final', type=str, default='True',
                        help='Whether to include the final checkpoint (True/False)')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension for the model')
    
    # Dataset paths
    parser.add_argument('--mhist_path', type=str, default='',
                        help='Path to MHIST dataset')
    parser.add_argument('--crc_path', type=str, default='',
                        help='Path to CRC dataset')
    parser.add_argument('--pcam_path', type=str, default='',
                        help='Path to PCam dataset')
    parser.add_argument('--pannuke_path', type=str, default='',
                        help='Path to PanNuke dataset')
    parser.add_argument('--monuseg_path', type=str, default='',
                        help='Path to MonuSeg dataset')
    parser.add_argument('--bracs_path', type=str, default='',
                    help='Path to BRACS dataset')
    parser.add_argument('--midog_path', type=str, default='',
                        help='Path to MiDOG++ classification dataset')
    parser.add_argument('--glas_path', type=str, default='',
                    help='Path to GLAS dataset')
    parser.add_argument('--bcss_path', type=str, default='',
                    help='Path to BCSS dataset root')

    
    # Dataset magnifications
    parser.add_argument('--pannuke_magnification', type=str, default='40x',
                        help='Magnification for PanNuke dataset (20x or 40x)')
    parser.add_argument('--monuseg_magnification', type=str, default='40x',
                        help='Magnification for MonuSeg dataset (20x or 40x)')
    
    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    
    # Augmentation parameters
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of images to sample for augmentation features')
    parser.add_argument('--augmentations_per_image', type=int, default=50,
                        help='Number of augmentations per image')
    parser.add_argument('--global_size', type=int, default=224,
                        help='Size of global crops')
    parser.add_argument('--local_size', type=int, default=96,
                        help='Size of local crops')
    parser.add_argument('--n_local_crops', type=int, default=1,
                        help='Number of local crops (default: 1)')
    parser.add_argument('--global_crop_scale_min', type=float, default=0.4,
                        help='Minimum scale for global crops')
    parser.add_argument('--global_crop_scale_max', type=float, default=1.0,
                        help='Maximum scale for global crops')
    parser.add_argument('--local_crop_scale_min', type=float, default=0.05,
                        help='Minimum scale for local crops')
    parser.add_argument('--local_crop_scale_max', type=float, default=0.4,
                        help='Maximum scale for local crops')
    
    # Normalization parameters
    parser.add_argument('--normalize_mean_r', type=float, default=0.485,
                        help='Mean R channel for normalization')
    parser.add_argument('--normalize_mean_g', type=float, default=0.456,
                        help='Mean G channel for normalization')
    parser.add_argument('--normalize_mean_b', type=float, default=0.406,
                        help='Mean B channel for normalization')
    parser.add_argument('--normalize_std_r', type=float, default=0.229,
                        help='Std R channel for normalization')
    parser.add_argument('--normalize_std_g', type=float, default=0.224,
                        help='Std G channel for normalization')
    parser.add_argument('--normalize_std_b', type=float, default=0.225,
                        help='Std B channel for normalization')
    
    # Convergence parameters
    parser.add_argument('--convergence_start_size', type=int, default=10000,
                        help='Start size for convergence testing')
    parser.add_argument('--convergence_step_size', type=int, default=1000,
                        help='Step size for convergence testing')
    parser.add_argument('--convergence_threshold', type=float, default=1e-3,
                        help='Threshold for convergence')
    parser.add_argument('--convergence_min_steps', type=int, default=5,
                        help='Minimum number of steps before checking convergence')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Confidence level for statistical convergence testing')
    parser.add_argument('--bootstrap_samples', type=int, default=100,
                        help='Number of bootstrap samples for confidence intervals')
    
    # Classification parameters
    parser.add_argument('--weight_decay_values', type=str, default='1e-5,1e-4,1e-3,1e-2,1e-1',
                        help='Comma-separated list of weight decay values to try')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Base learning rate for classification')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs for classification')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio for test evaluation')
    
    # Segmentation parameters
    parser.add_argument('--seg_learning_rate', type=float, default=1e-4,
                        help='Learning rate for segmentation')
    parser.add_argument('--seg_early_stop_patience', type=int, default=10,
                        help='Patience for early stopping in segmentation')
    parser.add_argument('--seg_max_epochs', type=int, default=50,
                        help='Maximum number of epochs for segmentation')
    parser.add_argument('--seg_val_split', type=float, default=0.2,
                        help='Validation split ratio for segmentation')
    
    # Monte Carlo parameters
    parser.add_argument('--monte_carlo_iterations', type=int, default=20,
                        help='Number of Monte Carlo iterations for evaluation')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up multiprocessing - use spawn method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    # Convert string boolean to actual boolean
    args.include_final = args.include_final.lower() == 'true'
    
    # Convert weight decay values string to list of floats
    args.weight_decay_values = [float(x) for x in args.weight_decay_values.split(',')]
    
    # Create tuples for crop scales and normalization
    args.global_crop_scale = (args.global_crop_scale_min, args.global_crop_scale_max)
    args.local_crop_scale = (args.local_crop_scale_min, args.local_crop_scale_max)
    args.normalize_mean = (args.normalize_mean_r, args.normalize_mean_g, args.normalize_mean_b)
    args.normalize_std = (args.normalize_std_r, args.normalize_std_g, args.normalize_std_b)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up main process logger
    logging_format = '%(asctime)s [Main] %(levelname)s: %(message)s'
    logging.basicConfig(
        filename=os.path.join(log_dir, "main_process.log"),
        level=logging.INFO,
        format=logging_format,
        filemode='a'
    )
    logger = logging.getLogger('main')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(console)
    
    # Print starting information
    logger.info("Starting DINO benchmarking with multiprocessing")
    
    # Determine available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPUs available, using CPU instead")
        num_gpus = 1  # Fall back to single process on CPU
    
    logger.info(f"Found {num_gpus} GPUs for processing")
    for i in range(num_gpus):
        if torch.cuda.is_available():
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Find checkpoints
    checkpoints = []
    for checkpoint_file in sorted(os.listdir(args.checkpoint_dir)):
        if checkpoint_file.startswith('checkpoint_iter_'):
            iteration = int(checkpoint_file.split('_')[-1].split('.')[0])
            if iteration % args.interval == 0 and iteration > 0:
                checkpoints.append(os.path.join(args.checkpoint_dir, checkpoint_file))
    
    # Add final checkpoint if requested
    if args.include_final:
        final_checkpoint = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(final_checkpoint):
            checkpoints.append(final_checkpoint)
    
    logger.info(f"Found {len(checkpoints)} total checkpoints to process")
    
    # Configure datasets
    datasets_config = {
        'classification': {},
        'segmentation': {}
    }
    
    # Add classification datasets with non-empty paths
    if args.mhist_path:
        datasets_config['classification']['MHIST'] = {
            'path': args.mhist_path,
            'num_classes': 2
        }
    
    if args.crc_path:
        datasets_config['classification']['CRC'] = {
            'path': args.crc_path,
            'num_classes': 9
        }
    
    if args.pcam_path:
        datasets_config['classification']['PCam'] = {
            'path': args.pcam_path,
            'num_classes': 2
        }
    
    
    # Add segmentation datasets with non-empty paths
    if args.pannuke_path:
        datasets_config['segmentation']['PanNuke'] = {
            'path': args.pannuke_path,
            'magnification': args.pannuke_magnification
        }
    
    if args.monuseg_path:
        datasets_config['segmentation']['MonuSeg'] = {
            'path': args.monuseg_path,
            'magnification': args.monuseg_magnification
        }
    
    if args.bracs_path:
        datasets_config['classification']['BRACS'] = {
            'path': args.bracs_path,
            'num_classes': 7
        }

    if args.midog_path:
        datasets_config['classification']['MiDOG'] = {
            'path': args.midog_path,
            'num_classes': 2
        }

    if args.bcss_path:
    datasets_config['segmentation']['BCSS'] = {
        'path': args.bcss_path,
        'magnification': '40x', # Placeholder for consistency
        'num_classes': 22
    }

    if args.glas_path:
    datasets_config['segmentation']['GLAS'] = {
        'path': args.glas_path,
        'magnification': '40x'  # Placeholder for consistency
    }

    # Create configuration dictionary
    cfg = vars(args)
    
    # Divide checkpoints among GPUs
    checkpoint_groups = [[] for _ in range(num_gpus)]
    for i, checkpoint in enumerate(checkpoints):
        checkpoint_groups[i % num_gpus].append(checkpoint)
    
    # Log checkpoint distribution
    for gpu_id in range(num_gpus):
        logger.info(f"GPU {gpu_id} will process {len(checkpoint_groups[gpu_id])} checkpoints")
        for cp in checkpoint_groups[gpu_id]:
            logger.info(f"  - {os.path.basename(cp)}")
    
    # Clear any existing completion files
    for gpu_id in range(num_gpus):
        completion_file = os.path.join(args.output_dir, f"gpu_{gpu_id}_complete")
        if os.path.exists(completion_file):
            os.remove(completion_file)
            logger.info(f"Removed existing completion file for GPU {gpu_id}")
    
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        if len(checkpoint_groups[gpu_id]) > 0:  # Only start workers that have checkpoints to process
            p = mp.Process(
                target=worker_process,
                args=(
                    gpu_id,
                    checkpoint_groups[gpu_id],
                    args.output_dir,
                    datasets_config,
                    cfg
                )
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker process for GPU {gpu_id} (PID: {p.pid})")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Check if all GPUs completed successfully
    all_complete = True
    for gpu_id in range(num_gpus):
        if len(checkpoint_groups[gpu_id]) > 0:  # Only check workers that had work to do
            completion_file = os.path.join(args.output_dir, f"gpu_{gpu_id}_complete")
            if not os.path.exists(completion_file):
                all_complete = False
                logger.warning(f"Worker for GPU {gpu_id} did not complete successfully")
    
    # Create summary tables if all processes completed
    if all_complete:
        logger.info("All worker processes completed successfully")
        logger.info("Creating summary tables...")
        try:
            create_summary_tables(args.output_dir)
            logger.info("Summary tables created successfully")
        except Exception as e:
            logger.error(f"Error creating summaries: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("Some worker processes did not complete. Skipping summary tables.")
    
    logger.info("DINO benchmarking completed")


if __name__ == '__main__':
    main()
