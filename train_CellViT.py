import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# You'll need to import from your existing modules
from utils import load_dino_backbone, WarmupDecayScheduler
from models import CellViT
from datasets import PanNukeDataset, SynchronizedTransform


class CombinedLoss(nn.Module):
    def __init__(self, w_xentropy=1.0, w_dice=1.0, w_mse=1.0, w_msge=1.0, w_ftversky=1.0):
        super(CombinedLoss, self).__init__()
        self.w_dice = w_dice
        self.w_mse = w_mse
        self.w_msge = w_msge
        self.w_ftversky = w_ftversky
        self.w_xentropy = w_xentropy

    def forward(self, true_mask, pred_mask, true_dist, pred_dist):
        xentropy = self.xentropy_loss(true_mask, pred_mask)
        dice = self.dice_loss(true_mask, pred_mask)
        mse = self.mse_loss(true_dist, pred_dist)
        msge = self.msge_loss(true_dist, pred_dist, true_mask)
        focal_tversky = self.focal_tversky_loss(true_mask, pred_mask)

        loss = (
            self.w_xentropy * xentropy
            + self.w_ftversky * focal_tversky
            + self.w_dice * dice
            + self.w_mse * mse
            + self.w_msge * msge
        )
        return loss

    def xentropy_loss(self, true, pred, reduction="mean"):
        epsilon = 1e-7
        if pred.dim() == 4:
            pred = pred.permute(0, 2, 3, 1)
            true = true.permute(0, 2, 3, 1)
        pred = F.softmax(pred, dim=-1)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum(true * torch.log(pred), dim=-1)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def dice_loss(self, true, pred, smooth=1e-5):
        loss = 0
        weights = [0.7, 0.3]
        for channel in range(true.shape[1]):
            inse = torch.sum(pred[:, channel] * true[:, channel], (1, 2))
            l = torch.sum(pred[:, channel], (1, 2))
            r = torch.sum(true[:, channel], (1, 2))
            loss += weights[channel] * (1.0 - (2.0 * inse + smooth) / (l + r + smooth))
        return loss.mean()
    
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
        
        pred = pred.permute(0, 2, 3, 1)
        true = true.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 0]
        
        focus = (focus[..., None]).float()
        focus = torch.cat([focus, focus], axis=-1)
        true_grad = get_gradient_hv(true)
        pred_grad = get_gradient_hv(pred)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)

        denominator = focus.sum() + 1e-8
        if denominator.item() < 1e-7:
            return torch.tensor(0.0, device=loss.device)

        loss = loss.sum() / denominator
        return loss


def train_cellvit(
    checkpoint_path,
    pannuke_data_path,
    output_path,
    magnification='20x',
    device='cuda',
    batch_size=16,
    num_workers=4,
    max_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    early_stop_patience=10,
    val_split=0.2,
    feature_dim=1024,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225)
):
    """
    Train CellViT model using DINOv2 checkpoint on PanNuke dataset.
    
    Args:
        checkpoint_path: Path to DINOv2 checkpoint
        pannuke_data_path: Path to PanNuke dataset root
        output_path: Path to save trained CellViT model
        magnification: '20x' or '40x'
        device: 'cuda' or 'cpu'
        batch_size: Training batch size
        num_workers: DataLoader workers
        max_epochs: Maximum training epochs
        learning_rate: Base learning rate
        weight_decay: Weight decay
        early_stop_patience: Patience for early stopping
        val_split: Validation split ratio
        feature_dim: Feature dimension of backbone
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    """
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load DINOv2 backbone
    print(f"Loading DINOv2 backbone from {checkpoint_path}...")
    backbone = load_dino_backbone(checkpoint_path, device)
    backbone.eval()  # Keep backbone frozen
    
    # Determine image size based on magnification
    img_size = 96 if magnification == '20x' else 224
    
    # Create transforms
    transform_settings = {
        "RandomRotate90": {"p": 0.5},
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Downscale": {"p": 0.15, "scale": 0.5},
        "Blur": {"p": 0.2, "blur_limit": 3},
        "ColorJitter": {"p": 0.2},
        "normalize": {"mean": normalize_mean, "std": normalize_std}
    }
    
    val_transform_settings = {
        "normalize": {"mean": normalize_mean, "std": normalize_std}
    }
    
    train_transform = SynchronizedTransform(transform_settings, input_shape=img_size)
    val_transform = SynchronizedTransform(val_transform_settings, input_shape=img_size)
    
    # Load datasets
    print("Loading PanNuke datasets...")
    train_dataset = PanNukeDataset(
        data_dir=pannuke_data_path,
        split='Training',
        magnification=magnification,
        transform=train_transform
    )
    
    test_dataset = PanNukeDataset(
        data_dir=pannuke_data_path,
        split='Test',
        magnification=magnification,
        transform=val_transform
    )
    
    # Split training into train/val
    train_size = len(train_dataset)
    val_size = int(train_size * val_split)
    train_indices = list(range(val_size, train_size))
    val_indices = list(range(val_size))
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize CellViT model
    print("Initializing CellViT model...")
    model = CellViT(backbone, encoder_dim=feature_dim, drop_rate=0.2).to(device)
    
    # Loss function
    criterion = CombinedLoss(
        w_xentropy=1.0,
        w_dice=1.0,
        w_mse=2.5,
        w_msge=8.0,
        w_ftversky=0.0
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-8,  # Start with very low LR for warmup
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    
    # Scheduler
    warmup_epochs = 5
    base_lr = learning_rate
    final_lr = learning_rate / 10
    warmup_start_lr = learning_rate / 100
    
    scheduler = WarmupDecayScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        base_lr=base_lr,
        final_lr=final_lr,
        warmup_start_lr=warmup_start_lr
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        # Train
        model.train()
        model.encoder.eval()  # Keep encoder frozen
        
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc="Training")
        
        for images, masks, distance_maps, instance_masks in train_progress:
            images = images.to(device)
            masks = masks.float().to(device)
            distance_maps = distance_maps.float().to(device)
            
            # Forward
            outputs = model(images, magnification)
            loss = criterion(masks, outputs['masks'], distance_maps, outputs['distances'])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=0.3
            )
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for images, masks, distance_maps, instance_masks in val_progress:
                images = images.to(device)
                masks = masks.float().to(device)
                distance_maps = distance_maps.float().to(device)
                
                outputs = model(images, magnification)
                loss = criterion(masks, outputs['masks'], distance_maps, outputs['distances'])
                
                val_loss += loss.item()
                val_progress.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"New best model! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save best model
    if best_model_state is not None:
        print(f"\nSaving trained CellViT model to {output_path}...")
        torch.save({
            'model_state_dict': best_model_state,
            'magnification': magnification,
            'feature_dim': feature_dim,
            'best_val_loss': best_val_loss
        }, output_path)
        print("Model saved successfully!")
    else:
        print("Warning: No model improvements found during training")
    
    return model


def main():
    parser = argparse.ArgumentParser("Train CellViT with DINOv2 backbone")
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, default='/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth',
                        help='Path to DINOv2 checkpoint')
    parser.add_argument('--pannuke_path', type=str, default='/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized',
                        help='Path to PanNuke dataset root')
    parser.add_argument('--output_path', type=str, default='/data1/vanderbc/nandas1/CellViT_models/TCGA_Dinov2_ViT-B_run2/model.pth',
                        help='Path to save trained CellViT model')
    
    # Optional arguments
    parser.add_argument('--magnification', type=str, default='40x',
                        choices=['20x', '40x'],
                        help='Magnification level (default: 20x)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='Feature dimension (default: 768)')
    
    args = parser.parse_args()
    
    # Train model
    train_cellvit(
        checkpoint_path=args.checkpoint_path,
        pannuke_data_path=args.pannuke_path,
        output_path=args.output_path,
        magnification=args.magnification,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        val_split=args.val_split,
        feature_dim=args.feature_dim
    )


if __name__ == '__main__':
    main()
