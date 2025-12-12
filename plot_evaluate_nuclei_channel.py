import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

# Add parent directory to path to find vision_transformer and other modules
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "TCGA_TMEDinov2_version2_ViT-B"))

from datasets import PanNukeDataset, MonuSegDataset, SynchronizedTransform
from vision_transformer import MaskModel_SpectralNorm
from soft_moe.vision_transformer import VisionTransformer

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def load_pretrained_mask_model(checkpoint_path, num_masks=3):
    """Load pre-trained mask model from checkpoint"""
    print(f"Loading mask model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create mask model architecture
    mask_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.1,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    mask_model = MaskModel_SpectralNorm(
        encoder=mask_encoder,
        num_masks=num_masks,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract mask model state dict
    if 'mask_model' in checkpoint:
        mask_state_dict = checkpoint['mask_model']
    else:
        raise KeyError("No 'mask_model' found in checkpoint")
    
    # Clean state dict keys (handle DDP)
    cleaned_state_dict = OrderedDict()
    for k, v in mask_state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    
    # Load weights
    mask_model.load_state_dict(cleaned_state_dict, strict=False)
    
    print(f"Successfully loaded mask model")
    return mask_model


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate IoU between predicted and ground truth masks
    pred_mask: [H, W] float tensor in [0, 1]
    gt_mask: [H, W] binary tensor {0, 1}
    """
    # Binarize prediction
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    # Calculate intersection and union
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = (intersection / union).item()
    return iou


def denormalize_image(image, mean=(0.6816, 0.5640, 0.7232), std=(0.1617, 0.1714, 0.1389)):
    """Denormalize image from normalized to [0, 1] range"""
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    return image_denorm


def evaluate_model_optimized(model, dataset, dataset_name, device, threshold=0.5, batch_size=32):
    """
    Optimized evaluation with batching - CORRECTED VERSION
    """
    model.eval()
    
    all_ious = []
    all_predictions = []
    all_gts = []
    all_images = []
    
    # Create DataLoader for batching
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"  Evaluating on {dataset_name} ({len(dataset)} samples) with batch_size={batch_size}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  {dataset_name}", leave=False)):
            images, mask_2ch, _, _ = batch
            
            # Extract GT nuclei masks (first channel)
            gt_nuclei = mask_2ch[:, 0]  # [B, H, W]
            
            # Move to GPU - images are ALREADY normalized from dataset
            images_normalized = images.to(device)
            
            # Resize to 224x224 if needed (batched operation)
            if images_normalized.shape[-1] != 224:
                images_normalized = F.interpolate(
                    images_normalized,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Forward pass (batched) - mask model expects normalized images
            with torch.cuda.amp.autocast():
                mask_output = model(images_normalized)
                predicted_masks = mask_output['masks']  # [B, num_masks, H, W]
            
            # Resize predictions back to GT size (batched)
            B, num_channels, H_pred, W_pred = predicted_masks.shape
            H_gt, W_gt = gt_nuclei.shape[1], gt_nuclei.shape[2]
            
            if (H_pred, W_pred) != (H_gt, W_gt):
                predicted_masks = F.interpolate(
                    predicted_masks,
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Move to CPU for IoU calculation (batched)
            predicted_masks_cpu = predicted_masks.cpu()
            gt_nuclei_cpu = gt_nuclei.cpu()
            
            # Calculate IoU for each sample in batch
            for b in range(predicted_masks_cpu.shape[0]):
                channel_ious = []
                for ch in range(num_channels):
                    iou = calculate_iou(
                        predicted_masks_cpu[b, ch],
                        gt_nuclei_cpu[b],
                        threshold=threshold
                    )
                    channel_ious.append(iou)
                all_ious.append(channel_ious)
                
                # Store for visualization (limit to first 100)
                if len(all_predictions) < 100:
                    all_predictions.append(predicted_masks_cpu[b])
                    all_gts.append(gt_nuclei_cpu[b])
                    all_images.append(images[b].cpu())
    
    all_ious = np.array(all_ious)  # Shape: [num_samples, num_channels]
    
    # Find best channel per sample
    best_channels = np.argmax(all_ious, axis=1)  # Shape: [num_samples]
    
    # Wrap in metrics dictionary
    metrics = {'iou': all_ious}
    
    return metrics, all_predictions, all_gts, all_images, best_channels
                




def create_visualization_multi_model(
    images_pannuke, gts_pannuke, 
    images_monuseg, gts_monuseg,
    all_model_results,  # Dict of {model_key: {'preds_pannuke': ..., 'preds_monuseg': ..., 'best_ch_pn': ..., 'best_ch_ms': ...}}
    save_path, 
    num_examples_per_dataset=10, 
    threshold=0.5
):
    """
    Create visualization with multiple model variants
    Row 0: Original images
    Row 1: Ground truth
    Rows 2+: Model predictions
    """
    n_models = len(all_model_results)
    n_rows = n_models + 2  # +2 for original and GT rows
    
    # Create figure with (n_models + 2) rows x 5 columns
    fig, axes = plt.subplots(n_rows, 5, figsize=(25, 5 * n_rows))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    # Select examples uniformly from each dataset
    pannuke_total = len(images_pannuke)
    monuseg_total = len(images_monuseg)
    
    pannuke_indices = np.linspace(0, pannuke_total-1, num_examples_per_dataset, dtype=int)
    monuseg_indices = np.linspace(0, monuseg_total-1, num_examples_per_dataset, dtype=int)
    
    # ========== ROW 0: Original Images ==========
    row_idx = 0
    
    # Columns 0-1: PanNuke originals
    for i in range(2):
        col = i
        ax = axes[row_idx, col]
        data_idx = pannuke_indices[i] if i < len(pannuke_indices) else pannuke_indices[0]
        
        image = images_pannuke[data_idx]
        image_vis = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
        ax.imshow(image_vis)
        ax.axis('off')
        
        # Label for first column
        if col == 0:
            ax.text(0.05, 0.95, 'Original\nImage',
                   transform=ax.transAxes, fontsize=18,
                   color='white', weight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='black', alpha=0.8))
        
        # Title for first row
        if col == 0:
            ax.set_title('PanNuke', fontsize=16, fontweight='bold', color='purple', pad=10)
        
        # Purple border
        rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='purple', 
                                  linewidth=3, transform=ax.transAxes)
        ax.add_patch(rect)
    
    # Columns 2-4: MonuSeg originals
    for i in range(3):
        col = 2 + i
        ax = axes[row_idx, col]
        data_idx = monuseg_indices[i] if i < len(monuseg_indices) else monuseg_indices[0]
        
        image = images_monuseg[data_idx]
        image_vis = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
        ax.imshow(image_vis)
        ax.axis('off')
        
        # Title for first row
        if col == 2:
            ax.set_title('MonuSeg', fontsize=16, fontweight='bold', color='orange', pad=10)
        
        # Orange border
        rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='orange', 
                                  linewidth=3, transform=ax.transAxes)
        ax.add_patch(rect)
    
    # ========== ROW 1: Ground Truth ==========
    row_idx = 1
    
    # Columns 0-1: PanNuke GT
    for i in range(2):
        col = i
        ax = axes[row_idx, col]
        data_idx = pannuke_indices[i] if i < len(pannuke_indices) else pannuke_indices[0]
        
        gt = gts_pannuke[data_idx]
        gt_vis = gt.cpu().numpy()
        ax.imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        
        # Label for first column
        if col == 0:
            ax.text(0.05, 0.95, 'Ground\nTruth',
                   transform=ax.transAxes, fontsize=18,
                   color='white', weight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='black', alpha=0.8))
        
        # Purple border
        rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='purple', 
                                  linewidth=3, transform=ax.transAxes)
        ax.add_patch(rect)
    
    # Columns 2-4: MonuSeg GT
    for i in range(3):
        col = 2 + i
        ax = axes[row_idx, col]
        data_idx = monuseg_indices[i] if i < len(monuseg_indices) else monuseg_indices[0]
        
        gt = gts_monuseg[data_idx]
        gt_vis = gt.cpu().numpy()
        ax.imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        
        # Orange border
        rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='orange', 
                                  linewidth=3, transform=ax.transAxes)
        ax.add_patch(rect)
    
    # ========== ROWS 2+: Model Predictions ==========
    for model_idx, (model_key, model_data) in enumerate(all_model_results.items()):
        row_idx = model_idx + 2  # Start from row 2
        
        preds_pannuke = model_data['preds_pannuke']
        preds_monuseg = model_data['preds_monuseg']
        best_ch_pn = model_data['best_ch_pn']
        best_ch_ms = model_data['best_ch_ms']
        model_name = model_data['name']
        iteration = model_data['iteration']
        
        # Columns 0-1: PanNuke predictions
        for i in range(2):
            col = i
            ax = axes[row_idx, col]
            data_idx = pannuke_indices[i] if i < len(pannuke_indices) else pannuke_indices[0]
            
            # Use per-sample best channel
            nuclei_ch = best_ch_pn[data_idx]
            pred = preds_pannuke[data_idx][nuclei_ch]
            pred_vis = (pred.cpu().numpy() > threshold).astype(np.float32)
            ax.imshow(pred_vis, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add channel indicator in corner
            ax.text(0.95, 0.05, f'Ch{nuclei_ch}',
                   transform=ax.transAxes, fontsize=12,
                   color='yellow', weight='bold',
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='black', alpha=0.7))
            
            # Add model label in first column
            if col == 0:
                label_text = f'{model_name}\n{iteration}'
                ax.text(0.05, 0.95, label_text,
                       transform=ax.transAxes, fontsize=18,
                       color='white', weight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor='black', alpha=0.8))
            
            # Purple border
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='purple', 
                                      linewidth=3, transform=ax.transAxes)
            ax.add_patch(rect)
        
        # Columns 2-4: MonuSeg predictions
        for i in range(3):
            col = 2 + i
            ax = axes[row_idx, col]
            data_idx = monuseg_indices[i] if i < len(monuseg_indices) else monuseg_indices[0]
            
            # Use per-sample best channel
            nuclei_ch = best_ch_ms[data_idx]
            pred = preds_monuseg[data_idx][nuclei_ch]
            pred_vis = (pred.cpu().numpy() > threshold).astype(np.float32)
            ax.imshow(pred_vis, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add channel indicator in corner
            ax.text(0.95, 0.05, f'Ch{nuclei_ch}',
                   transform=ax.transAxes, fontsize=12,
                   color='yellow', weight='bold',
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='black', alpha=0.7))
            
            # Orange border
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='orange', 
                                      linewidth=3, transform=ax.transAxes)
            ax.add_patch(rect) 
        
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved to {save_path}")

  

    


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model paths - expanded with all variants and iterations
    variant_paths = {
        'ADIOS_20k': {
            'name': 'ADIOS',
            'iteration': '20k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B/logs/checkpoint_iter_00020000.pth'
        },
        'ADIOS_30k': {
            'name': 'ADIOS',
            'iteration': '30k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B/logs/checkpoint_iter_00030000.pth'
        },
        'ADIOS_40k': {
            'name': 'ADIOS',
            'iteration': '40k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B/logs/checkpoint_iter_00040000.pth'
        },
        'ADIOS_50k': {
            'name': 'ADIOS',
            'iteration': '50k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B/logs/checkpoint_iter_00050000.pth'
        },
        'Ours_20k': {
            'name': 'Ours',
            'iteration': '20k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B_L1_Perceptual/logs/old/checkpoint_iter_00020000.pth'
        },
        'Ours_30k': {
            'name': 'Ours',
            'iteration': '30k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B_L1_Perceptual/logs/old/checkpoint_iter_00030000.pth'
        },
        'Ours_40k': {
            'name': 'Ours',
            'iteration': '40k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B_L1_Perceptual/logs/old/checkpoint_iter_00040000.pth'
        },
        'Ours_50k': {
            'name': 'Ours',
            'iteration': '50k',
            'path': '/data1/vanderbc/nandas1/TCGA_TMEMIM_ViT-B_L1_Perceptual/logs/old/checkpoint_iter_00050000.pth'
        },
    }
    
    # Dataset paths
    pannuke_path = "/data1/vanderbc/nandas1/Benchmarks/PanNuke_patches_unnormalized"
    monuseg_path = "/data1/vanderbc/nandas1/Benchmarks/MonuSeg_patches_unnormalized"
    
    # Create output directory
    output_dir = Path("nuclei_channel_evaluation_multi")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Multi-Model Nuclei Channel Identification and mIOU Evaluation")
    print("="*80)
    
    # Create transform (no augmentation for evaluation)
    transform_settings = {
        "normalize": {
            "mean": [0.6816, 0.5640, 0.7232],
            "std": [0.1617, 0.1714, 0.1389]
        },
        "RandomRotate90": {"p": 0},
        "HorizontalFlip": {"p": 0},
        "VerticalFlip": {"p": 0},
        "Downscale": {"scale": 0.5, "p": 0},
        "Blur": {"blur_limit": 7, "p": 0},
        "ColorJitter": {"scale_setting": 0.25, "scale_color": 0.1, "p": 0}
    }
    transform = SynchronizedTransform(transform_settings, input_shape=96)
    
    # Load PanNuke datasets
    print("\n[1/4] Loading PanNuke datasets...")
    pannuke_train = PanNukeDataset(
        data_dir=pannuke_path,
        split='Training',
        magnification='40x',
        transform=transform
    )
    
    pannuke_test = PanNukeDataset(
        data_dir=pannuke_path,
        split='Test',
        magnification='40x',
        transform=transform
    )
    
    print(f"  PanNuke Training: {len(pannuke_train)} samples")
    print(f"  PanNuke Test: {len(pannuke_test)} samples")
    
    # Load MonuSeg datasets
    print("\n[2/4] Loading MonuSeg datasets...")
    monuseg_train = MonuSegDataset(
        data_dir=monuseg_path,
        split='Training',
        magnification='40x',
        transform=transform
    )
    
    monuseg_test = MonuSegDataset(
        data_dir=monuseg_path,
        split='Test',
        magnification='40x',
        transform=transform
    )
    
    print(f"  MonuSeg Training: {len(monuseg_train)} samples")
    print(f"  MonuSeg Test: {len(monuseg_test)} samples")
    
    # Combine datasets per benchmark
    class CombinedDataset:
        def __init__(self, train_ds, test_ds):
            self.train_ds = train_ds
            self.test_ds = test_ds
            self.train_len = len(train_ds)
        
        def __len__(self):
            return len(self.train_ds) + len(self.test_ds)
        
        def __getitem__(self, idx):
            if idx < self.train_len:
                return self.train_ds[idx]
            else:
                return self.test_ds[idx - self.train_len]
    
    pannuke_full = CombinedDataset(pannuke_train, pannuke_test)
    monuseg_full = CombinedDataset(monuseg_train, monuseg_test)
    
    print(f"\nCombined datasets:")
    print(f"  PanNuke total: {len(pannuke_full)} samples")
    print(f"  MonuSeg total: {len(monuseg_full)} samples")
    
    # Store results for all models
    all_model_results = {}
    all_results_dict = {}
    
    # Get GT data once (will be reused for visualization)
    images_pannuke = None
    gts_pannuke = None
    images_monuseg = None
    gts_monuseg = None
    
    # Evaluate each model variant
    print("\n[3/4] Evaluating all model variants...")

    # Initialize model variable
    model = None

    for model_key, model_info in variant_paths.items():
        model_name = model_info['name']
        iteration = model_info['iteration']
        checkpoint_path = model_info['path']
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} at {iteration}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(checkpoint_path):
            print(f"  WARNING: Checkpoint not found, skipping...")
            continue
        
        # FORCE COMPLETE CLEANUP
        if model is not None:
            del model
            model = None
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Sleep briefly to ensure cleanup (paranoid but effective)
        import time
        time.sleep(0.5)
    
        # CREATE COMPLETELY FRESH MODEL ARCHITECTURE
        print(f"  Creating fresh model architecture...")
        from soft_moe.vision_transformer import VisionTransformer
        
        mask_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_norm=False,
            dual_norm=False,
            drop_path_rate=0.1,
            pre_norm=False,
            num_register_tokens=4,
        )
        
        model = MaskModel_SpectralNorm(
            encoder=mask_encoder,
            num_masks=3,
            encoder_dim=192,
            drop_rate=0.2
        )
        
        # LOAD CHECKPOINT
        print(f"  Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract mask model state dict
        if 'mask_model' in checkpoint:
            mask_state_dict = checkpoint['mask_model']
        else:
            print(f"  ERROR: No 'mask_model' found in checkpoint!")
            print(f"  Available keys: {list(checkpoint.keys())}")
            continue
        
        # Clean state dict keys (handle DDP)
        cleaned_state_dict = {}
        for k, v in mask_state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[name] = v
    
        # VERIFY we're loading different weights
        # Get first parameter value as fingerprint
        first_param_name = list(cleaned_state_dict.keys())[0]
        first_param_value = cleaned_state_dict[first_param_name]
        param_fingerprint = float(first_param_value.flatten()[0])
        print(f"  Checkpoint fingerprint: {first_param_name} = {param_fingerprint:.8f}")
        
        # Load weights with strict checking
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"  WARNING: Missing {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"  WARNING: Unexpected {len(unexpected_keys)} keys")
        
        # VERIFY loaded correctly - check first parameter matches
        loaded_param_value = float(dict(model.named_parameters())[first_param_name].flatten()[0])
        print(f"  Loaded fingerprint:     {first_param_name} = {loaded_param_value:.8f}")
        
        if abs(param_fingerprint - loaded_param_value) > 1e-6:
            print(f"  ERROR: Weights didn't load correctly!")
            continue
        
        # Move to GPU AFTER loading
        model = model.to(device)
        model.eval()  # Set to eval mode
        
        # Print checkpoint info
        if 'iteration' in checkpoint:
            print(f"  Checkpoint was saved at iteration: {checkpoint['iteration']}")
        
        print(f"  Model ready for evaluation")
    
        # Evaluate on PanNuke
        print(f"\n  Evaluating on PanNuke...")
        metrics_pannuke, preds_pannuke, gts_pn, imgs_pn, best_ch_pn = evaluate_model_optimized(
            model, pannuke_full, "PanNuke", device, threshold=0.5, batch_size=32
        )
        
        # Store GT data from first model
        if images_pannuke is None:
            images_pannuke = imgs_pn
            gts_pannuke = gts_pn
        
        # Evaluate on MonuSeg
        print(f"\n  Evaluating on MonuSeg...")
        metrics_monuseg, preds_monuseg, gts_ms, imgs_ms, best_ch_ms = evaluate_model_optimized(
            model, monuseg_full, "MonuSeg", device, threshold=0.5, batch_size=32
        )
        
        # Store GT data from first model
        if images_monuseg is None:
            images_monuseg = imgs_ms
            gts_monuseg = gts_ms
        
        # Analyze results using PER-SAMPLE best channels
        # Get IoU using each sample's best channel
        best_iou_pn = metrics_pannuke['iou'][np.arange(len(best_ch_pn)), best_ch_pn]
        best_iou_ms = metrics_monuseg['iou'][np.arange(len(best_ch_ms)), best_ch_ms]
        
        # Channel distribution statistics
        from collections import Counter
        pn_channel_dist = Counter(best_ch_pn)
        ms_channel_dist = Counter(best_ch_ms)
        
        print(f"\n  PanNuke Channel Distribution:")
        for ch in range(3):
            count = pn_channel_dist.get(ch, 0)
            pct = 100.0 * count / len(best_ch_pn)
            print(f"    Channel {ch}: {count:>5d} samples ({pct:>5.1f}%)")
        
        print(f"\n  PanNuke Results (per-sample best channel):")
        mean_val = best_iou_pn.mean()
        std_val = best_iou_pn.std()
        print(f"    {'IoU':>20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n  MonuSeg Channel Distribution:")
        for ch in range(3):
            count = ms_channel_dist.get(ch, 0)
            pct = 100.0 * count / len(best_ch_ms)
            print(f"    Channel {ch}: {count:>5d} samples ({pct:>5.1f}%)")
        
        print(f"\n  MonuSeg Results (per-sample best channel):")
        mean_val = best_iou_ms.mean()
        std_val = best_iou_ms.std()
        print(f"    {'IoU':>20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Store for visualization and JSON
        all_model_results[model_key] = {
            'name': model_name,
            'iteration': iteration,
            'preds_pannuke': preds_pannuke,
            'preds_monuseg': preds_monuseg,
            'best_ch_pn': best_ch_pn,
            'best_ch_ms': best_ch_ms,
        }
    
        # Store for JSON results
        pannuke_results = {
            'iou_mean': float(best_iou_pn.mean()),
            'iou_std': float(best_iou_pn.std()),
            'channel_distribution': {int(k): int(v) for k, v in pn_channel_dist.items()},
            'iou_all_channels_mean': metrics_pannuke['iou'].mean(axis=0).tolist(),
        }
        
        monuseg_results = {
            'iou_mean': float(best_iou_ms.mean()),
            'iou_std': float(best_iou_ms.std()),
            'channel_distribution': {int(k): int(v) for k, v in ms_channel_dist.items()},
            'iou_all_channels_mean': metrics_monuseg['iou'].mean(axis=0).tolist(),
        } 
        
        all_results_dict[model_key] = {
            'name': model_name,
            'iteration': iteration,
            'checkpoint_path': checkpoint_path,
            'pannuke': pannuke_results,
            'monuseg': monuseg_results,
        }

    # Final cleanup
    if model is not None:
        del model
    torch.cuda.empty_cache()
    
    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY - All Models")
    print("="*80)
    
    # Group by model type for comparison
    adios_results = {k: v for k, v in all_results_dict.items() if v['name'] == 'ADIOS'}
    ours_results = {k: v for k, v in all_results_dict.items() if v['name'] == 'Ours'}
    
    print("\nADIOS Models:")
    for model_key, data in sorted(adios_results.items()):
        print(f"  {data['iteration']:>4s}: PanNuke={data['pannuke']['iou_mean']:.4f}±{data['pannuke']['iou_std']:.4f}, "
              f"MonuSeg={data['monuseg']['iou_mean']:.4f}±{data['monuseg']['iou_std']:.4f}")
    
    print("\nOurs Models:")
    for model_key, data in sorted(ours_results.items()):
        print(f"  {data['iteration']:>4s}: PanNuke={data['pannuke']['iou_mean']:.4f}±{data['pannuke']['iou_std']:.4f}, "
              f"MonuSeg={data['monuseg']['iou_mean']:.4f}±{data['monuseg']['iou_std']:.4f}")
    
    # Save numerical results
    import json
    with open(output_dir / 'results_all_models.json', 'w') as f:
        json.dump(all_results_dict, f, indent=2)
    print(f"\nNumerical results saved to {output_dir / 'results_all_models.json'}")
    
    # Create visualization
    print("\n[4/4] Creating visualization...")
    create_visualization_multi_model(
        images_pannuke, gts_pannuke,
        images_monuseg, gts_monuseg,
        all_model_results,
        save_path=output_dir / 'comparison_all_models.png',
        num_examples_per_dataset=10,
        threshold=0.5
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
