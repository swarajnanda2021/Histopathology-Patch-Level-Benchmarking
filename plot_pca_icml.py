#!/usr/bin/env python3
"""
Dual-Model PCA Visualization with Hue Comparison
Produces: original_image.png, pca_vanilla.png, pca_adios.png, hue_comparison.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import ndimage
import openslide
import argparse

# Add project root to path
sys.path.insert(0, '/data1/vanderbc/nandas1/FoundationModel_ViT-B_p16_b1024')

from models.vision_transformer.modern_vit import VisionTransformer


def load_model_from_checkpoint(checkpoint_path, device):
    """Load teacher backbone using architecture from checkpoint args"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'args' not in checkpoint:
        raise KeyError("No 'args' found in checkpoint!")
    
    args = checkpoint['args']
    
    patch_size = getattr(args, 'patch_size', 16)
    embed_dim = getattr(args, 'embeddingdim', 768)
    depth = getattr(args, 'vitdepth', 12)
    num_heads = getattr(args, 'vitheads', 12)
    
    print(f"  Architecture: patch={patch_size}, dim={embed_dim}, depth={depth}, heads={num_heads}")
    
    teacher_encoder = VisionTransformer(
        img_size=224,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.4,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    teacher_state = checkpoint['teacher']
    
    backbone_state = {}
    for k, v in teacher_state.items():
        if k.startswith('module.backbone.'):
            new_key = k.replace('module.backbone.', '')
            backbone_state[new_key] = v
    
    teacher_encoder.load_state_dict(backbone_state, strict=False)
    teacher_encoder = teacher_encoder.to(device)
    teacher_encoder.eval()
    
    print(f"  ✓ Loaded successfully")
    return teacher_encoder


def find_tissue_regions_diverse(slide, tissue_threshold=240, min_tissue_ratio=0.5, 
                                n_regions=10, thumbnail_size=2000):
    """Find diverse tissue regions"""
    thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_np = np.array(thumbnail)
    
    gray = np.dot(thumbnail_np[...,:3], [0.299, 0.587, 0.114])
    tissue_mask = gray < tissue_threshold
    
    tissue_mask = ndimage.binary_erosion(tissue_mask, iterations=2)
    tissue_mask = ndimage.binary_dilation(tissue_mask, iterations=2)
    
    scale_x = slide.dimensions[0] / thumbnail.size[0]
    scale_y = slide.dimensions[1] / thumbnail.size[1]
    
    patch_size = 3584
    patch_size_thumb = int(patch_size / scale_x)
    
    valid_regions = []
    step = 20
    
    for y in range(0, tissue_mask.shape[0] - patch_size_thumb, step):
        for x in range(0, tissue_mask.shape[1] - patch_size_thumb, step):
            patch_mask = tissue_mask[y:y+patch_size_thumb, x:x+patch_size_thumb]
            tissue_ratio = np.mean(patch_mask)
            
            if tissue_ratio >= min_tissue_ratio:
                patch_gray = gray[y:y+patch_size_thumb, x:x+patch_size_thumb]
                tissue_pixels = patch_gray[patch_mask]
                
                if len(tissue_pixels) > 0:
                    darkness_score = np.mean(tissue_pixels)
                    if darkness_score > 230:
                        continue
                    darkness_std = np.std(tissue_pixels)
                else:
                    continue
                
                patch_edges = ndimage.sobel(patch_mask.astype(float))
                edge_score = np.std(patch_edges)
                
                valid_regions.append({
                    'location': (int(x * scale_x), int(y * scale_y)),
                    'tissue_ratio': tissue_ratio,
                    'darkness_score': darkness_score,
                    'darkness_std': darkness_std,
                    'edge_score': edge_score,
                    'thumb_coords': (x, y),
                    'center_x': x + patch_size_thumb // 2,
                    'center_y': y + patch_size_thumb // 2
                })
    
    if not valid_regions:
        raise ValueError("No valid tissue regions found!")
    
    for region in valid_regions:
        darkness_norm = 1.0 - (region['darkness_score'] / 240.0)
        region['combined_score'] = (darkness_norm * 0.5 + region['tissue_ratio'] * 0.3 + 
                                   min(region['edge_score'], 1.0) * 0.2)
    
    valid_regions.sort(key=lambda x: x['combined_score'], reverse=True)
    
    selected_regions = []
    min_distance_thumb = patch_size_thumb * 0.5
    
    for region in valid_regions:
        far_enough = True
        for selected in selected_regions:
            distance = np.sqrt((region['center_x'] - selected['center_x'])**2 + 
                             (region['center_y'] - selected['center_y'])**2)
            if distance < min_distance_thumb:
                far_enough = False
                break
        
        if far_enough:
            selected_regions.append(region)
            if len(selected_regions) >= n_regions:
                break
    
    if len(selected_regions) < n_regions:
        for region in valid_regions:
            if region not in selected_regions:
                selected_regions.append(region)
                if len(selected_regions) >= n_regions:
                    break
    
    print(f"  Found {len(valid_regions)} valid regions, selected {len(selected_regions)}")
    return selected_regions


def extract_3584_region(slide, location):
    """Extract a 3584x3584 region"""
    x, y = location
    region = slide.read_region((x, y), 0, (3584, 3584))
    region = np.array(region)[:, :, :3]
    return region


def extract_patches_from_region(region_3584):
    """Divide into 256 patches of 224x224"""
    patches = []
    for i in range(16):
        for j in range(16):
            patch = region_3584[i*224:(i+1)*224, j*224:(j+1)*224]
            patches.append(patch)
    return patches


def normalize_image(image):
    """Normalize image"""
    mean = np.array([0.6816, 0.5640, 0.7232])
    std = np.array([0.1617, 0.1714, 0.1389])
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def extract_features_from_patches(model, patches, device):
    """Extract features"""
    all_features = []
    batch_size = 8
    
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        batch_tensors = []
        
        for patch in batch_patches:
            patch_normalized = normalize_image(patch)
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).float()
            batch_tensors.append(patch_tensor)
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            output = model(batch)
            if isinstance(output, dict):
                patch_tokens = output['patchtokens']
            else:
                num_registers = 4
                patch_tokens = output[:, 1+num_registers:, :]
        
        for j in range(patch_tokens.shape[0]):
            all_features.append(patch_tokens[j].cpu().numpy())
    
    return np.vstack(all_features)


def compute_foreground_mask(features, patches, brightness_threshold=220):
    """Compute foreground mask based on pixel brightness (white = background)"""

    foreground_mask = []

    for patch in patches:
        # Each patch is 224x224, divided into 14x14 = 196 tokens of 16x16 pixels each
        for i in range(14):
            for j in range(14):
                region = patch[i*16:(i+1)*16, j*16:(j+1)*16]
                # Convert to grayscale
                gray_value = np.mean(np.dot(region, [0.299, 0.587, 0.114]))
                # Foreground if darker than threshold
                foreground_mask.append(gray_value < brightness_threshold)

    foreground_mask = np.array(foreground_mask, dtype=bool)

    n_foreground = np.sum(foreground_mask)
    print(f"    Foreground pixels: {n_foreground} ({n_foreground/len(foreground_mask)*100:.1f}%)")

    return foreground_mask


def temp_compute_foreground_mask(features, patches):
    """Compute foreground/background mask using 3D PCA clustering"""
    from sklearn.cluster import KMeans
    
    pca = PCA(n_components=3)
    pca_features_3d = pca.fit_transform(features)
    
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features_3d)
    
    actual_brightness = []
    for patch in patches:
        for i in range(14):
            for j in range(14):
                region = patch[i*16:(i+1)*16, j*16:(j+1)*16]
                gray_value = np.mean(np.dot(region, [0.299, 0.587, 0.114]))
                actual_brightness.append(gray_value)
    actual_brightness = np.array(actual_brightness)
    
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_pixels = np.where(cluster_mask)[0]
        
        if len(cluster_pixels) > 0:
            cluster_brightness = actual_brightness[cluster_mask]
            mean_brightness = np.mean(cluster_brightness)
            std_brightness = np.std(cluster_brightness)
            
            cluster_pca_values = pca_features_3d[cluster_mask]
            mean_pc1 = np.mean(cluster_pca_values[:, 0])
            
            cluster_mask_2d = cluster_mask.reshape(256, 196).any(axis=1).reshape(16, 16)
            labeled, n_components = ndimage.label(cluster_mask_2d)
            
            periphery_mask = np.zeros((16, 16), dtype=bool)
            periphery_mask[0, :] = True
            periphery_mask[-1, :] = True
            periphery_mask[:, 0] = True
            periphery_mask[:, -1] = True
            
            periphery_ratio = np.sum(cluster_mask_2d & periphery_mask) / max(np.sum(cluster_mask_2d), 1)
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_pixels),
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'mean_pc1': mean_pc1,
                'n_components': n_components,
                'periphery_ratio': periphery_ratio,
            })
    
    background_scores = []
    for stats in cluster_stats:
        score = 0
        
        if stats['mean_brightness'] > 245:
            score += 3
        elif stats['mean_brightness'] > 235:
            score += 2
        elif stats['mean_brightness'] > 225:
            score += 1
        
        if stats['std_brightness'] < 10:
            score += 1
        
        if stats['periphery_ratio'] > 0.5:
            score += 1
        
        if stats['n_components'] <= 2:
            score += 1
        
        all_pc1_values = [s['mean_pc1'] for s in cluster_stats]
        if stats['mean_pc1'] == max(all_pc1_values) or stats['mean_pc1'] == min(all_pc1_values):
            score += 1
        
        background_scores.append({
            'cluster_id': stats['cluster_id'],
            'score': score,
            'stats': stats
        })
    
    background_threshold_score = 4
    background_clusters = [bs for bs in background_scores if bs['score'] >= background_threshold_score]
    
    if not background_clusters:
        best_candidate = max(background_scores, key=lambda x: x['score'])
        if best_candidate['stats']['mean_brightness'] > 220:
            background_clusters = [best_candidate]
        else:
            return np.ones(len(features), dtype=bool)
    
    background_mask = np.zeros(len(features), dtype=bool)
    for bg_cluster in background_clusters:
        cluster_id = bg_cluster['cluster_id']
        background_mask |= (cluster_labels == cluster_id)
    
    removed_dark_pixels = np.sum(background_mask & (actual_brightness < 200))
    if removed_dark_pixels > 100:
        background_mask_2d = background_mask.reshape(256, 196)
        
        for patch_idx in range(256):
            patch_bg_mask = background_mask_2d[patch_idx]
            patch_bg_mask_2d = patch_bg_mask.reshape(14, 14)
            
            labeled, n_features = ndimage.label(patch_bg_mask_2d)
            if n_features > 0:
                sizes = ndimage.sum(patch_bg_mask_2d, labeled, range(1, n_features + 1))
                if len(sizes) > 0:
                    max_size = max(sizes)
                    if max_size < 49:
                        patch_bg_mask_2d[:] = False
                    else:
                        max_label = np.argmax(sizes) + 1
                        patch_bg_mask_2d[labeled != max_label] = False
            
            background_mask_2d[patch_idx] = patch_bg_mask_2d.flatten()
        
        background_mask = background_mask_2d.flatten()
    
    foreground_mask = ~background_mask
    n_foreground = np.sum(foreground_mask)
    print(f"    Foreground pixels: {n_foreground} ({n_foreground/len(foreground_mask)*100:.1f}%)")
    
    return foreground_mask


def apply_pca_rgb_and_hue(features, foreground_mask):
    """Apply PCA and return RGB-normalized features and hue values"""
    n_foreground = np.sum(foreground_mask)
    
    if n_foreground < 10:
        return None, None
    
    pca = PCA(n_components=3)
    foreground_features = features[foreground_mask]
    pca_features = pca.fit_transform(foreground_features)
    
    pca_features_normalized = pca_features.copy()
    for i in range(3):
        if pca_features_normalized[:, i].max() > pca_features_normalized[:, i].min():
            pca_features_normalized[:, i] = (
                (pca_features_normalized[:, i] - pca_features_normalized[:, i].min()) /
                (pca_features_normalized[:, i].max() - pca_features_normalized[:, i].min())
            )
    
    pca_features_rgb = np.zeros((len(foreground_mask), 3))
    pca_features_rgb[~foreground_mask] = 0
    pca_features_rgb[foreground_mask] = pca_features_normalized
    
    # Extract hue values
    hsv_pixels = np.zeros_like(pca_features_normalized)
    for i in range(len(pca_features_normalized)):
        rgb_single = pca_features_normalized[i].reshape(1, 1, 3)
        hsv_single = mcolors.rgb_to_hsv(rgb_single)
        hsv_pixels[i] = hsv_single.reshape(3)
    
    hue_values = hsv_pixels[:, 0] * 360
    
    return pca_features_rgb, hue_values


def create_rgb_grid(pca_features_rgb):
    """Create 224x224 RGB grid from features"""
    rgb_grid = np.zeros((224, 224, 3))
    patch_idx = 0
    
    for i in range(16):
        for j in range(16):
            start_idx = patch_idx * 196
            end_idx = start_idx + 196
            patch_rgb = pca_features_rgb[start_idx:end_idx].reshape(14, 14, 3)
            rgb_grid[i*14:(i+1)*14, j*14:(j+1)*14] = patch_rgb
            patch_idx += 1
    
    return rgb_grid


def save_original_image(original_image, output_path):
    """Save original H&E image"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.imshow(original_image)
    ax.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def save_pca_image(rgb_grid, output_path):
    """Save PCA-RGB image"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.imshow(rgb_grid, interpolation='nearest')
    ax.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def save_hue_comparison_plot(hue_vanilla, hue_adios, output_path):
    """Save combined polar hue plot comparing both models"""
    
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    
    n_bins = 72
    theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    # Compute histograms
    hue_rad_vanilla = hue_vanilla * np.pi / 180
    hue_rad_adios = hue_adios * np.pi / 180
    
    counts_vanilla, _ = np.histogram(hue_rad_vanilla, bins=theta_edges)
    counts_adios, _ = np.histogram(hue_rad_adios, bins=theta_edges)
    
    # Normalize each independently
    def normalize_counts(counts):
        min_c, max_c = counts.min(), counts.max()
        if max_c > min_c:
            return (counts - min_c) / (max_c - min_c)
        return np.ones_like(counts) * 0.5
    
    counts_vanilla_norm = normalize_counts(counts_vanilla) * 0.75
    counts_adios_norm = normalize_counts(counts_adios) * 0.75
    
    # Close the loop
    theta_plot = np.append(theta_centers, theta_centers[0])
    vanilla_plot = np.append(counts_vanilla_norm, counts_vanilla_norm[0])
    adios_plot = np.append(counts_adios_norm, counts_adios_norm[0])
    
    # Plot both distributions
    ax.plot(theta_plot, vanilla_plot, color='#1E88E5', linewidth=2.5, alpha=0.9, label='Vanilla')
    ax.fill(theta_plot, vanilla_plot, color='#1E88E5', alpha=0.15)
    
    ax.plot(theta_plot, adios_plot, color='#E53935', linewidth=2.5, alpha=0.9, label='ADIOS')
    ax.fill(theta_plot, adios_plot, color='#E53935', alpha=0.15)
    
    # Formatting
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=12, alpha=0.7)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    
    # Color ring
    n_segments = 360
    theta_ring = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    ring_bottom = 0.78
    ring_height = 0.06
    
    for i, angle in enumerate(theta_ring):
        hue_degrees = (90 - np.degrees(angle)) % 360
        hsv_color = np.array([[[hue_degrees / 360.0, 1.0, 1.0]]])
        rgb_color = mcolors.hsv_to_rgb(hsv_color)[0, 0]
        ax.bar(angle, ring_height, width=2*np.pi/n_segments, 
               bottom=ring_bottom, color=rgb_color, edgecolor='none', linewidth=0)
    
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta_circle, np.ones_like(theta_circle) * ring_bottom, 'k-', linewidth=0.5, alpha=0.5)
    ax.plot(theta_circle, np.ones_like(theta_circle) * (ring_bottom + ring_height), 'k-', linewidth=0.5, alpha=0.5)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=14)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def process_region_dual(slide, region_info, region_index, model_vanilla, model_adios, 
                        device, output_dir, cancer_type):
    """Process single region with both models"""
    
    print(f"\n  Region {region_index}")
    
    region_dir = os.path.join(output_dir, cancer_type, f"region_{region_index}")
    os.makedirs(region_dir, exist_ok=True)
    
    original_image = extract_3584_region(slide, region_info['location'])
    patches = extract_patches_from_region(original_image)
    
    save_original_image(original_image, os.path.join(region_dir, "original_image.png"))
    
    print("    Extracting vanilla features...")
    features_vanilla = extract_features_from_patches(model_vanilla, patches, device)
    
    print("    Extracting ADIOS features...")
    features_adios = extract_features_from_patches(model_adios, patches, device)
    
    print("    Computing foreground mask (from vanilla)...")
    foreground_mask = compute_foreground_mask(features_vanilla, patches)
    
    print("    Applying PCA to vanilla...")
    pca_rgb_vanilla, hue_vanilla = apply_pca_rgb_and_hue(features_vanilla, foreground_mask)
    
    print("    Applying PCA to ADIOS...")
    pca_rgb_adios, hue_adios = apply_pca_rgb_and_hue(features_adios, foreground_mask)
    
    if pca_rgb_vanilla is None or pca_rgb_adios is None:
        print("    ⚠️ Skipping region due to insufficient foreground pixels")
        return False
    
    rgb_grid_vanilla = create_rgb_grid(pca_rgb_vanilla)
    rgb_grid_adios = create_rgb_grid(pca_rgb_adios)
    
    save_pca_image(rgb_grid_vanilla, os.path.join(region_dir, "pca_vanilla.png"))
    save_pca_image(rgb_grid_adios, os.path.join(region_dir, "pca_adios.png"))
    save_hue_comparison_plot(hue_vanilla, hue_adios, os.path.join(region_dir, "hue_comparison.png"))
    
    return True


def process_wsi_dual(wsi_path, cancer_type, output_dir, model_vanilla, model_adios, 
                     device, n_regions=10):
    """Process WSI with both models"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {cancer_type}")
    print(f"{'='*60}")
    
    slide = None
    try:
        slide = openslide.OpenSlide(wsi_path)
        regions = find_tissue_regions_diverse(slide, min_tissue_ratio=0.5, n_regions=n_regions)
        
        for i in range(len(regions)):
            process_region_dual(slide, regions[i], i, model_vanilla, model_adios, 
                               device, output_dir, cancer_type)
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if slide:
            slide.close()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Dual-Model PCA Visualization')
    parser.add_argument('--output_dir', type=str, default='pca_dual_visualizations',
                       help='Output directory')
    parser.add_argument('--n_regions', type=int, default=5,
                       help='Number of regions per WSI')
    args = parser.parse_args()
    
    vanilla_checkpoint = "/data1/vanderbc/nandas1/FoundationModel_ViT-L_p16_b2048/logs/checkpoint_iter_00140000.pth"
    adios_checkpoint = "/data1/vanderbc/nandas1/FoundationModel_ViT-L_p16_b2048_adios/logs/checkpoint_iter_00140000.pth"
    
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
        'STAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-STAD_svs/svs/fa9ef6ca-2b68-4951-ae4e-0faa7f437569/TCGA-D7-A6ET-01Z-00-DX1.A4FF5141-6B2A-456B-9EA2-E5DE72156647.svs",
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    print("\nLoading VANILLA model:")
    model_vanilla = load_model_from_checkpoint(vanilla_checkpoint, device)
    
    print("\nLoading ADIOS model:")
    model_adios = load_model_from_checkpoint(adios_checkpoint, device)
    
    for cancer_type, wsi_path in wsi_paths.items():
        process_wsi_dual(wsi_path, cancer_type, args.output_dir, 
                        model_vanilla, model_adios, device, n_regions=args.n_regions)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    CANCER_TYPE/")
    print(f"      region_N/")
    print(f"        original_image.png")
    print(f"        pca_vanilla.png")
    print(f"        pca_adios.png")
    print(f"        hue_comparison.png")


if __name__ == "__main__":
    main()
