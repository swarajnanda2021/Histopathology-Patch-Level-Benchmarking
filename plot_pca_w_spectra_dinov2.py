#!/usr/bin/env python3
"""
PCA Visualization with Integrated Polar Plots - DINOv2 Only Version
Uses 2x4 grid layout: Original image (2x2) | DINOv2 visualizations (2x2)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d
import openslide
import sys

# Add path to your models
sys.path.append('/data1/vanderbc/nandas1/PostProc')
from utils import load_dino_backbone

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.5

def find_tissue_regions_diverse(slide, tissue_threshold=240, min_tissue_ratio=0.5, n_regions=10, thumbnail_size=2000, output_dir=None):
    """Find diverse tissue regions"""
    from scipy import ndimage
    
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
    
    print(f"Found {len(valid_regions)} valid tissue regions")
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
    
    print(f"Selected {len(selected_regions)} regions")
    
    if len(selected_regions) > 0 and output_dir:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(tissue_mask, cmap='gray')
        colors = ['red', 'yellow', 'green', 'blue', 'cyan', 'magenta', 'orange', 'lime', 'pink', 'purple']
        for i, region in enumerate(selected_regions):
            x, y = region['thumb_coords']
            rect = patches.Rectangle((x, y), patch_size_thumb, patch_size_thumb, 
                                    linewidth=2, edgecolor=colors[i % len(colors)], 
                                    facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f"Selected {len(selected_regions)} Tissue Regions")
        plt.savefig(os.path.join(output_dir, 'tissue_detection_debug.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    return selected_regions

def extract_3584_region(slide, location):
    """Extract a 3584x3584 region"""
    x, y = location
    region = slide.read_region((x, y), 0, (3584, 3584))
    region = np.array(region)[:, :, :3]
    return region

def extract_patches_from_region(region_3584):
    """Divide into 256 patches"""
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
            output = model(batch, return_type='structured')
            patch_tokens = output['patch_tokens']
        
        for j in range(patch_tokens.shape[0]):
            all_features.append(patch_tokens[j].cpu().numpy())
    
    return np.vstack(all_features)

def find_optimal_threshold_otsu(values):
    """Find optimal threshold using Otsu's method"""
    values_normalized = ((values - values.min()) / (values.max() - values.min()) * 255).astype(np.uint8)
    threshold_value = threshold_otsu(values_normalized)
    threshold = values.min() + (threshold_value / 255) * (values.max() - values.min())
    print(f"    Otsu threshold: {threshold:.3f} (normalized: {threshold_value})")
    return threshold

def compute_foreground_mask(features, patches):
    """Compute foreground/background mask using 3D PCA clustering with multiple validation"""
    from sklearn.cluster import KMeans
    from scipy import ndimage
    
    print("  Computing background/foreground separation (3D PCA clustering)")
    
    # Stage 1: PCA to 3D
    pca = PCA(n_components=3)
    pca_features_3d = pca.fit_transform(features)
    
    print(f"    PCA: {features.shape[1]} -> 3 dimensions")
    print(f"    Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"    Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")


    print("\n=== PCA Variance Analysis ===")
    for k in range(1, 201, 20):
        pca_temp = PCA(n_components=k)
        pca_temp.fit(features)
        total_var = pca_temp.explained_variance_ratio_.sum()
        print(f"k={k:2d}: {total_var:.4f} ({total_var*100:.2f}%)")
    print("="*40 + "\n")
        
    # Stage 2: Clustering in 3D space
    n_clusters = 4
    print(f"\n  Clustering in 3D PCA space with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features_3d)
    cluster_centers = kmeans.cluster_centers_
    
    # Stage 3: Compute actual brightness for validation
    actual_brightness = []
    for patch in patches:
        for i in range(14):
            for j in range(14):
                region = patch[i*16:(i+1)*16, j*16:(j+1)*16]
                gray_value = np.mean(np.dot(region, [0.299, 0.587, 0.114]))
                actual_brightness.append(gray_value)
    actual_brightness = np.array(actual_brightness)
    
    # Stage 4: Identify which cluster is background
    print("\n  Analyzing clusters to identify background...")
    
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
            mean_pc2 = np.mean(cluster_pca_values[:, 1])
            mean_pc3 = np.mean(cluster_pca_values[:, 2])
            
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
                'mean_pc2': mean_pc2,
                'mean_pc3': mean_pc3,
                'n_components': n_components,
                'periphery_ratio': periphery_ratio,
                'cluster_center': cluster_centers[cluster_id]
            })
            
            print(f"    Cluster {cluster_id}:")
            print(f"      Size: {len(cluster_pixels)} pixels ({len(cluster_pixels)/len(cluster_labels)*100:.1f}%)")
            print(f"      Mean brightness: {mean_brightness:.1f} ± {std_brightness:.1f}")
            print(f"      PCA center: [{mean_pc1:.2f}, {mean_pc2:.2f}, {mean_pc3:.2f}]")
            print(f"      Spatial: {n_components} components, {periphery_ratio:.1%} at periphery")
    
    # Stage 5: Score each cluster for likelihood of being background
    print("\n  Scoring clusters for background likelihood...")
    
    background_scores = []
    for stats in cluster_stats:
        score = 0
        reasons = []
        
        if stats['mean_brightness'] > 245:
            score += 3
            reasons.append("very bright (>245)")
        elif stats['mean_brightness'] > 235:
            score += 2
            reasons.append("bright (>235)")
        elif stats['mean_brightness'] > 225:
            score += 1
            reasons.append("moderately bright (>225)")
        
        if stats['std_brightness'] < 10:
            score += 1
            reasons.append("uniform brightness")
        
        if stats['periphery_ratio'] > 0.5:
            score += 1
            reasons.append("peripheral location")
        
        if stats['n_components'] <= 2:
            score += 1
            reasons.append("spatially coherent")
        
        all_pc1_values = [s['mean_pc1'] for s in cluster_stats]
        if stats['mean_pc1'] == max(all_pc1_values) or stats['mean_pc1'] == min(all_pc1_values):
            score += 1
            reasons.append("extreme PC1 value")
        
        background_scores.append({
            'cluster_id': stats['cluster_id'],
            'score': score,
            'reasons': reasons,
            'stats': stats
        })
        
        print(f"    Cluster {stats['cluster_id']}: score={score}, reasons={reasons}")
    
    # Stage 6: Select background cluster(s)
    background_threshold_score = 4
    
    background_clusters = [bs for bs in background_scores if bs['score'] >= background_threshold_score]
    
    if not background_clusters:
        print("\n  ⚠️ No cluster meets background criteria (score >= 4)")
        print("     Falling back to most likely candidate...")
        best_candidate = max(background_scores, key=lambda x: x['score'])
        if best_candidate['stats']['mean_brightness'] > 220:
            background_clusters = [best_candidate]
            print(f"     Selected cluster {best_candidate['cluster_id']} (score={best_candidate['score']}, brightness={best_candidate['stats']['mean_brightness']:.1f})")
        else:
            print("     No suitable background cluster found - keeping all pixels as foreground")
            return np.ones(len(features), dtype=bool)
    
    # Stage 7: Create background mask
    background_mask = np.zeros(len(features), dtype=bool)
    for bg_cluster in background_clusters:
        cluster_id = bg_cluster['cluster_id']
        background_mask |= (cluster_labels == cluster_id)
        print(f"\n  ✓ Cluster {cluster_id} identified as background")
        print(f"     Reasons: {', '.join(bg_cluster['reasons'])}")
    
    # Stage 8: Additional validation
    removed_dark_pixels = np.sum(background_mask & (actual_brightness < 200))
    if removed_dark_pixels > 100:
        print(f"\n  ⚠️ WARNING: Background mask would remove {removed_dark_pixels} dark pixels!")
        print("     Applying morphological operations to preserve tissue...")
        
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
    
    # Final statistics
    foreground_mask = ~background_mask
    n_background = np.sum(background_mask)
    n_foreground = np.sum(foreground_mask)
    
    actual_whitespace = actual_brightness > 240
    actual_tissue = actual_brightness < 200
    whitespace_recall = np.sum(background_mask & actual_whitespace) / max(np.sum(actual_whitespace), 1)
    tissue_preserved = 1.0 - (np.sum(background_mask & actual_tissue) / max(np.sum(actual_tissue), 1))
    
    print(f"\n  Final separation statistics:")
    print(f"    Background pixels: {n_background} ({n_background/len(background_mask)*100:.1f}%)")
    print(f"    Foreground pixels: {n_foreground} ({n_foreground/len(foreground_mask)*100:.1f}%)")
    print(f"    Whitespace recall: {whitespace_recall:.1%}")
    print(f"    Tissue preserved: {tissue_preserved:.1%}")
    
    return foreground_mask

def compute_consensus_mask(all_masks):
    """Compute consensus mask from all models' masks"""
    masks_list = list(all_masks.values())
    consensus_mask = masks_list[0].copy()
    
    for mask in masks_list[1:]:
        consensus_mask = consensus_mask & mask
    
    n_consensus = np.sum(consensus_mask)
    n_total = len(consensus_mask)
    
    if n_consensus < 1000:
        consensus_mask = masks_list[0].copy()
        for mask in masks_list[1:]:
            consensus_mask = consensus_mask | mask
    
    return consensus_mask

def apply_pca_and_extract_hues(features, consensus_foreground_mask):
    """Apply PCA using consensus mask and get hues"""
    n_foreground = np.sum(consensus_foreground_mask)
    
    if n_foreground < 10:
        return None, None
    
    pca = PCA(n_components=3)
    foreground_features = features[consensus_foreground_mask]
    pca_features = pca.fit_transform(foreground_features)
    
    pca_features_normalized = pca_features.copy()
    for i in range(3):
        if pca_features_normalized[:, i].max() > pca_features_normalized[:, i].min():
            pca_features_normalized[:, i] = (
                (pca_features_normalized[:, i] - pca_features_normalized[:, i].min()) /
                (pca_features_normalized[:, i].max() - pca_features_normalized[:, i].min())
            )
    
    pca_features_rgb = np.zeros((len(consensus_foreground_mask), 3))
    pca_features_rgb[~consensus_foreground_mask] = 0
    pca_features_rgb[consensus_foreground_mask] = pca_features_normalized
    
    hsv_pixels = np.zeros_like(pca_features_normalized)
    for i in range(len(pca_features_normalized)):
        rgb_single = pca_features_normalized[i].reshape(1, 1, 3)
        hsv_single = mcolors.rgb_to_hsv(rgb_single)
        hsv_pixels[i] = hsv_single.reshape(3)
    
    hue_values = hsv_pixels[:, 0] * 360
    
    return pca_features_rgb, hue_values

def create_rgb_grid(pca_features_rgb):
    """Create RGB grid"""
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

def plot_polar_dinov2(ax, hue_data):
    """Polar plot for DINOv2 with individual normalization and color ring"""
    # Filter only DINOv2 data
    filtered_data = {k: v for k, v in hue_data.items() 
                     if v is not None and len(v) > 0}
    
    if not filtered_data:
        return
    
    aug_colors = {'standard': '#1E88E5', 'random': '#DC3545', 'mixed': '#FFA726'}
    
    # Process each histogram with individual normalization
    n_bins = 72
    theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    # Track plotted augmentation types for labels
    plotted_augs = []
    
    for model_name, hue_values in filtered_data.items():
        # Determine augmentation type from model name
        if 'standard' in model_name:
            aug_type = 'standard'
        elif 'random' in model_name:
            aug_type = 'random'
        elif 'mixed' in model_name:
            aug_type = 'mixed'
        else:
            continue  # Skip if can't determine type
            
        hue_radians = hue_values * np.pi / 180
        counts, _ = np.histogram(hue_radians, bins=theta_edges)
        
        # Individual min-max normalization
        min_count = counts.min()
        max_count = counts.max()
        if max_count > min_count:
            counts_norm = (counts - min_count) / (max_count - min_count)
        else:
            counts_norm = np.ones_like(counts) * 0.5
        
        theta_plot = np.append(theta_centers, theta_centers[0])
        counts_plot = np.append(counts_norm, counts_norm[0])
        
        # Scale down the plots to fit within the new limits (multiply by 0.75)
        counts_plot_scaled = counts_plot * 0.75
        ax.plot(theta_plot, counts_plot_scaled, color=aug_colors[aug_type], linewidth=2.5, alpha=0.9)
        ax.fill(theta_plot, counts_plot_scaled, color=aug_colors[aug_type], alpha=0.15)
        
        if aug_type not in plotted_augs:
            plotted_augs.append(aug_type)
    
    # Formatting
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Remove angular text labels but keep tick marks for reference
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # Empty labels
    
    # Radial axis
    ax.set_ylim(0, 0.9)  # Further reduced from 1.15
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=11, alpha=0.7)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    
    # Add color ring using polar bar plot
    n_segments = 360  # Number of segments for smooth gradient
    theta_ring = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    
    # Ring parameters
    ring_bottom = 0.78
    ring_height = 0.06
    
    # Create colors for each segment
    for i, angle in enumerate(theta_ring):
        hue_degrees = (90 - np.degrees(angle)) % 360  # Adjust so 0° (North) = Red
        hsv_color = np.array([[[hue_degrees / 360.0, 1.0, 1.0]]])
        rgb_color = mcolors.hsv_to_rgb(hsv_color)[0, 0]
        
        ax.bar(angle, ring_height, width=2*np.pi/n_segments, 
               bottom=ring_bottom, color=rgb_color, 
               edgecolor='none', linewidth=0)
    
    # Add border circles for the ring
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta_circle, np.ones_like(theta_circle) * ring_bottom, 
            'k-', linewidth=0.5, alpha=0.5)
    ax.plot(theta_circle, np.ones_like(theta_circle) * (ring_bottom + ring_height), 
            'k-', linewidth=0.5, alpha=0.5)
    
    # Augmentation type labels - positioned for better visibility
    label_positions = {
        'standard': (0.75, 0.55),  # bottom-right
        'random': (0.75, 0.45),     # below standard
        'mixed': (0.75, 0.35)       # below mixed
    }
    
    for aug_type in sorted(plotted_augs):
        if aug_type in label_positions:
            x_pos, y_pos = label_positions[aug_type]
            ax.text(x_pos, y_pos, aug_type,
                   transform=ax.transAxes,
                   fontsize=13, fontweight='bold',
                   color=aug_colors[aug_type],
                   bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor=aug_colors[aug_type],
                            alpha=0.8, linewidth=1.5))

def create_final_figure_dinov2(original_image, all_rgb_grids, all_hue_data, output_path, highlight_subplot=None):
    """Create final integrated figure for DINOv2 only with 2x4 grid layout
    
    Layout:
    - Left 2x2: Original image
    - Right 2x2: [standard, mixed]
                 [masked, polar]
    """
    
    # Calculate dimensions for square cells
    fig_width = 16  # inches
    fig_height = 8  # inches (half of width for 2x4 grid)
    
    # Create figure with WHITE background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Create 2x4 grid where original image spans 2x2 on left
    gs = fig.add_gridspec(2, 4, 
                          hspace=0.04,  # Minimal horizontal spacing
                          wspace=0.04,  # Minimal vertical spacing
                          left=0.04, right=0.96, 
                          top=0.96, bottom=0.04)
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    idx = 0
    
    # Dictionary to store axes for potential highlighting
    axes_dict = {}
    
    # Original image spanning left 2x2
    ax = fig.add_subplot(gs[:, :2])  # Spans both rows, first two columns
    ax.imshow(original_image, extent=[0, 3584, 3584, 0], interpolation='nearest')
    ax.set_xlabel('Pixels', fontsize=14)
    ax.set_ylabel('Pixels', fontsize=14)
    ax.set_xticks([0, 1792, 3584])
    ax.set_yticks([0, 1792, 3584])
    ax.tick_params(axis='both', labelsize=13)
    
    # Add label as text box inside plot
    ax.text(0.02, 0.98, f'{labels[idx]} Original Image (3584×3584)',
           transform=ax.transAxes,
           fontsize=15, fontweight='bold',
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='#333333',
                    alpha=0.9))
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(2)
    
    idx += 1
    
    # DINOv2 PCA plots in specific positions
    dinov2_positions = {
        'DINOv2: standard augmentations': (0, 2, 'standard'),  # Top-left of right grid
        'DINOv2: mixed augmentations': (0, 3, 'mixed'),        # Top-right of right grid
        'DINOv2: random augmentations': (1, 2, 'random'),      # Bottom-left of right grid
    }
    
    for model_name, (row, col, aug_type) in dinov2_positions.items():
        if model_name in all_rgb_grids:
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(all_rgb_grids[model_name], extent=[0, 3584, 3584, 0], 
                     interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Store axis for potential highlighting
            axes_dict[model_name] = ax
            
            # Add label as text box inside plot
            ax.text(0.02, 0.98, f'{labels[idx]} DINOv2 ({aug_type})',
                   transform=ax.transAxes,
                   fontsize=15, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='#333333',
                            alpha=0.9))
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
            
            idx += 1
    
    # Add highlighting if requested
    if highlight_subplot and highlight_subplot in axes_dict:
        ax_to_highlight = axes_dict[highlight_subplot]
        bbox = ax_to_highlight.get_position()
        
        highlight_rect = Rectangle(
            (bbox.x0 - 0.005, bbox.y0 - 0.005),
            bbox.width + 0.01, 
            bbox.height + 0.01,
            transform=fig.transFigure,
            fill=False,
            edgecolor='red',
            linewidth=3,
            linestyle='--',
            zorder=1000
        )
        fig.patches.append(highlight_rect)
    
    # Polar plot in bottom-right position
    ax_polar = fig.add_subplot(gs[1, 3], projection='polar')
    plot_polar_dinov2(ax_polar, all_hue_data)
    
    # Add label for polar plot
    ax_polar.text(0.5, 0.95, f'{labels[idx]} Hue Distribution',
                  transform=ax_polar.transAxes,
                  fontsize=14, fontweight='bold',  # Slightly smaller font
                  ha='center',
                  bbox=dict(boxstyle='round,pad=0.2',  # Reduced padding
                           facecolor='white',
                           edgecolor='#333333',
                           alpha=0.9))
    
    # Save with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def process_region(slide, region_info, region_index, checkpoints, device, output_dir, cancer_type):
    """Process single region with consensus masking for DINOv2 only"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING REGION {region_index}")
    print(f"{'='*60}")
    
    original_image = extract_3584_region(slide, region_info['location'])
    patches = extract_patches_from_region(original_image)
    
    all_features = {}
    all_masks = {}
    
    # PHASE 1: Extract features and compute individual masks (DINOv2 only)
    print("PHASE 1: Feature extraction and individual mask computation (DINOv2 only)")
    for model_name, checkpoint_path in checkpoints.items():
        if not os.path.exists(checkpoint_path):
            continue
        
        print(f"Processing: {model_name}")
        
        try:
            model = load_dino_backbone(checkpoint_path, device)
            model.eval()
            
            features = extract_features_from_patches(model, patches, device)
            mask = compute_foreground_mask(features, patches)
            
            all_features[model_name] = features
            all_masks[model_name] = mask
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    if not all_features:
        return
    
    # PHASE 2: Compute consensus mask
    print("\nPHASE 2: Computing consensus mask")
    consensus_mask = compute_consensus_mask(all_masks)
    print(f"  Consensus foreground: {np.sum(consensus_mask)} pixels")
    
    # PHASE 3: Apply PCA with consensus mask
    print("\nPHASE 3: Applying PCA with consensus mask")
    all_rgb_grids = {}
    all_hue_data = {}
    
    for model_name, features in all_features.items():
        print(f"  Applying consensus PCA: {model_name}")
        pca_features_rgb, hue_values = apply_pca_and_extract_hues(features, consensus_mask)
        
        if pca_features_rgb is not None:
            rgb_grid = create_rgb_grid(pca_features_rgb)
            all_rgb_grids[model_name] = rgb_grid
            all_hue_data[model_name] = hue_values
    
    # Create figures with DINOv2-only layout
    if all_rgb_grids:
        # Version without highlighting (for main paper)
        output_path = os.path.join(output_dir, f"{cancer_type}_{region_index}.png")
        create_final_figure_dinov2(original_image, all_rgb_grids, all_hue_data, output_path, 
                                   highlight_subplot=None)
        
        # Version with highlighting (for appendix) - highlight DINOv2 mixed by default
        output_path_highlighted = os.path.join(output_dir, f"{cancer_type}_{region_index}_highlighted.png")
        create_final_figure_dinov2(original_image, all_rgb_grids, all_hue_data, output_path_highlighted,
                                   highlight_subplot='DINOv2: mixed augmentations')

def process_wsi(wsi_path, cancer_type, output_dir, checkpoints, device, n_regions=10):
    """Process WSI with DINOv2 only"""
    
    print(f"\n{'#'*60}")
    print(f"# {cancer_type}")
    print(f"{'#'*60}")
    
    slide = None
    try:
        slide = openslide.OpenSlide(wsi_path)
        regions = find_tissue_regions_diverse(slide, min_tissue_ratio=0.5, 
                                             n_regions=n_regions, output_dir=output_dir)
        
        for i in range(len(regions)):
            process_region(slide, regions[i], i, checkpoints, device, output_dir, cancer_type)
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
        
    finally:
        if slide:
            slide.close()
        torch.cuda.empty_cache()

def main():
    # Updated output directory name
    base_output_dir = "pca_visualizations_with_polar_3_dinov2_rebuttal"
    
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
        'STAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-STAD_svs/svs/fa9ef6ca-2b68-4951-ae4e-0faa7f437569/TCGA-D7-A6ET-01Z-00-DX1.A4FF5141-6B2A-456B-9EA2-E5DE72156647.svs",
    }
    
    # Only DINOv2 checkpoints
    checkpoints = {
        'DINOv2: standard augmentations': '/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth',
        #'DINOv2: masked augmentations': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version3_ViT-B/logs/checkpoint.pth',
        'DINOv2: random augmentations': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version4_random_masking_ViT-B/logs/checkpoint.pth',
        'DINOv2: mixed augmentations': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version2_ViT-B/logs/checkpoint.pth',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    for cancer_type, wsi_path in wsi_paths.items():
        cancer_dir = os.path.join(base_output_dir, cancer_type)
        os.makedirs(cancer_dir, exist_ok=True)
        process_wsi(wsi_path, cancer_type, cancer_dir, checkpoints, device, n_regions=9)
    
    print(f"\nOutput files:")
    print(f"  Main paper: {base_output_dir}/CANCER_TYPE/CANCER_TYPE_0.png, CANCER_TYPE_1.png, ...")
    print(f"  Appendix (highlighted): {base_output_dir}/CANCER_TYPE/CANCER_TYPE_0_highlighted.png, ...")
    print("Note: Highlighting defaults to 'DINOv2: mixed augmentations'")
    print("      To change, modify highlight_subplot parameter in process_region()")

if __name__ == "__main__":
    main()
