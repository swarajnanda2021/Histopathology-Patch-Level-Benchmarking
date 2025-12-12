#!/usr/bin/env python3
"""
Combined t-SNE Visualization for CLS Tokens with Clustering Quality Metrics
- Clusters on original 768-dimensional CLS token features (principled approach)
- Visualizes clusters on t-SNE projection with colors
- Uses all CLS tokens (no subsampling needed - much fewer tokens than patch tokens)
- 2×2 grid layout suitable for papers
- Independent axis limits per checkpoint
- Uses consensus masking across checkpoints to filter background images
- Computes clustering quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import openslide
from scipy import ndimage
import time
from tqdm import tqdm
import argparse

# GPU-accelerated imports
try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE
    from cuml.decomposition import PCA as cuPCA
    from cuml.cluster import KMeans as cuKMeans
    import rmm
    RAPIDS_AVAILABLE = True
except ImportError:
    print("Warning: RAPIDS not available, falling back to CPU")
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    RAPIDS_AVAILABLE = False

# Clustering metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add path to your models
sys.path.append('/data1/vanderbc/nandas1/TCGA_TMEDinov2_ViT-B_version2_run2')
from soft_moe.vision_transformer import VisionTransformer

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ============================================================================
# GPU SETUP AND OPTIMIZATION
# ============================================================================

def setup_gpu_optimizations():
    """Configure GPU for maximum performance on A100"""
    if not RAPIDS_AVAILABLE:
        return
    
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    try:
        rmm.reinitialize(pool_allocator=True, 
                        initial_pool_size=2**34,
                        maximum_pool_size=2**36)
    except Exception as e:
        print(f"RMM initialization warning: {e}")
    
    import cuml
    cuml.set_global_output_type('numpy')
    
    print(f"GPU Optimizations enabled for A100")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CACHING SYSTEM
# ============================================================================

class FeatureCache:
    """Efficient caching system for features and masks"""
    
    def __init__(self, cache_dir="feature_cache_cls"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.max_memory_items = 50
    
    def _get_key(self, cancer_type, region_idx, checkpoint_name, data_type="features"):
        return f"{cancer_type}_{region_idx}_{checkpoint_name}_{data_type}"
    
    def _get_path(self, key):
        return self.cache_dir / f"{key}.npz"
    
    def exists(self, cancer_type, region_idx, checkpoint_name, data_type="features"):
        key = self._get_key(cancer_type, region_idx, checkpoint_name, data_type)
        return key in self.memory_cache or self._get_path(key).exists()
    
    def load(self, cancer_type, region_idx, checkpoint_name, data_type="features"):
        key = self._get_key(cancer_type, region_idx, checkpoint_name, data_type)
        
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        path = self._get_path(key)
        if path.exists():
            try:
                data = np.load(path)['data']
                if len(self.memory_cache) >= self.max_memory_items:
                    self.memory_cache.pop(next(iter(self.memory_cache)))
                self.memory_cache[key] = data
                return data
            except Exception as e:
                print(f"    Warning: Corrupted cache file {path}, removing it: {e}")
                path.unlink()
                return None
        return None
    
    def save(self, data, cancer_type, region_idx, checkpoint_name, data_type="features"):
        key = self._get_key(cancer_type, region_idx, checkpoint_name, data_type)
        path = self._get_path(key)
        np.savez_compressed(path, data=data)
        
        if len(self.memory_cache) >= self.max_memory_items:
            self.memory_cache.pop(next(iter(self.memory_cache)))
        self.memory_cache[key] = data

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_checkpoint(checkpoint_path, device, checkpoint_name='Starting'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        num_register_tokens=4,
    )
    
    if checkpoint_name == 'Starting':
        encoder_state = {}
        for k, v in checkpoint['student'].items():
            if k.startswith('module.backbone.'):
                encoder_state[k.replace('module.backbone.', '')] = v
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        encoder_state = {}
        for k, v in checkpoint['student'].items():
            if 'module.encoder.' in k:
                new_k = k.replace('module.encoder.', '')
                encoder_state[new_k] = v
        encoder.load_state_dict(encoder_state, strict=False)
    
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

# ============================================================================
# WSI PROCESSING
# ============================================================================

def find_tissue_regions_fast(slide, tissue_threshold=240, min_tissue_ratio=0.5, 
                             n_regions=10, thumbnail_size=1000):
    """Faster tissue region finding with reduced thumbnail size"""
    thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_np = np.array(thumbnail)
    
    gray = np.dot(thumbnail_np[...,:3], [0.299, 0.587, 0.114])
    tissue_mask = gray < tissue_threshold
    
    tissue_mask = ndimage.binary_erosion(tissue_mask, iterations=1)
    tissue_mask = ndimage.binary_dilation(tissue_mask, iterations=1)
    
    scale_x = slide.dimensions[0] / thumbnail.size[0]
    scale_y = slide.dimensions[1] / thumbnail.size[1]
    
    patch_size = 3584
    patch_size_thumb = int(patch_size / scale_x)
    
    valid_regions = []
    step = 30
    
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
                else:
                    continue
                
                valid_regions.append({
                    'location': (int(x * scale_x), int(y * scale_y)),
                    'tissue_ratio': tissue_ratio,
                    'darkness_score': darkness_score,
                    'center_x': x + patch_size_thumb // 2,
                    'center_y': y + patch_size_thumb // 2
                })
    
    if not valid_regions:
        raise ValueError("No valid tissue regions found!")
    
    for region in valid_regions:
        darkness_norm = 1.0 - (region['darkness_score'] / 240.0)
        region['combined_score'] = darkness_norm * 0.5 + region['tissue_ratio'] * 0.5
    
    valid_regions.sort(key=lambda x: x['combined_score'], reverse=True)
    
    selected_regions = []
    min_distance_thumb = patch_size_thumb * 0.5
    
    for region in valid_regions:
        if not selected_regions:
            selected_regions.append(region)
            continue
            
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
    
    return selected_regions

def extract_3584_region(slide, location):
    """Extract a 3584x3584 region from WSI"""
    x, y = location
    region = slide.read_region((x, y), 0, (3584, 3584))
    region = np.array(region)[:, :, :3]
    return region

def extract_patches_from_region(region_3584):
    """Divide 3584x3584 region into 256 224x224 patches"""
    patches = []
    for i in range(16):
        for j in range(16):
            patch = region_3584[i*224:(i+1)*224, j*224:(j+1)*224]
            patches.append(patch)
    return patches

# ============================================================================
# FEATURE EXTRACTION - CLS TOKEN
# ============================================================================

def normalize_batch(patches_batch):
    """Normalize a batch of patches efficiently"""
    mean = np.array([0.6816, 0.5640, 0.7232])
    std = np.array([0.1617, 0.1714, 0.1389])
    
    patches_normalized = patches_batch.astype(np.float32) / 255.0
    patches_normalized = (patches_normalized - mean) / std
    return patches_normalized

def extract_cls_features_optimized(model, patches, device, batch_size=32):
    """Optimized CLS token feature extraction - one CLS token per 224x224 image"""
    all_cls_features = []
    n_patches = len(patches)
    
    for i in range(0, n_patches, batch_size):
        batch_patches = patches[i:i+batch_size]
        
        patches_array = np.stack(batch_patches)
        patches_normalized = normalize_batch(patches_array)
        
        batch_tensor = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2).float().to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(batch_tensor)
                
                # Extract CLS token (first token, index 0)
                if isinstance(output, dict):
                    # The actual key is 'clstoken' (no underscore)
                    cls_token = output.get('clstoken')
                    if cls_token is None:
                        raise ValueError(f"Could not find CLS token. Available keys: {output.keys()}")
                else:
                    cls_token = output[:, 0]
                
                # cls_token shape: (batch_size, feature_dim)
                all_cls_features.append(cls_token)
    
    all_cls_features = torch.vstack(all_cls_features).cpu().numpy()
    return all_cls_features

# ============================================================================
# FOREGROUND DETECTION - IMAGE LEVEL FOR CLS TOKENS
# ============================================================================

def compute_foreground_mask_image_level(patches, brightness_threshold=220):
    """
    Image-level foreground detection for CLS tokens.
    Filter out images that are mostly background (too bright).
    
    Args:
        patches: list of 224x224 images
        brightness_threshold: images with mean brightness > this are considered background
    
    Returns:
        foreground_mask: boolean array indicating which images are foreground
    """
    print("  Computing image-level foreground mask...")
    
    foreground_mask = []
    
    for patch in patches:
        # Compute mean brightness of entire 224x224 image
        gray = np.dot(patch, [0.299, 0.587, 0.114])
        mean_brightness = np.mean(gray)
        
        # Image is foreground if it's not too bright
        is_foreground = mean_brightness < brightness_threshold
        foreground_mask.append(is_foreground)
    
    foreground_mask = np.array(foreground_mask)
    print(f"    Foreground images: {np.sum(foreground_mask)} / {len(foreground_mask)}")
    
    return foreground_mask

# ============================================================================
# T-SNE
# ============================================================================

def apply_tsne(features, **kwargs):
    """Apply t-SNE to features"""
    n_features = len(features)
    
    if n_features < 10:
        print("    Warning: Too few features for t-SNE")
        return None
    
    print(f"    Running t-SNE on {n_features:,} points...")
    
    perplexity = kwargs.get('perplexity', 100)
    n_iter = kwargs.get('n_iter', 2000)
    learning_rate = kwargs.get('learning_rate', 500.0)
    if learning_rate == 'auto':
        learning_rate = max(n_features / 12, 200.0)
    early_exaggeration = kwargs.get('early_exaggeration', 12.0)
    pca_components = kwargs.get('pca_components', 50)
    standardize = kwargs.get('standardize_features', True)
    
    if RAPIDS_AVAILABLE:
        features_gpu = cp.asarray(features)
        
        if standardize:
            from cuml.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_gpu = scaler.fit_transform(features_gpu)
        
        pca_components = min(pca_components, features_gpu.shape[1]-1, features_gpu.shape[0]-1)
        pca = cuPCA(n_components=pca_components)
        features_reduced = pca.fit_transform(features_gpu)
        
        perplexity = min(perplexity, len(features_reduced) // 4)
        tsne = cuTSNE(n_components=2, 
                      perplexity=perplexity,
                      n_iter=n_iter,
                      learning_rate=learning_rate,
                      early_exaggeration=early_exaggeration,
                      init='pca',
                      method='fft',
                      random_state=42,
                      verbose=0)
        
        tsne_coords = tsne.fit_transform(features_reduced)
        return cp.asnumpy(tsne_coords)
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        if standardize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        pca_components = min(pca_components, features.shape[1]-1, features.shape[0]-1)
        pca = PCA(n_components=pca_components)
        features_reduced = pca.fit_transform(features)
        
        perplexity = min(perplexity, len(features_reduced) // 4)
        tsne = TSNE(n_components=2, 
                    perplexity=perplexity,
                    n_iter=n_iter,
                    init='pca',
                    learning_rate=learning_rate,
                    early_exaggeration=early_exaggeration,
                    random_state=42,
                    verbose=0)
        
        return tsne.fit_transform(features_reduced)

# ============================================================================
# COMBINED FEATURE COLLECTION WITH CONSENSUS MASKING - CLS TOKENS
# ============================================================================

def collect_all_cls_features(wsi_paths, models, device, checkpoint_names, cache, n_regions=3, recompute=False):
    """
    Collect CLS token features from all WSIs and regions across all checkpoints.
    Uses CONSENSUS MASKING at image level across checkpoints.
    
    Returns: 
        all_features[checkpoint_name] = list of CLS feature arrays
    """
    print("\n" + "="*80)
    print("PHASE 1: COLLECTING CLS TOKEN FEATURES WITH CONSENSUS MASKING")
    print("="*80)
    
    all_features = {checkpoint_name: [] for checkpoint_name in checkpoint_names}
    
    for cancer_type, wsi_path in wsi_paths.items():
        print(f"\n{'#'*60}")
        print(f"# Processing {cancer_type}")
        print(f"{'#'*60}")
        
        if not os.path.exists(wsi_path):
            print(f"Warning: WSI not found: {wsi_path}")
            continue
        
        try:
            slide = openslide.OpenSlide(wsi_path)
            
            print("Finding tissue regions...")
            regions = find_tissue_regions_fast(slide, min_tissue_ratio=0.5, 
                                              n_regions=n_regions)
            
            print(f"Extracting {len(regions)} regions from WSI...")
            all_region_data = []
            for idx, region_info in enumerate(tqdm(regions, desc="Extracting")):
                original_image = extract_3584_region(slide, region_info['location'])
                patches = extract_patches_from_region(original_image)
                all_region_data.append({
                    'patches': patches,
                    'idx': idx
                })
            
            print("\nExtracting CLS features and computing image-level masks...")
            region_features = {}
            region_masks = {}
            
            for checkpoint_name in checkpoint_names:
                print(f"\n{checkpoint_name} checkpoint:")
                model = models[checkpoint_name]
                
                for region_data in tqdm(all_region_data, desc=f"  {checkpoint_name}"):
                    idx = region_data['idx']
                    
                    # Check cache unless recompute is True
                    if not recompute:
                        cached_features = cache.load(cancer_type, idx, checkpoint_name, "cls_features")
                        cached_mask = cache.load(cancer_type, idx, checkpoint_name, "cls_mask")
                    else:
                        cached_features = None
                        cached_mask = None
                    
                    if cached_features is not None and cached_mask is not None:
                        features = cached_features
                        mask = cached_mask
                    else:
                        # Extract CLS tokens (one per 224x224 image)
                        features = extract_cls_features_optimized(model, region_data['patches'], 
                                                                  device, batch_size=32)
                        # Image-level masking
                        mask = compute_foreground_mask_image_level(region_data['patches'])
                        
                        cache.save(features, cancer_type, idx, checkpoint_name, "cls_features")
                        cache.save(mask, cancer_type, idx, checkpoint_name, "cls_mask")
                    
                    region_features[(checkpoint_name, idx)] = features
                    region_masks[(checkpoint_name, idx)] = mask
            
            print("\nComputing CONSENSUS masks across checkpoints...")
            for region_data in all_region_data:
                idx = region_data['idx']
                
                masks_for_region = [region_masks[(ckpt, idx)] for ckpt in checkpoint_names]
                
                # Use intersection of masks (images that are foreground in ALL checkpoints)
                consensus_mask = masks_for_region[0].copy()
                for mask in masks_for_region[1:]:
                    consensus_mask = consensus_mask & mask
                
                # If intersection is too small, use union instead
                if np.sum(consensus_mask) < 50:
                    print(f"    Region {idx} ({cancer_type}): Intersection too small, using UNION")
                    consensus_mask = masks_for_region[0].copy()
                    for mask in masks_for_region[1:]:
                        consensus_mask = consensus_mask | mask
                
                n_consensus = np.sum(consensus_mask)
                print(f"  Region {idx} ({cancer_type}): {n_consensus:,} consensus foreground images")
                
                # Extract foreground CLS features
                for checkpoint_name in checkpoint_names:
                    features = region_features[(checkpoint_name, idx)]
                    foreground_features = features[consensus_mask]
                    all_features[checkpoint_name].append(foreground_features)
            
            slide.close()
            torch.cuda.empty_cache()
            if RAPIDS_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
        except Exception as e:
            print(f"Error processing {cancer_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_features

# ============================================================================
# AGGREGATION, CLUSTERING, AND T-SNE - NO SUBSAMPLING
# ============================================================================

def aggregate_cluster_and_analyze(all_features, checkpoint_names, cache, n_clusters=10, 
                                  recompute=False, **tsne_params):
    """
    Aggregate CLS features per checkpoint (no subsampling needed), cluster on original features,
    and run t-SNE for visualization.
    
    Returns: 
        original_features[checkpoint] - original 768D features
        cluster_labels[checkpoint] - cluster assignments
        tsne_coords[checkpoint] - 2D t-SNE coordinates
    """
    print("\n" + "="*80)
    print("PHASE 2: AGGREGATING, CLUSTERING (768D), AND RUNNING T-SNE")
    print("="*80)
    
    original_features = {}
    cluster_labels = {}
    tsne_coords = {}
    
    for checkpoint_name in checkpoint_names:
        print(f"\n{'='*60}")
        print(f"Processing checkpoint: {checkpoint_name}")
        print(f"{'='*60}")
        
        # Concatenate all CLS features for this checkpoint
        feature_list = all_features[checkpoint_name]
        combined_features = np.vstack(feature_list)
        n_total = combined_features.shape[0]
        
        print(f"Total CLS tokens: {n_total:,} (one per 224x224 image)")
        print(f"Feature dimension: {combined_features.shape[1]}")
        print(f"No subsampling needed - using all CLS tokens")
        
        # Store original features for clustering metrics
        original_features[checkpoint_name] = combined_features
        
        # CLUSTER ON ORIGINAL 768D FEATURES
        print(f"  Clustering on original {combined_features.shape[1]}D features with k={n_clusters}...")
        start_time = time.time()
        
        if RAPIDS_AVAILABLE:
            features_gpu = cp.asarray(combined_features)
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features_gpu)
            labels = cp.asnumpy(labels)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(combined_features)
        
        elapsed = time.time() - start_time
        print(f"  Clustering completed in {elapsed:.1f}s")
        cluster_labels[checkpoint_name] = labels
        
        # t-SNE with caching
        param_str = '_'.join([f"{k}{v}" for k, v in sorted(tsne_params.items())])
        cache_key = f"CLS_ALL_{n_total}_{checkpoint_name}_tsne_{param_str}"
        
        # Check cache unless recompute is True
        if not recompute:
            cached_tsne = cache.load("COMBINED", 0, cache_key, "tsne")
        else:
            cached_tsne = None
        
        if cached_tsne is not None:
            print("  Loading t-SNE from cache...")
            tsne_result = cached_tsne
        else:
            print("  Running t-SNE for visualization...")
            start_time = time.time()
            tsne_result = apply_tsne(combined_features, **tsne_params)
            elapsed = time.time() - start_time
            print(f"  t-SNE completed in {elapsed:.1f}s")
            
            if tsne_result is not None:
                cache.save(tsne_result, "COMBINED", 0, cache_key, "tsne")
        
        tsne_coords[checkpoint_name] = tsne_result
    
    return original_features, cluster_labels, tsne_coords

# ============================================================================
# CLUSTERING METRICS ON ORIGINAL FEATURES
# ============================================================================

def compute_clustering_metrics(original_features, cluster_labels):
    """
    Compute clustering quality metrics on ORIGINAL 768D features (principled approach).
    
    Args:
        original_features: numpy array of shape (n_samples, 768)
        cluster_labels: numpy array of shape (n_samples,) with cluster assignments
    
    Returns:
        dict with 'silhouette', 'davies_bouldin', 'calinski_harabasz'
    """
    if original_features is None or cluster_labels is None or len(original_features) < 10:
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None
        }
    
    # Compute metrics on ORIGINAL high-dimensional features
    silhouette = silhouette_score(original_features, cluster_labels)
    davies_bouldin = davies_bouldin_score(original_features, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(original_features, cluster_labels)
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

# ============================================================================
# VISUALIZATION - 2×2 GRID WITH COLORED CLUSTERS
# ============================================================================

def create_combined_visualization(tsne_coords, cluster_labels, checkpoint_names, output_path):
    """
    Create 2×2 grid t-SNE visualization with clusters colored.
    Each plot has independent axis limits for optimal viewing.
    """
    print("\n" + "="*80)
    print("PHASE 3: CREATING 2×2 VISUALIZATION WITH COLORED CLUSTERS")
    print("="*80)
    
    n_checkpoints = len(checkpoint_names)
    
    # 2×2 GRID LAYOUT
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Define a colormap for clusters
    cmap = plt.cm.get_cmap('tab10')
    
    # Plot each checkpoint with colored clusters
    for idx, checkpoint_name in enumerate(checkpoint_names):
        ax = axes[idx]
        
        if checkpoint_name in tsne_coords and tsne_coords[checkpoint_name] is not None:
            coords = tsne_coords[checkpoint_name]
            labels = cluster_labels[checkpoint_name]
            
            # Get number of unique clusters
            n_clusters = len(np.unique(labels))
            
            # Scatter plot with cluster colors
            scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                               c=labels, cmap=cmap, 
                               s=1.5, alpha=0.6, rasterized=True)
            
            # INDEPENDENT axis limits - let each plot find its own natural bounds
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            # Add small padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = 0.05 * x_range
            y_padding = 0.05 * y_range
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(checkpoint_name, fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.2)
    
    # Adjust spacing
    plt.tight_layout(pad=2.0)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Visualization saved: {output_path}")

# ============================================================================
# CLUSTERING METRICS TABLE
# ============================================================================

def print_clustering_metrics_table(original_features, cluster_labels, checkpoint_names):
    """Print a formatted table of clustering metrics computed on original 768D CLS features"""
    print("\n" + "="*80)
    print("CLS TOKEN CLUSTERING METRICS (computed on original 768D features)")
    print("="*80)
    
    # Compute metrics for all checkpoints
    all_metrics = {}
    for checkpoint_name in checkpoint_names:
        features = original_features.get(checkpoint_name)
        labels = cluster_labels.get(checkpoint_name)
        metrics = compute_clustering_metrics(features, labels)
        all_metrics[checkpoint_name] = metrics
    
    # Print table header
    print(f"{'Model':<20} {'Silhouette':>15} {'Davies-Bouldin':>18} {'Calinski-Harabasz':>20}")
    print("-" * 80)
    
    # Print metrics for each model
    for checkpoint_name in checkpoint_names:
        metrics = all_metrics[checkpoint_name]
        
        if metrics['silhouette'] is not None:
            sil = f"{metrics['silhouette']:.3f}"
            db = f"{metrics['davies_bouldin']:.3f}"
            ch = f"{metrics['calinski_harabasz']:.2f}"
        else:
            sil = "N/A"
            db = "N/A"
            ch = "N/A"
        
        print(f"{checkpoint_name:<20} {sil:>15} {db:>18} {ch:>20}")
    
    print("=" * 80)
    print("\nNote: Higher Silhouette and Calinski-Harabasz are better.")
    print("      Lower Davies-Bouldin is better.")
    print(f"      Metrics computed on original 768D CLS token features (principled approach).")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main processing function for CLS token t-SNE visualization"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CLS token t-SNE visualization with clustering metrics')
    parser.add_argument('--recompute', action='store_true', 
                       help='Recompute all features and t-SNE even if cached')
    parser.add_argument('--n_clusters', type=int, default=10,
                       help='Number of clusters for k-means (default: 10)')
    args = parser.parse_args()
    
    # ============================================================================
    # t-SNE PARAMETERS
    # ============================================================================
    
    TSNE_PARAMS = {
        'perplexity': 15,             # Good for smaller number of CLS tokens
        'n_iter': 500,
        'learning_rate': 'auto',
        'early_exaggeration': 12.0,
        'pca_components': 768,
        'standardize_features': True,
    }
    
    # ============================================================================
    
    setup_gpu_optimizations()
    
    cache = FeatureCache("feature_cache_cls")
    
    output_dir = "tsne_cls_output"
    os.makedirs(output_dir, exist_ok=True)
    
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
        'STAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-STAD_svs/svs/fa9ef6ca-2b68-4951-ae4e-0faa7f437569/TCGA-D7-A6ET-01Z-00-DX1.A4FF5141-6B2A-456B-9EA2-E5DE72156647.svs",
    }
    
    checkpoint_paths = {
        'Standard': "/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2/logs/checkpoint.pth",
        'Mixed': "/data1/vanderbc/nandas1/TCGA_TMEDinov2_version2_ViT-B/logs/checkpoint.pth",
        'Masked': "/data1/vanderbc/nandas1/TCGA_TMEDinov2_version3_ViT-B/logs/checkpoint.pth",
        'Random': "/data1/vanderbc/nandas1/TCGA_TMEDinov2_version4_random_masking_ViT-B/logs/checkpoint.pth",
    }
    
    checkpoint_names = list(checkpoint_paths.keys())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Method: CLS token clustering on 768D features + t-SNE visualization")
    print(f"Recompute mode: {'ENABLED (ignoring cache)' if args.recompute else 'DISABLED (using cache)'}")
    print(f"t-SNE Parameters: {TSNE_PARAMS}")
    print(f"Using ALL CLS tokens (no subsampling)")
    print(f"Clustering: k={args.n_clusters} clusters on original 768D CLS features")
    
    if RAPIDS_AVAILABLE:
        print(f"RAPIDS cuML acceleration: ENABLED")
    else:
        print(f"RAPIDS cuML acceleration: DISABLED (CPU fallback)")
    
    print(f"Checkpoints: {checkpoint_names}")
    print(f"Cancer types: {list(wsi_paths.keys())}")
    
    # Load all models
    print("\nLoading models...")
    models = {}
    for checkpoint_name, checkpoint_path in checkpoint_paths.items():
        print(f"  Loading {checkpoint_name}...")
        start_time = time.time()
        models[checkpoint_name] = load_model_checkpoint(checkpoint_path, device, checkpoint_name)
        print(f"    Loaded in {time.time() - start_time:.1f}s")
    
    print(f"\nGPU Memory after loading models: "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # PHASE 1: Collect all CLS features with consensus masking
    total_start = time.time()
    all_features = collect_all_cls_features(wsi_paths, models, device, checkpoint_names, 
                                           cache, n_regions=3, recompute=args.recompute)
    
    # PHASE 2: Aggregate, cluster on 768D, and run t-SNE (no subsampling)
    original_features, cluster_labels, tsne_coords = aggregate_cluster_and_analyze(
        all_features, checkpoint_names, cache, 
        n_clusters=args.n_clusters,
        recompute=args.recompute, 
        **TSNE_PARAMS
    )
    
    # PHASE 3: Create 2×2 visualization with colored clusters
    output_path = os.path.join(output_dir, "cls_tsne_2x2_clustered.png")
    create_combined_visualization(tsne_coords, cluster_labels, checkpoint_names, output_path)
    
    # PHASE 4: Print clustering metrics table (computed on original 768D CLS features)
    print_clustering_metrics_table(original_features, cluster_labels, checkpoint_names)
    
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Output: {output_path}")
    print(f"Cache: {cache.cache_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
