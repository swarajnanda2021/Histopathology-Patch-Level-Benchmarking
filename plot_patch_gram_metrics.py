#!/usr/bin/env python3
"""
Comprehensive Patch Clustering Analysis
Analyzes multiple clustering metrics: Effective Rank, Soft K-means, Entropy, and Sinkhorn
Creates separate plots for each metric with consistent styling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import pickle
import os
from torch.utils.data import DataLoader
from scipy import stats

# Add paths for imports
sys.path.append('/data1/vanderbc/nandas1/PostProc')
sys.path.append('/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2')

# Set ggplot style and professional font
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.facecolor'] = '#F5F5F5'
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

# Define consistent colors
AUGMENTATION_COLORS = {
    'standard': '#1E88E5',  # Professional blue
    'mixed': '#FFA726',      # Professional orange
    'masked': '#DC3545',     # Professional red
}

# Map model names to augmentation types
MODEL_TO_AUGMENTATION = {
    'Dino_ViT-B': 'standard',
    'TMEDinov1_ViT-B_version2': 'mixed',
    'TMEDinov1_ViT-B_version3': 'masked',
    'Dinov2_ViT-B': 'standard',
    'TMEDinov2_ViT-B_version2': 'mixed',
    'TMEDinov2_ViT-B_version3': 'masked'
}

# Explicit mapping of models to SSL methods
SSL_METHOD_MAPPING = {
    'Dinov1': ['Dino_ViT-B', 'TMEDinov1_ViT-B_version2', 'TMEDinov1_ViT-B_version3'],
    'Dinov2': ['Dinov2_ViT-B', 'TMEDinov2_ViT-B_version2', 'TMEDinov2_ViT-B_version3']
}

# Different markers for visual distinction
VARIANT_MARKERS = {
    'standard': 'o',    # Circle
    'mixed': 's',        # Square  
    'masked': '^'        # Triangle
}


class ComprehensiveClusteringAnalyzer:
    """
    Analyzes multiple clustering metrics for patch representations.
    """
    
    def __init__(self):
        self.results_cache = {}
        
    def get_checkpoints_at_intervals(self, checkpoint_dir: Path, 
                                    interval: int = 10000, 
                                    max_iter: int = 300000) -> List[Tuple[int, str]]:
        """Get checkpoint paths at specified intervals."""
        checkpoints = []
        checkpoint_dir = Path(checkpoint_dir)
        
        # Check for iteration 0 checkpoint
        iter0_path = checkpoint_dir / 'logs' / 'checkpoint_iter_00000000.pth'
        if iter0_path.exists():
            checkpoints.append((0, str(iter0_path)))
        
        # Get checkpoints at regular intervals
        for iteration in range(interval, min(max_iter + 1, 300001), interval):
            ckpt_path = checkpoint_dir / 'logs' / f'checkpoint_iter_{iteration:08d}.pth'
            if ckpt_path.exists():
                checkpoints.append((iteration, str(ckpt_path)))
        
        # Add final checkpoint if it exists
        final_path = checkpoint_dir / 'logs' / f'checkpoint_iter_{max_iter-1:08d}.pth'
        if final_path.exists() and (max_iter-1, str(final_path)) not in checkpoints:
            checkpoints.append((max_iter-1, str(final_path)))
            
        print(f"  Found {len(checkpoints)} checkpoints for {checkpoint_dir.name}")
        return checkpoints
    
    def extract_patch_embeddings(self, model, images: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from the vision transformer model."""
        model.eval()
        
        # Model configuration
        num_registers = getattr(model, 'numregisters', 4)
        has_cls = model.cls_token is not None
        
        with torch.no_grad():
            # Forward pass through the model
            x = model.prepare_tokens(images)
            x = model.patch_drop(x)
            x = model.norm_pre(x)
            
            # Process through all transformer blocks
            for block in model.blocks:
                x = block(x)
            
            # Apply final normalization
            x = model.norm(x)
            
            # Extract only patch tokens (exclude CLS and register tokens)
            if has_cls:
                patch_embeddings = x[:, 1 + num_registers:, :]
            else:
                patch_embeddings = x[:, num_registers:, :]
        
        return patch_embeddings
    
    def compute_effective_rank(self, patch_embeddings: torch.Tensor) -> List[float]:
        """Compute effective rank for each image."""
        B, N, D = patch_embeddings.shape
        
        # Normalize embeddings
        patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
        
        # Compute Gram matrix
        gram = torch.bmm(patch_embeddings, patch_embeddings.transpose(1, 2))
        
        # Compute SVD
        try:
            U, S, V = torch.svd(gram)
        except:
            gram_cpu = gram.cpu().double()
            U, S, V = torch.svd(gram_cpu)
            S = S.float().cuda()
        
        # Calculate effective rank for each image
        effective_ranks = []
        for i in range(B):
            singular_values = S[i]
            singular_sum = singular_values.sum()
            if singular_sum > 0:
                normalized_singular = singular_values / singular_sum
                entropy = -(normalized_singular * torch.log(normalized_singular + 1e-10)).sum()
                effective_rank = torch.exp(entropy).item()
                effective_ranks.append(effective_rank)
        
        return effective_ranks
    
    def compute_soft_kmeans(self, patch_embeddings: torch.Tensor, 
                           n_clusters: int = 5, 
                           n_iterations: int = 3,
                           temperature: float = 0.1) -> List[float]:
        """Compute soft k-means clustering loss for each image."""
        B, N, D = patch_embeddings.shape
        device = patch_embeddings.device
        
        losses = []
        
        for b in range(B):
            embeddings = patch_embeddings[b]  # (N, D)
            
            # Initialize cluster centers randomly
            indices = torch.randperm(N)[:n_clusters].to(device)
            centers = embeddings[indices].clone()  # (K, D)
            
            for _ in range(n_iterations):
                # Compute distances from patches to centers
                distances = torch.cdist(embeddings.unsqueeze(0), centers.unsqueeze(0), p=2).squeeze(0)  # (N, K)
                
                # Compute soft assignments
                assignments = F.softmax(-distances / temperature, dim=-1)  # (N, K)
                
                # Update centers as weighted average
                centers = torch.einsum('nk,nd->kd', assignments, embeddings)
                normalizer = assignments.sum(dim=0, keepdim=True).T  # (K, 1)
                centers = centers / (normalizer + 1e-8)
            
            # Final loss: weighted distances
            final_distances = torch.cdist(embeddings.unsqueeze(0), centers.unsqueeze(0), p=2).squeeze(0)
            final_assignments = F.softmax(-final_distances / temperature, dim=-1)
            loss = (final_assignments * final_distances).sum(dim=-1).mean().item()
            losses.append(loss)
        
        return losses
    
    def compute_entropy_clustering(self, patch_embeddings: torch.Tensor,
                                  temperature: float = 0.1) -> List[float]:
        """Compute entropy-based clustering metric for each image."""
        B, N, D = patch_embeddings.shape
        
        # Normalize embeddings for cosine similarity
        patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
        
        entropies = []
        
        for b in range(B):
            embeddings = patch_embeddings[b]  # (N, D)
            
            # Compute cosine similarity matrix
            similarity_matrix = torch.mm(embeddings, embeddings.t())  # (N, N)
            
            # Convert to probability distribution per patch
            prob_matrix = F.softmax(similarity_matrix / temperature, dim=-1)  # (N, N)
            
            # Compute entropy for each patch's similarity distribution
            entropy = -(prob_matrix * torch.log(prob_matrix + 1e-10)).sum(dim=-1)  # (N,)
            
            # Average entropy across all patches
            avg_entropy = entropy.mean().item()
            entropies.append(avg_entropy)
        
        return entropies
    
    def compute_sinkhorn(self, patch_embeddings: torch.Tensor,
                        n_clusters: int = 5,
                        n_iterations: int = 10,
                        epsilon: float = 0.1) -> List[float]:
        """Compute Sinkhorn-based clustering metric for each image."""
        B, N, D = patch_embeddings.shape
        device = patch_embeddings.device
        
        sinkhorn_losses = []
        
        for b in range(B):
            embeddings = patch_embeddings[b]  # (N, D)
            
            # Initialize cluster prototypes
            indices = torch.randperm(N)[:n_clusters].to(device)
            prototypes = embeddings[indices].clone()  # (K, D)
            
            # Compute cost matrix (distances)
            C = torch.cdist(embeddings.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)  # (N, K)
            
            # Initialize uniform distributions
            a = torch.ones(N, device=device) / N  # Source distribution
            b = torch.ones(n_clusters, device=device) / n_clusters  # Target distribution
            
            # Sinkhorn iterations
            K = torch.exp(-C / epsilon)  # (N, K)
            u = torch.ones(N, device=device)
            
            for _ in range(n_iterations):
                v = b / (K.t() @ u + 1e-10)
                u = a / (K @ v + 1e-10)
            
            # Transport plan
            T = torch.diag(u) @ K @ torch.diag(v)  # (N, K)
            
            # Compute transport cost
            transport_cost = (T * C).sum().item()
            sinkhorn_losses.append(transport_cost)
        
        return sinkhorn_losses
    
    def analyze_single_model_all_metrics(self, checkpoint_dir: str, 
                                        model_name: str,
                                        interval: int = 10000, 
                                        max_iter: int = 300000,
                                        batch_size: int = 8, 
                                        n_batches: int = 10) -> Dict:
        """
        Analyze all clustering metrics for a single model.
        First checks for existing cache files from effective rank analysis.
        """
        # Check for existing effective rank cache
        rank_cache_key = f"{model_name}_rank_stats.pkl"
        metrics_cache_key = f"{model_name}_all_clustering_metrics.pkl"
        
        # Try to load comprehensive metrics cache first
        if os.path.exists(metrics_cache_key):
            print(f"  Loading cached comprehensive metrics for {model_name}")
            with open(metrics_cache_key, 'rb') as f:
                return pickle.load(f)
        
        # Load test dataset
        from DataAug_TME import MemoryEfficientShardedPathologyDataset
        
        dataset = MemoryEfficientShardedPathologyDataset(
            base_dir="/data1/vanderbc/foundation_model_training_images/TCGA",
            index_file="dataset_index.pkl",
            worker_id=0,
            num_workers=1,
            rank=0,
            world_size=1,
            seed=42,
            global_size=224,
            local_size=96,
            n_local_crops=8,
            local_crop_scale=(0.05, 0.4),
            global_crop_scale=(0.4, 1.0),
            mean=(0.6816, 0.5640, 0.7232),
            std=(0.1617, 0.1714, 0.1389)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        # Collect test batches
        all_test_images = []
        batch_count = 0
        
        print(f"  Collecting {n_batches} batches for analysis...")
        for batch_data in dataloader:
            if batch_count >= n_batches:
                break
            
            if isinstance(batch_data, (list, tuple)):
                test_images = batch_data[0]
            else:
                test_images = batch_data
            
            if len(test_images.shape) == 5:
                test_images = test_images[:, 0]
            
            all_test_images.append(test_images)
            batch_count += 1
        
        # Get checkpoints
        checkpoints = self.get_checkpoints_at_intervals(
            Path(checkpoint_dir), interval, max_iter
        )
        
        # Import model loader
        from utils import load_dino_backbone
        
        # Initialize results dictionary for all metrics
        results = {
            'iterations': [],
            'effective_rank': {'mean': [], 'std': [], 'ci_lower': [], 'ci_upper': []},
            'soft_kmeans': {'mean': [], 'std': [], 'ci_lower': [], 'ci_upper': []},
            'entropy': {'mean': [], 'std': [], 'ci_lower': [], 'ci_upper': []},
            'sinkhorn': {'mean': [], 'std': [], 'ci_lower': [], 'ci_upper': []},
            'model_name': model_name
        }
        
        # If we have existing rank cache, load it for effective rank
        if os.path.exists(rank_cache_key):
            print(f"    Using cached effective rank data from {rank_cache_key}")
            with open(rank_cache_key, 'rb') as f:
                rank_data = pickle.load(f)
                if 'iterations' in rank_data and 'mean_rank' in rank_data:
                    # Use the cached effective rank data
                    cached_iterations = rank_data['iterations']
                    results['effective_rank']['mean'] = rank_data['mean_rank']
                    results['effective_rank']['std'] = rank_data['std_rank']
                    results['effective_rank']['ci_lower'] = rank_data['ci_lower']
                    results['effective_rank']['ci_upper'] = rank_data['ci_upper']
        
        # Analyze each checkpoint
        for idx, (iteration, ckpt_path) in enumerate(checkpoints):
            if idx % 5 == 0:
                print(f"    Processing checkpoint {iteration} ({idx+1}/{len(checkpoints)})")
            
            # Skip if we already have effective rank data for this iteration
            skip_effective_rank = (os.path.exists(rank_cache_key) and 
                                  results['effective_rank']['mean'] and 
                                  len(results['effective_rank']['mean']) > idx)
            
            try:
                # Load model
                model = load_dino_backbone(ckpt_path, 'cuda')
                
                # Collect all metrics for this checkpoint
                all_effective_ranks = [] if not skip_effective_rank else None
                all_soft_kmeans = []
                all_entropies = []
                all_sinkhorn = []
                
                for test_images in all_test_images:
                    test_images = test_images.cuda()
                    patch_embeddings = self.extract_patch_embeddings(model, test_images)
                    
                    # Compute metrics for each image in this batch
                    if not skip_effective_rank:
                        batch_ranks = self.compute_effective_rank(patch_embeddings)
                        all_effective_ranks.extend(batch_ranks)
                    
                    batch_kmeans = self.compute_soft_kmeans(patch_embeddings)
                    all_soft_kmeans.extend(batch_kmeans)
                    
                    batch_entropy = self.compute_entropy_clustering(patch_embeddings)
                    all_entropies.extend(batch_entropy)
                    
                    batch_sinkhorn = self.compute_sinkhorn(patch_embeddings)
                    all_sinkhorn.extend(batch_sinkhorn)
                
                # Process and store results for each metric
                results['iterations'].append(iteration)
                
                # Helper function to compute statistics
                def compute_stats(values):
                    values = np.array(values)
                    # Remove outliers using IQR
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    filtered = values[(values >= lower_bound) & (values <= upper_bound)]
                    
                    if len(filtered) >= 3:
                        mean_val = np.mean(filtered)
                        std_val = np.std(filtered, ddof=1)
                        std_error = std_val / np.sqrt(len(filtered))
                        
                        # 95% CI using t-distribution
                        confidence = 0.95
                        df = len(filtered) - 1
                        t_value = stats.t.ppf((1 + confidence) / 2, df)
                        ci_lower = mean_val - t_value * std_error
                        ci_upper = mean_val + t_value * std_error
                        
                        return mean_val, std_val, ci_lower, ci_upper
                    return None, None, None, None
                
                # Compute statistics for each metric
                if not skip_effective_rank and all_effective_ranks:
                    mean_val, std_val, ci_lower, ci_upper = compute_stats(all_effective_ranks)
                    if mean_val is not None:
                        results['effective_rank']['mean'].append(mean_val)
                        results['effective_rank']['std'].append(std_val)
                        results['effective_rank']['ci_lower'].append(ci_lower)
                        results['effective_rank']['ci_upper'].append(ci_upper)
                
                mean_val, std_val, ci_lower, ci_upper = compute_stats(all_soft_kmeans)
                if mean_val is not None:
                    results['soft_kmeans']['mean'].append(mean_val)
                    results['soft_kmeans']['std'].append(std_val)
                    results['soft_kmeans']['ci_lower'].append(ci_lower)
                    results['soft_kmeans']['ci_upper'].append(ci_upper)
                
                mean_val, std_val, ci_lower, ci_upper = compute_stats(all_entropies)
                if mean_val is not None:
                    results['entropy']['mean'].append(mean_val)
                    results['entropy']['std'].append(std_val)
                    results['entropy']['ci_lower'].append(ci_lower)
                    results['entropy']['ci_upper'].append(ci_upper)
                
                mean_val, std_val, ci_lower, ci_upper = compute_stats(all_sinkhorn)
                if mean_val is not None:
                    results['sinkhorn']['mean'].append(mean_val)
                    results['sinkhorn']['std'].append(std_val)
                    results['sinkhorn']['ci_lower'].append(ci_lower)
                    results['sinkhorn']['ci_upper'].append(ci_upper)
                
                # Clean up memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error at iteration {iteration}: {e}")
                continue
        
        # Cache comprehensive results
        with open(metrics_cache_key, 'wb') as f:
            pickle.dump(results, f)
        print(f"  Saved comprehensive metrics cache: {metrics_cache_key}")
        
        return results
    
    def format_iteration_axis(self, ax, iterations):
        """Format x-axis to show iterations in k units."""
        def formatter(x, pos):
            if len(iterations) == 0:
                return ''
            
            idx = int(round(x))
            if idx < 0 or idx >= len(iterations):
                return ''
            
            iter_val = iterations[idx]
            if iter_val == 0:
                return '0'
            elif iter_val >= 1000:
                return f'{int(iter_val/1000)}k'
            else:
                return str(int(iter_val))
        
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.xaxis.set_major_formatter(FuncFormatter(formatter))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    
    def create_metric_plot(self, all_results: List[Dict], metric_name: str, 
                          metric_label: str, output_path: str,
                          reverse_y: bool = False):
        """
        Create a 1×2 plot for a specific metric comparing all models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # SSL methods
        ssl_methods = ['Dinov1', 'Dinov2']
        
        # Store lines for legend
        legend_lines = {}
        
        # Process each SSL method
        for col_idx, ssl_method in enumerate(ssl_methods):
            ax = axes[col_idx]
            
            # Filter results for this SSL method
            model_names_for_ssl = SSL_METHOD_MAPPING[ssl_method]
            ssl_results = [r for r in all_results if r['model_name'] in model_names_for_ssl]
            
            # Plot each model variant with error bars
            for result in ssl_results:
                model_name = result['model_name']
                
                # Determine augmentation type and styling
                aug_type = MODEL_TO_AUGMENTATION.get(model_name, 'standard')
                color = AUGMENTATION_COLORS[aug_type]
                marker = VARIANT_MARKERS[aug_type]
                
                # Set label for legend
                if aug_type == 'standard':
                    label = 'Standard'
                elif aug_type == 'mixed':
                    label = 'Mixed'
                else:
                    label = 'Masked'
                
                # Get data for this metric
                metric_data = result[metric_name]
                if not metric_data['mean']:
                    continue
                
                # Ensure we only use as many iterations as we have data points
                n_points = len(metric_data['mean'])
                iterations = result['iterations'][:n_points]
                x_points = list(range(n_points))
                y_means = metric_data['mean'][:n_points]
                
                # Calculate error bars (ensure same length)
                yerr_lower = [y_means[i] - metric_data['ci_lower'][i] for i in range(n_points)]
                yerr_upper = [metric_data['ci_upper'][i] - y_means[i] for i in range(n_points)]
                
                # Plot with error bars
                line = ax.errorbar(x_points, y_means,
                                  yerr=[yerr_lower, yerr_upper],
                                  color=color,
                                  marker=marker,
                                  markersize=6,
                                  linewidth=2.5,
                                  capsize=3,
                                  capthick=1.2,
                                  alpha=0.9,
                                  elinewidth=1.2,
                                  label=label if col_idx == 0 else None)
                
                if aug_type not in legend_lines:
                    legend_lines[aug_type] = (line, label)
            
            # Format x-axis
            if ssl_results and result['iterations']:
                self.format_iteration_axis(ax, result['iterations'])
            
            # Set title
            ax.set_title(f'{ssl_method}', fontsize=12, fontweight='bold', pad=10)
            
            # Set axis labels
            ax.set_xlabel('Training Iteration', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            
            # Reverse y-axis if needed (for metrics where lower is better)
            if reverse_y:
                ax.invert_yaxis()
            
            # Style the axis
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            
            ax.tick_params(axis='both', labelsize=10, width=1.2,
                         colors='#333333', length=5)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
        
        # Synchronize y-axis limits
        y_mins = []
        y_maxs = []
        
        for ax in axes:
            if len(ax.lines) > 0:
                y_min, y_max = ax.get_ylim()
                y_mins.append(y_min)
                y_maxs.append(y_max)
        
        if y_mins and y_maxs:
            overall_min = min(y_mins)
            overall_max = max(y_maxs)
            y_range = overall_max - overall_min
            
            padding = 0.05 * y_range if y_range > 0 else 0.5
            
            for ax in axes:
                if reverse_y:
                    ax.set_ylim(overall_max + padding, overall_min - padding)
                else:
                    ax.set_ylim(overall_min - padding, overall_max + padding)
        
        # Add legend
        if legend_lines:
            handles = []
            labels = []
            for aug_type in ['standard', 'mixed', 'masked']:
                if aug_type in legend_lines:
                    handles.append(legend_lines[aug_type][0])
                    labels.append(legend_lines[aug_type][1])
            
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                      ncol=3, fontsize=11, frameon=True, fancybox=False,
                      edgecolor='#333333', framealpha=0.95)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved {metric_label} plot: {output_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("COMPREHENSIVE PATCH CLUSTERING ANALYSIS")
    print("Computing: Effective Rank, Soft K-means, Entropy, and Sinkhorn metrics")
    print("=" * 70)
    
    # Define model configurations
    model_configs = [
        # Dinov1 models
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_Dino_ViT-B_run2',
            'model_name': 'Dino_ViT-B'
        },
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_TMEDinov1_version2_ViT-B',
            'model_name': 'TMEDinov1_ViT-B_version2'
        },
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_TMEDinov1_version3_ViT-B',
            'model_name': 'TMEDinov1_ViT-B_version3'
        },
        # Dinov2 models
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2',
            'model_name': 'Dinov2_ViT-B'
        },
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version2_ViT-B',
            'model_name': 'TMEDinov2_ViT-B_version2'
        },
        {
            'checkpoint_dir': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version3_ViT-B',
            'model_name': 'TMEDinov2_ViT-B_version3'
        }
    ]
    
    # Initialize analyzer
    analyzer = ComprehensiveClusteringAnalyzer()
    
    # Analyze all models
    all_results = []
    
    print("\nPhase 1: Computing clustering metrics for all models")
    print("-" * 50)
    
    for config in model_configs:
        print(f"\nAnalyzing {config['model_name']}...")
        results = analyzer.analyze_single_model_all_metrics(
            checkpoint_dir=config['checkpoint_dir'],
            model_name=config['model_name'],
            interval=10000,
            max_iter=300000,
            batch_size=8,
            n_batches=10
        )
        all_results.append(results)
    
    # Create separate plots for each metric
    print("\nPhase 2: Creating visualizations")
    print("-" * 50)
    
    # Plot configurations: (metric_name, label, output_file, reverse_y_axis)
    plot_configs = [
        ('effective_rank', 'Effective Rank', 'effective_rank_comparison.png', False),
        ('soft_kmeans', 'Soft K-means Loss', 'soft_kmeans_comparison.png', False),
        ('entropy', 'Entropy', 'entropy_comparison.png', False),
        ('sinkhorn', 'Sinkhorn Distance', 'sinkhorn_comparison.png', False)
    ]
    
    for metric_name, label, output_file, reverse_y in plot_configs:
        analyzer.create_metric_plot(
            all_results,
            metric_name,
            label,
            output_file,
            reverse_y
        )
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("✓ Generated plots:")
    print("  - effective_rank_comparison.png")
    print("  - soft_kmeans_comparison.png")
    print("  - entropy_comparison.png")
    print("  - sinkhorn_comparison.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
