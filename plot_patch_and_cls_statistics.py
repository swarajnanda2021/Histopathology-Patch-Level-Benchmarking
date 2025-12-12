#!/usr/bin/env python3
"""
Script to create task-agnostic metrics plot comparing three model variants 
(Standard, Mixed, Masked) for Dinov1 and Dinov2 SSL methods.
Creates a 2 rows × 4 columns plot for RankMe (CLS), α-ReQ (CLS), CLID (CLS), and Entropy (Patch).
Uses ggplot styling with professional formatting optimized for ICLR A4 paper.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add paths for imports (for entropy calculation)
sys.path.append('/data1/vanderbc/nandas1/PostProc')
sys.path.append('/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2')

# Set ggplot style and professional font - adjusted for ICLR paper
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 13  # Reduced for paper
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.facecolor'] = '#F5F5F5'  # Lighter gray background
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['axes.labelsize'] = 13  # Axis label size
mpl.rcParams['axes.titlesize'] = 15  # Subplot title size
mpl.rcParams['xtick.labelsize'] = 12  # X-tick label size
mpl.rcParams['ytick.labelsize'] = 12  # Y-tick label size
mpl.rcParams['legend.fontsize'] = 13  # Legend font size

# Define consistent colors - matching mutation benchmarking code
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

# Map model names to checkpoint directories for entropy calculation
MODEL_TO_CHECKPOINT_DIR = {
    'Dino_ViT-B': '/data1/vanderbc/nandas1/TCGA_Dino_ViT-B_run2',
    'TMEDinov1_ViT-B_version2': '/data1/vanderbc/nandas1/TCGA_TMEDinov1_version2_ViT-B',
    'TMEDinov1_ViT-B_version3': '/data1/vanderbc/nandas1/TCGA_TMEDinov1_version3_ViT-B',
    'Dinov2_ViT-B': '/data1/vanderbc/nandas1/TCGA_Dinov2_ViT-B_run2',
    'TMEDinov2_ViT-B_version2': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version2_ViT-B',
    'TMEDinov2_ViT-B_version3': '/data1/vanderbc/nandas1/TCGA_TMEDinov2_version3_ViT-B'
}

def format_iteration_axis(ax):
    """Format x-axis to show iterations in k units"""
    def formatter(x, pos):
        if x == 0:
            return '0'
        elif x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return str(int(x))
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(formatter))

def load_all_metrics(base_dir):
    """Load all metrics from iteration directories"""
    metrics_data = []
    missing_metrics = []
    empty_metrics = []
    
    iteration_dirs = glob.glob(os.path.join(base_dir, "iteration_*"))
    
    for iter_dir in iteration_dirs:
        iter_name = os.path.basename(iter_dir)
        
        # Handle 'iteration_final' as 300000
        if 'final' in iter_name:
            iteration = 300000
        else:
            try:
                iteration = int(iter_name.replace('iteration_', ''))
            except ValueError:
                print(f"⚠️  Warning: Could not parse iteration from '{iter_name}'")
                continue
            
        metrics_file = os.path.join(iter_dir, "all_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                try:
                    metrics = json.load(f)
                    if metrics:  # Check if metrics dict is not empty
                        metrics_data.append({
                            'iteration': iteration,
                            'metrics': metrics
                        })
                    else:
                        empty_metrics.append((iteration, metrics_file))
                except json.JSONDecodeError:
                    print(f"⚠️  Warning: Invalid JSON in {metrics_file}")
                    empty_metrics.append((iteration, metrics_file))
        else:
            missing_metrics.append((iteration, metrics_file))
    
    # Report missing or empty metrics
    if missing_metrics:
        print(f"⚠️  Missing metrics files for iterations: {[i for i, _ in missing_metrics]}")
    if empty_metrics:
        print(f"⚠️  Empty/invalid metrics for iterations: {[i for i, _ in empty_metrics]}")
    
    return metrics_data

def extract_task_agnostic_data(metrics_data, metric_name):
    """Extract task-agnostic metrics from all_metrics.json"""
    data = []
    
    for entry in metrics_data:
        iteration = entry['iteration']
        metrics = entry['metrics']
        
        if 'task_agnostic' in metrics and metric_name in metrics['task_agnostic']:
            ta_data = metrics['task_agnostic'][metric_name]
            
            row = {
                'checkpoint_iteration': iteration,
                'mean': ta_data.get('mean'),
                'ci_lower': ta_data.get('ci_lower'),
                'ci_upper': ta_data.get('ci_upper')
            }
            
            if row['mean'] is not None:
                data.append(row)
    
    return pd.DataFrame(data)

# ============== ENTROPY CALCULATION FUNCTIONS ==============

def get_checkpoints_at_intervals(checkpoint_dir, interval=10000, max_iter=300000):
    """Get checkpoint paths at specified intervals."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    
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
    
    return checkpoints

def compute_patch_token_entropy(patch_embeddings):
    """
    Compute entropy for the Gram matrix of patch tokens.
    
    Args:
        patch_embeddings: (B, N, D) tensor of patch features
    
    Returns:
        entropy: Entropy value
    """
    B, N, D = patch_embeddings.shape
    
    # Normalize embeddings for cosine similarity
    patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
    
    # Compute Gram matrix (cosine similarity)
    gram = torch.bmm(patch_embeddings, patch_embeddings.transpose(1, 2))  # (B, N, N)
    
    # Compute SVD to get singular values
    try:
        U, S, V = torch.svd(gram)
    except:
        # If SVD fails, try with CPU and double precision
        gram_cpu = gram.cpu().double()
        U, S, V = torch.svd(gram_cpu)
        S = S.float().cuda() if torch.cuda.is_available() else S.float()
    
    # Compute entropy
    # Normalize singular values to form probability distribution
    S_normalized = S / (S.sum(dim=1, keepdim=True) + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum(dim=1)
    
    return entropy.mean().item()

def extract_patch_embeddings(model, images):
    """Extract patch embeddings from the model."""
    model.eval()
    
    # Model configuration
    num_registers = getattr(model, 'numregisters', 4)
    has_cls = model.cls_token is not None
    
    with torch.no_grad():
        # Prepare tokens
        x = model.prepare_tokens(images)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        
        # Process through all transformer blocks
        for block in model.blocks:
            x = block(x)
        
        # Apply final norm
        x = model.norm(x)
        
        # Extract patch tokens only (skip CLS and registers)
        if has_cls:
            patch_embeddings = x[:, 1 + num_registers:, :]
        else:
            patch_embeddings = x[:, num_registers:, :]
    
    return patch_embeddings

def calculate_entropy_for_model(model_name, checkpoint_dir, interval=10000, max_iter=300000,
                               batch_size=8, n_batches=5):
    """Calculate entropy across training iterations for a specific model."""
    from torch.utils.data import DataLoader
    
    print(f"  Calculating entropy for {model_name}...")
    
    # Import required modules
    try:
        from DataAug_TME import MemoryEfficientShardedPathologyDataset
        from utils import load_dino_backbone
    except ImportError:
        print(f"  ⚠️  Warning: Could not import required modules for {model_name}")
        return None
    
    # Create dataset
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
    
    # Create DataLoader
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
    
    for batch_data in dataloader:
        if batch_count >= n_batches:
            break
        
        # Handle different possible return formats from the dataset
        if isinstance(batch_data, (list, tuple)):
            test_images = batch_data[0]
        else:
            test_images = batch_data
        
        # Ensure we have the right shape
        if len(test_images.shape) == 5:  # If shape is (B, n_crops, C, H, W)
            test_images = test_images[:, 0]  # Take first crop
        
        all_test_images.append(test_images)
        batch_count += 1
    
    # Get checkpoints
    checkpoints = get_checkpoints_at_intervals(checkpoint_dir, interval, max_iter)
    
    # Store results
    results = {
        'iterations': [],
        'entropy': []
    }
    
    # Analyze each checkpoint
    for idx, (iteration, ckpt_path) in enumerate(checkpoints):
        try:
            # Load model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_dino_backbone(ckpt_path, device)
            
            # Compute metrics for each batch and average
            batch_entropies = []
            
            for test_images in all_test_images:
                test_images = test_images.to(device)
                
                # Extract patch embeddings
                patch_embeddings = extract_patch_embeddings(model, test_images)
                
                # Compute entropy
                entropy = compute_patch_token_entropy(patch_embeddings)
                batch_entropies.append(entropy)
            
            # Average across batches and store
            results['iterations'].append(iteration)
            results['entropy'].append(np.mean(batch_entropies))
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    Error at iteration {iteration}: {e}")
            continue
    
    return results

def load_or_calculate_entropy(model_name, RUN_FRESH=False):
    """Load precomputed entropy or calculate fresh."""
    # Define pickle file path for this model
    pickle_path = f"gram_entropy_{model_name}.pkl"
    
    if not RUN_FRESH and os.path.exists(pickle_path):
        print(f"  Loading precomputed entropy for {model_name} from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            results = pickle.load(f)
        
        # If results contain effective_rank, convert to entropy
        if 'effective_rank' in results and 'entropy' not in results:
            results['entropy'] = [np.log(er) for er in results['effective_rank']]
    else:
        print(f"  Calculating fresh entropy for {model_name}")
        checkpoint_dir = MODEL_TO_CHECKPOINT_DIR.get(model_name)
        
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            print(f"    Warning: Checkpoint directory not found for {model_name}")
            return None
        
        results = calculate_entropy_for_model(
            model_name, 
            checkpoint_dir,
            interval=10000,
            max_iter=300000,
            batch_size=16,
            n_batches=100
        )
        
        # Save results
        if results:
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"    Saved entropy results to {pickle_path}")
    
    # Convert to DataFrame if results exist
    if results and 'iterations' in results and 'entropy' in results:
        df = pd.DataFrame({
            'checkpoint_iteration': results['iterations'],
            'mean': results['entropy'],
            'ci_lower': None,  # No CI for entropy
            'ci_upper': None
        })
        return df
    
    return None

# ============== PLOTTING FUNCTIONS ==============

def plot_on_axis(ax, data_dict, dataset_name, metric_type, show_legend=False):
    """Plot data on a specific axis for three models"""
    
    lines_for_legend = {}
    
    for model_name, df in data_dict.items():
        if df is None or df.empty:
            continue
        
        # Get augmentation type and color
        aug_type = MODEL_TO_AUGMENTATION.get(model_name, 'standard')
        color = AUGMENTATION_COLORS[aug_type]
        
        # Set label based on augmentation type
        if aug_type == 'standard':
            label = 'Standard'
        elif aug_type == 'mixed':
            label = 'Mixed'
        else:  # masked
            label = 'Masked'
        
        # Plot metrics
        df_sorted = df.sort_values('checkpoint_iteration')
        x = df_sorted['checkpoint_iteration']
        y = df_sorted['mean']
        
        # Handle confidence intervals
        if 'ci_lower' in df_sorted.columns and 'ci_upper' in df_sorted.columns:
            yerr_lower = y - df_sorted['ci_lower'].fillna(y)
            yerr_upper = df_sorted['ci_upper'].fillna(y) - y
        else:
            yerr_lower = yerr_upper = 0
        
        line = ax.errorbar(x, y,
                   yerr=[yerr_lower, yerr_upper],
                   color=color,
                   marker='o',
                   markersize=4,  # Reduced for paper
                   linewidth=1.5,  # Reduced for paper
                   capsize=2,  # Reduced for paper
                   capthick=1,
                   alpha=0.9,
                   label=label if show_legend else None)
        
        if aug_type not in lines_for_legend:
            lines_for_legend[aug_type] = (line, label)
    
    # Format x-axis
    format_iteration_axis(ax)
    
    # Darken borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)  # Slightly reduced for paper
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return lines_for_legend

def create_task_agnostic_plot(comparisons, base_path, output_dir, RUN_FRESH=False):
    """Create task-agnostic metrics plot with 2 rows × 4 columns"""
    
    print("\n" + "="*60)
    print("Creating Task-Agnostic Metrics Plot with Patch Token Entropy")
    print(f"RUN_FRESH mode: {RUN_FRESH}")
    print("="*60)
    
    # Define metrics - updated with (CLS) suffix
    metrics = [
        ('rankme', 'RankMe (CLS)'),
        ('alphareq', r'$\alpha$-ReQ (CLS)'),
        ('clid', 'CLID (CLS)'),
        ('entropy', 'Entropy (Patch)')  # New column
    ]
    
    # Create figure with subplots - adjusted for ICLR paper
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))  # Wider for 4 columns, compact height for paper
    
    # Store lines for legend
    legend_lines = {}
    
    # Process each SSL method (row)
    for row_idx, comp in enumerate(comparisons):
        print(f"\nProcessing row {row_idx+1}: {comp['ssl_method']}")
        
        # Load metrics once per SSL method
        model1_metrics = load_all_metrics(comp['model1_path']) if os.path.exists(comp['model1_path']) else []
        model2_metrics = load_all_metrics(comp['model2_path']) if os.path.exists(comp['model2_path']) else []
        model3_metrics = load_all_metrics(comp['model3_path']) if os.path.exists(comp['model3_path']) else []
        
        # Process each metric (column)
        for col_idx, (metric_key, metric_name) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data_dict = {}
            
            if metric_key == 'entropy':
                # Handle entropy column
                print(f"  Processing Patch Token Entropy for {comp['ssl_method']}")
                
                # Load or calculate entropy for all three models
                df1 = load_or_calculate_entropy(comp['model1_name'], RUN_FRESH)
                if df1 is not None and not df1.empty:
                    data_dict[comp['model1_name']] = df1
                
                df2 = load_or_calculate_entropy(comp['model2_name'], RUN_FRESH)
                if df2 is not None and not df2.empty:
                    data_dict[comp['model2_name']] = df2
                
                df3 = load_or_calculate_entropy(comp['model3_name'], RUN_FRESH)
                if df3 is not None and not df3.empty:
                    data_dict[comp['model3_name']] = df3
            else:
                # Extract task-agnostic data for CLS-based metrics
                if model1_metrics:
                    df1 = extract_task_agnostic_data(model1_metrics, metric_key)
                    if not df1.empty:
                        data_dict[comp['model1_name']] = df1
                
                if model2_metrics:
                    df2 = extract_task_agnostic_data(model2_metrics, metric_key)
                    if not df2.empty:
                        data_dict[comp['model2_name']] = df2
                
                if model3_metrics:
                    df3 = extract_task_agnostic_data(model3_metrics, metric_key)
                    if not df3.empty:
                        data_dict[comp['model3_name']] = df3
            
            # Plot the data
            lines = plot_on_axis(ax, data_dict, metric_name, 'task_agnostic', show_legend=False)
            legend_lines.update(lines)
            
            # Set column titles (metric names) - only on top row
            if row_idx == 0:
                ax.set_title(f'{metric_name}', fontsize=10, fontweight='bold', pad=8)
            
            # Add SSL method textbox in top-left corner of first column
            if col_idx == 0:
                ax.text(0.02, 0.98, comp['ssl_method'], transform=ax.transAxes,
                       fontsize=9, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#333333', linewidth=1.2, alpha=0.95))
            
            # Set y-label for first column
            if col_idx == 0:
                ax.set_ylabel('Score', fontsize=9, fontweight='bold')
            
            # Set x-label for bottom row
            if row_idx == 1:
                ax.set_xlabel('Iteration', fontsize=9, fontweight='bold')
    
    # After all plotting is done, adjust y-axis for each column to fit data tightly
    for col_idx in range(4):
        # Collect the actual y-limits matplotlib calculated for both rows
        y_mins = []
        y_maxs = []
        
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            if len(ax.lines) > 0:  # Only if there's data
                # Get the current y-limits that matplotlib auto-calculated
                y_min, y_max = ax.get_ylim()
                y_mins.append(y_min)
                y_maxs.append(y_max)
        
        # If we have data, set consistent tight limits for this column
        if y_mins and y_maxs:
            overall_min = min(y_mins)
            overall_max = max(y_maxs)
            y_range = overall_max - overall_min
            
            # Add just 3% padding around the data
            padding = 0.03 * y_range if y_range > 0 else 0.5
            
            # Apply the same limits to both rows
            for row_idx in range(2):
                axes[row_idx, col_idx].set_ylim(overall_min - padding, overall_max + padding)
    
    # Add legend with three augmentation types
    if legend_lines:
        handles = []
        labels = []
        for aug_type in ['standard', 'mixed', 'masked']:
            if aug_type in legend_lines:
                handles.append(legend_lines[aug_type][0])
                labels.append(legend_lines[aug_type][1])
        
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                  ncol=3, fontsize=9, frameon=True, fancybox=False,
                  edgecolor='#333333', framealpha=0.95, borderpad=0.3,
                  columnspacing=1.5)  # Adjust spacing for clarity
    
    # Adjust layout with proper spacing for ICLR paper
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], h_pad=2.0, w_pad=2.5)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "task_agnostic_metrics_with_entropy.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def main():
    # Configuration
    RUN_FRESH = False  # Set to True to recalculate entropy, False to load from pickle files
    
    # Base path for benchmark results
    base_path = "/data1/vanderbc/nandas1/PostProc/benchmark_results"
    output_dir = "patch_benchmark_plots"
    
    print("=" * 60)
    print("Creating Task-Agnostic Metrics Visualization with Patch Token Entropy")
    print("Optimized for ICLR A4 paper format")
    print(f"RUN_FRESH mode: {RUN_FRESH}")
    print("=" * 60)
    
    # Define the 2 SSL methods with 3 models each
    comparisons = [
        {
            'model1_path': os.path.join(base_path, "TCGA_Dino_ViT-B_run2"),
            'model2_path': os.path.join(base_path, "TCGA_TMEDinov1_version2_ViT-B"),
            'model3_path': os.path.join(base_path, "TCGA_TMEDinov1_version3_ViT-B"),
            'model1_name': 'Dino_ViT-B_run2',
            'model2_name': 'TMEDinov1_ViT-B_version2',
            'model3_name': 'TMEDinov1_ViT-B_version3',
            'ssl_method': 'Dinov1'
        },
        {
            'model1_path': os.path.join(base_path, "TCGA_Dinov2_ViT-B_run2"),
            'model2_path': os.path.join(base_path, "TCGA_TMEDinov2_version2_ViT-B"),
            'model3_path': os.path.join(base_path, "TCGA_TMEDinov2_version3_ViT-B"),
            'model1_name': 'Dinov2_ViT-B_run2',
            'model2_name': 'TMEDinov2_ViT-B_version2',
            'model3_name': 'TMEDinov2_ViT-B_version3',
            'ssl_method': 'Dinov2'
        }
    ]
    
    # Create the task-agnostic metrics plot with entropy
    create_task_agnostic_plot(comparisons, base_path, output_dir, RUN_FRESH)
    
    print("\n" + "=" * 60)
    print("✓ Task-Agnostic Metrics visualization complete!")
    print(f"✓ Output file: {output_dir}/task_agnostic_metrics_with_entropy.png (2×4 grid)")
    print("✓ Columns: RankMe (CLS) | α-ReQ (CLS) | CLID (CLS) | Entropy (Patch)")
    print("=" * 60)

if __name__ == "__main__":
    main()
