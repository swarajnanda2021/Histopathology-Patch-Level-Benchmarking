#!/usr/bin/env python3
"""
Script to create DINOv2-only plots:
1. Slide mutation prediction for LUAD-EGFR and BLCA-FGFR3
2. Patch token entropy plot
Both plots show only DINOv2 variants (Standard, Mixed, Masked, RandMask).
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from pathlib import Path
import re
from scipy import stats
import pickle

# Set ggplot style and professional font
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.facecolor'] = '#F5F5F5'
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

# Define colors for different model types
MODEL_COLORS = {
    'standard': '#1E88E5',      # Professional blue
    'masked_only': '#DC3545',   # Professional red  
    'mixed': '#FFA726',         # Professional orange
    'random_masking': '#43A047' # Professional green
}

# Map model names to their types for slide mutations
SLIDE_MODEL_CONFIG = {
    'TCGA_Dinov2_ViT-B_run2': ('standard', 'Standard'),
    'TCGA_TMEDinov2_version2_ViT-B': ('mixed', 'Mixed'),
    'TCGA_TMEDinov2_version3_ViT-B': ('masked_only', 'Masked'),
    'TCGA_TMEDinov2_version4_random_masking_ViT-B': ('random_masking', 'RandMask')
}

# Map model names for entropy plot
ENTROPY_MODEL_NAMES = {
    'Dinov2_ViT-B_run2': 'standard',
    'TMEDinov2_ViT-B_version2': 'mixed',
    'TMEDinov2_ViT-B_version3': 'masked_only',
    'TMEDinov2_ViT-B_version4_random_masking': 'random_masking'
}

# ============== SLIDE MUTATION FUNCTIONS ==============

def extract_iteration_from_path(file_path):
    """Extract iteration number from file path."""
    match = re.search(r'iter_(\d+)', file_path)
    if match:
        return int(match.group(1))
    return None

def extract_split_from_path(file_path):
    """Extract split number from file path."""
    match = re.search(r'split_(\d+)_set', file_path)
    if match:
        return int(match.group(1))
    return None

def get_highest_epoch_auc(csv_file):
    """Extract test_auc value from the highest epoch."""
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['test_auc'])
        df = df[df['test_auc'] != '']
        
        if df.empty:
            return None
        
        highest_epoch = df['epoch'].max()
        highest_epoch_data = df[df['epoch'] == highest_epoch]
        
        if highest_epoch_data.empty:
            return None
        
        return highest_epoch_data['test_auc'].iloc[0]
    
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def collect_slide_mutation_data(base_dir, model_names, mutation_name):
    """Collect data for specific models."""
    all_data = []
    
    for model_name in model_names:
        pattern = os.path.join(base_dir, 
                               f"{model_name}_checkpoint_iter_*",
                               f"{mutation_name}_gma",
                               f"{mutation_name}_split_*_set_{model_name}*_convergence_{mutation_name}_gma.csv")
        
        csv_files = glob.glob(pattern)
        print(f"  {model_name}: Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            iteration = extract_iteration_from_path(csv_file)
            split = extract_split_from_path(csv_file)
            test_auc = get_highest_epoch_auc(csv_file)
            
            if iteration is not None and split is not None and test_auc is not None:
                all_data.append({
                    'model': model_name,
                    'iteration': iteration,
                    'split': split,
                    'test_auc': test_auc
                })
    
    return pd.DataFrame(all_data)

def calculate_shared_ylim(df, models):
    """Calculate shared y-axis limits adjusted to multiples of 0.05."""
    all_aucs = []
    for model in models:
        model_data = df[df['model'] == model]['test_auc']
        if not model_data.empty:
            all_aucs.extend(model_data.values)
    
    if not all_aucs:
        return 0, 1
    
    min_auc = min(all_aucs)
    max_auc = max(all_aucs)
    padding = (max_auc - min_auc) * 0.05
    
    min_with_padding = max(0, min_auc - padding)
    max_with_padding = min(1, max_auc + padding)
    
    ylim_min = np.floor(min_with_padding / 0.05) * 0.05
    ylim_max = np.ceil(max_with_padding / 0.05) * 0.05
    
    ylim_min = max(0, ylim_min)
    ylim_max = min(1, ylim_max)
    
    return ylim_min, ylim_max

def generate_nice_yticks(ylim, step=0.05):
    """Generate y-axis tick positions at strict multiples of step."""
    ymin, ymax = ylim
    n_ticks = int(round((ymax - ymin) / step)) + 1
    ticks = [ymin + i * step for i in range(n_ticks)]
    ticks = [round(t, 3) for t in ticks]
    return ticks

def plot_slide_mutation_group(df, models, ax, ylim, title_text="", show_ylabel=True, show_yticks=True):
    """Plot a group of models on a single axis."""
    ax.clear()
    ax.set_facecolor('#F5F5F5')
    
    all_iterations = sorted(df['iteration'].unique())
    
    group_width = 0.8
    n_models = len([m for m in models if not df[df['model'] == m].empty])
    if n_models == 0:
        return
    
    point_spacing = group_width / n_models
    
    ax.set_ylim(ylim)
    yticks = generate_nice_yticks(ylim)
    ax.set_yticks(yticks)
    
    # Add alternating white/gray bands
    for iter_idx in range(len(all_iterations)):
        band_half_width = 0.45
        left_edge = iter_idx - band_half_width
        right_edge = iter_idx + band_half_width
        
        ax.axvspan(left_edge, right_edge, facecolor='white', edgecolor='none', zorder=0)
        ax.axvline(x=left_edge, color='#333333', linewidth=0.5, zorder=2)
        ax.axvline(x=right_edge, color='#333333', linewidth=0.5, zorder=2)
        
        if iter_idx < len(all_iterations) - 1:
            ax.axvspan(right_edge, iter_idx + 1 - band_half_width, 
                      facecolor='#F5F5F5', edgecolor='none', zorder=0)
    
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=1.2, 
            color='#888888', alpha=0.7, zorder=1)
    ax.grid(False, which='both', axis='x')
    
    # Plot each model's data
    for iter_idx, iteration in enumerate(all_iterations):
        iter_data = df[df['iteration'] == iteration]
        model_offset = -(n_models - 1) * point_spacing / 2
        
        for model_idx, model_name in enumerate(models):
            model_data = iter_data[iter_data['model'] == model_name]['test_auc'].values
            
            if len(model_data) > 0:
                model_type, display_name = SLIDE_MODEL_CONFIG[model_name]
                color = MODEL_COLORS[model_type]
                
                x_pos = iter_idx + model_offset
                
                mean_val = np.mean(model_data)
                sem = stats.sem(model_data)
                ci_95 = sem * stats.t.ppf(0.975, len(model_data) - 1)
                
                ax.scatter(x_pos, mean_val, color=color, s=150, zorder=12, 
                          edgecolors='#333333', linewidths=2, marker='D')
                
                ax.errorbar(x_pos, mean_val, yerr=ci_95, 
                           color='#333333', linewidth=2, capsize=5, 
                           capthick=2, zorder=11, alpha=0.8)
                
                model_offset += point_spacing
    
    ax.set_xticks(range(len(all_iterations)))
    ax.set_xticklabels([f'{int(it/1000)}k' if it >= 1000 else str(int(it)) 
                        for it in all_iterations], fontsize=22)
    
    ax.set_title(title_text, fontsize=26, fontweight='bold', pad=10)
    
    if show_yticks:
        ax.set_yticklabels([f'{t:.2f}' for t in yticks], fontsize=22)
    else:
        ax.set_yticklabels([])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
        spine.set_zorder(20)
    
    ax.set_xlabel('Iteration', fontsize=24, fontweight='bold')
    
    if show_ylabel:
        ax.set_ylabel('AUC', fontsize=24, fontweight='bold')
    else:
        ax.set_ylabel('')

# ============== ENTROPY FUNCTIONS ==============

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

def load_entropy_from_pickle(model_name):
    """Load precomputed entropy from pickle file."""
    pickle_path = f"gram_entropy_{model_name}.pkl"
    
    if os.path.exists(pickle_path):
        print(f"  Loading entropy for {model_name} from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            results = pickle.load(f)
        
        # Convert to DataFrame
        if results and 'iterations' in results and 'entropy' in results:
            df = pd.DataFrame({
                'checkpoint_iteration': results['iterations'],
                'mean': results['entropy']
            })
            return df
        else:
            print(f"  Warning: Invalid data structure in {pickle_path}")
    else:
        print(f"  Warning: Pickle file not found: {pickle_path}")
    
    return None

def plot_entropy(ax, data_dict):
    """Plot entropy data for DINOv2 models."""
    
    lines_for_legend = {}
    
    for model_name, df in data_dict.items():
        if df is None or df.empty:
            continue
        
        # Get model type and color
        model_type = ENTROPY_MODEL_NAMES.get(model_name, 'standard')
        color = MODEL_COLORS[model_type]
        
        # Set label
        if model_type == 'standard':
            label = 'Standard'
        elif model_type == 'mixed':
            label = 'Mixed'
        elif model_type == 'masked_only':
            label = 'Masked'
        else:  # random_masking
            label = 'RandMask'
        
        # Plot entropy
        df_sorted = df.sort_values('checkpoint_iteration')
        x = df_sorted['checkpoint_iteration']
        y = df_sorted['mean']
        
        line, = ax.plot(x, y,
                   color=color,
                   marker='o',
                   markersize=10,  # Increased
                   linewidth=3,    # Increased
                   alpha=0.9,
                   label=label
                   )  

        lines_for_legend[model_type] = (line, label)
    
    # Format axes
    format_iteration_axis(ax)
    
    # Style
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    ax.set_xlabel('Iteration', fontsize=24, fontweight='bold')
    ax.set_ylabel('Entropy (Patch)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=22)

    
    # Adjust y-limits with padding
    if len(ax.lines) > 0:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.01
        ax.set_ylim(y_min - padding, y_max + padding)
    
    return lines_for_legend

# ============== MAIN FUNCTION ==============

def main():
    print("="*80)
    print("Creating DINOv2-Only Plots")
    print("="*80)
    
    # Output directory
    output_dir = "dinov2_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== PLOT 1: SLIDE MUTATIONS ==========
    print("\n" + "="*60)
    print("Creating Slide Mutation Plot (DINOv2 only)")
    print("="*60)
    
    # Base directories for mutations
    base_dirs = [
        ("/data1/vanderbc/foundation_model_training_images/IMPACT/LUAD/checkpoints/EGFR", "LUAD", "EGFR"),
        ("/data1/vanderbc/foundation_model_training_images/IMPACT/BLCA/checkpoints/FGFR3", "BLCA", "FGFR3")
    ]
    
    # DINOv2 models only (now with 4 models)
    dinov2_models = [
        'TCGA_Dinov2_ViT-B_run2',                             # Standard
        'TCGA_TMEDinov2_version3_ViT-B',                      # Masked  
        'TCGA_TMEDinov2_version2_ViT-B',                      # Mixed
        'TCGA_TMEDinov2_version4_random_masking_ViT-B'        # RandMask
    ]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Process each mutation
    for idx, (base_dir, cancer_type, mutation_name) in enumerate(base_dirs):
        plot_title = f"{cancer_type}-{mutation_name}"
        print(f"\nProcessing {plot_title}...")
        
        # Collect data
        df = collect_slide_mutation_data(base_dir, dinov2_models, mutation_name)

        df = df[df['iteration'] != 98000]
        
        if df.empty:
            print(f"  No valid data found for {plot_title}!")
            continue
        
        # Calculate y-limits
        ylim = calculate_shared_ylim(df, dinov2_models)
        print(f"  Y-axis limits: [{ylim[0]:.2f}, {ylim[1]:.2f}]")
        
        # Plot
        ax = axes[idx]
        plot_slide_mutation_group(df, dinov2_models, ax, ylim, 
                                 title_text=plot_title,
                                 show_ylabel=(idx == 0),
                                 show_yticks=True)
    
    # Add legend (now with 4 items)
    legend_elements = [
        Patch(facecolor=MODEL_COLORS['standard'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Standard'),
        Patch(facecolor=MODEL_COLORS['masked_only'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Masked'),
        Patch(facecolor=MODEL_COLORS['mixed'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Mixed'),
        Patch(facecolor=MODEL_COLORS['random_masking'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='RandMask')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fontsize=20, frameon=True, fancybox=False, 
              edgecolor='#333333', framealpha=0.95)
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.92])
    
    # Save
    output_path = os.path.join(output_dir, "slide_mutations_dinov2.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    # ========== PLOT 2: PATCH ENTROPY ==========
    print("\n" + "="*60)
    print("Creating Patch Token Entropy Plot (DINOv2 only)")
    print("="*60)
    
    # Create single figure for entropy
    fig, ax = plt.subplots(1, 1, figsize=(7, 6)) 
    
    # Load entropy data for DINOv2 models
    entropy_data = {}
    for model_name in ENTROPY_MODEL_NAMES.keys():
        df = load_entropy_from_pickle(model_name)
        if df is not None:
            # Exclude iteration 98000
            df = df[df['checkpoint_iteration'] != 98000]
            entropy_data[model_name] = df
    
    # Plot entropy
    lines_for_legend = plot_entropy(ax, entropy_data)
    
    # Add legend with ordered items (now including random_masking)
    if lines_for_legend:
        handles = []
        labels = []
        for model_type in ['standard', 'masked_only', 'mixed', 'random_masking']:
            if model_type in lines_for_legend:
                handles.append(lines_for_legend[model_type][0])
                labels.append(lines_for_legend[model_type][1])
        
        ax.legend(handles, labels, loc='best', fontsize=20, frameon=True, 
                 fancybox=False, edgecolor='#333333', framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "patch_entropy_dinov2.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "="*80)
    print("✓ All DINOv2 plots complete!")
    print(f"✓ Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
