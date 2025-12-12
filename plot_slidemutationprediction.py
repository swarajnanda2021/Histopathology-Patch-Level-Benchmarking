#!/usr/bin/env python3
"""
Script to create professional mutation performance plots with ggplot styling.
Creates a single row of plots: alternating between Dinov1 and Dinov2 for each mutation.
Layout: [LUAD-EGFR-Dinov1, LUAD-EGFR-Dinov2] [gap] [BLCA-FGFR3-Dinov1, BLCA-FGFR3-Dinov2]
Plots are fused in pairs with standardized y-axes and proper spacing between pairs.
Enhanced with alternating white/gray bands to clearly group iterations.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from pathlib import Path
import re
from scipy import stats

# Set ggplot style and professional font (matching plotting.py style)
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 24  # Base font size
mpl.rcParams['axes.linewidth'] = 2.5
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.facecolor'] = '#F5F5F5'  # Lighter gray background
mpl.rcParams['grid.alpha'] = 0.7  # Increased for visibility
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 1.2
mpl.rcParams['grid.color'] = '#888888'  # Darker gray for better visibility
mpl.rcParams['xtick.labelsize'] = 22  # Tick label size reduced by 2
mpl.rcParams['ytick.labelsize'] = 22  # Tick label size reduced by 2

# Define colors for different model types
MODEL_COLORS = {
    'standard': '#1E88E5',      # Professional blue
    'masked_only': '#DC3545',   # Professional red  
    'mixed': '#FFA726'           # Professional orange
}

# Map model names to their types and display names
MODEL_CONFIG = {
    'TCGA_Dinov1_ViT-B_run2': ('standard', 'Standard'),
    'TCGA_Dinov2_ViT-B_run2': ('standard', 'Standard'),
    'TCGA_TMEDinov1_version2_ViT-B': ('mixed', 'Mixed'),
    'TCGA_TMEDinov2_version2_ViT-B': ('mixed', 'Mixed'),
    'TCGA_TMEDinov1_version3_ViT-B': ('masked_only', 'Masked Only'),
    'TCGA_TMEDinov2_version3_ViT-B': ('masked_only', 'Masked Only')
}

def extract_cancer_and_mutation_from_path(base_dir_path):
    """Extract cancer type and mutation from base directory path using regex."""
    # Pattern to match: /path/to/IMPACT/{CANCER_TYPE}/checkpoints/{MUTATION}/
    pattern = r'/IMPACT/([^/]+)/checkpoints/([^/]+)'
    match = re.search(pattern, base_dir_path)
    if match:
        cancer_type = match.group(1)  # e.g., LUAD, BLCA
        mutation = match.group(2)      # e.g., EGFR, FGFR3
        return cancer_type, mutation
    return None, None

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
    """Extract test_auc value from the highest epoch (typically 200)."""
    try:
        df = pd.read_csv(csv_file)
        
        # Filter out rows where test_auc is empty, NA, or NaN
        df = df.dropna(subset=['test_auc'])
        df = df[df['test_auc'] != '']
        
        if df.empty:
            return None
        
        # Get the highest epoch (should be 200)
        highest_epoch = df['epoch'].max()
        highest_epoch_data = df[df['epoch'] == highest_epoch]
        
        if highest_epoch_data.empty:
            return None
        
        # Return the test_auc value from the highest epoch
        return highest_epoch_data['test_auc'].iloc[0]
    
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def collect_data_for_models(base_dir, model_names, mutation_name):
    """Collect data for specific models."""
    all_data = []
    
    for model_name in model_names:
        # Pattern to match CSV files for this model
        # Adjust pattern to use mutation_name for the subdirectory
        pattern = os.path.join(base_dir, 
                               f"{model_name}_checkpoint_iter_*",
                               f"{mutation_name}_gma",
                               f"{mutation_name}_split_*_set_{model_name}*_convergence_{mutation_name}_gma.csv")
        
        csv_files = glob.glob(pattern)
        print(f"\n{model_name}: Found {len(csv_files)} CSV files")
        
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

def calculate_shared_ylim(df, models_list):
    """Calculate shared y-axis limits for a group of models, adjusted to multiples of 0.05."""
    # Get all AUC values for the specified models
    all_aucs = []
    for models in models_list:
        for model in models:
            model_data = df[df['model'] == model]['test_auc']
            if not model_data.empty:
                all_aucs.extend(model_data.values)
    
    if not all_aucs:
        return 0, 1
    
    # Calculate raw limits with some padding
    min_auc = min(all_aucs)
    max_auc = max(all_aucs)
    padding = (max_auc - min_auc) * 0.05  # 5% padding
    
    # Add padding
    min_with_padding = max(0, min_auc - padding)
    max_with_padding = min(1, max_auc + padding)
    
    # Round DOWN to nearest 0.05 for minimum
    ylim_min = np.floor(min_with_padding / 0.05) * 0.05
    
    # Round UP to nearest 0.05 for maximum
    ylim_max = np.ceil(max_with_padding / 0.05) * 0.05
    
    # Ensure we don't go outside [0, 1]
    ylim_min = max(0, ylim_min)
    ylim_max = min(1, ylim_max)
    
    return ylim_min, ylim_max

def generate_nice_yticks(ylim, step=0.05):
    """Generate y-axis tick positions at strict multiples of step."""
    ymin, ymax = ylim
    
    # Generate ticks from ymin to ymax at step intervals
    # Both ymin and ymax should already be multiples of step
    n_ticks = int(round((ymax - ymin) / step)) + 1
    ticks = [ymin + i * step for i in range(n_ticks)]
    
    # Round to avoid floating point errors
    ticks = [round(t, 3) for t in ticks]
    
    return ticks

def plot_model_group(df, models, ax, ylim, add_title=False, title_text="", 
                     show_ylabel=True, show_yticks=True):
    """Plot a group of models on a single axis with strip plots showing all points, means, and CIs."""
    
    # Clear any existing content
    ax.clear()
    
    # Set face color first (will be overridden by bands)
    ax.set_facecolor('#F5F5F5')
    
    # Get all unique iterations across all models
    all_iterations = sorted(df['iteration'].unique())
    
    # Width between model groups and spacing
    group_width = 0.8
    n_models = len([m for m in models if not df[df['model'] == m].empty])
    if n_models == 0:
        return
    
    point_spacing = group_width / n_models
    
    # Set y-axis limits BEFORE drawing anything
    ax.set_ylim(ylim)
    
    # Generate and set nice y-ticks
    yticks = generate_nice_yticks(ylim)
    ax.set_yticks(yticks)
    
    # Add alternating white/gray bands BEFORE plots
    for iter_idx in range(len(all_iterations)):
        # Calculate band boundaries
        # White band is centered on the iteration position with width to contain all points
        band_half_width = 0.45  # Slightly less than 0.5 to leave small gaps
        left_edge = iter_idx - band_half_width
        right_edge = iter_idx + band_half_width
        
        # White band for data area
        ax.axvspan(left_edge, right_edge, facecolor='white', edgecolor='none', zorder=0)
        
        # Add thin black borders on the edges of white bands
        ax.axvline(x=left_edge, color='#333333', linewidth=0.5, zorder=2)
        ax.axvline(x=right_edge, color='#333333', linewidth=0.5, zorder=2)
        
        # Gray bands will be the remaining space (background color handles this)
        # If not the last iteration, add gray band between this and next iteration
        if iter_idx < len(all_iterations) - 1:
            ax.axvspan(right_edge, iter_idx + 1 - band_half_width, 
                      facecolor='#F5F5F5', edgecolor='none', zorder=0)
    
    # Add horizontal gridlines ONLY (no vertical)
    ax.grid(True, which='both', axis='y',  # Only y-axis gridlines
            linestyle='--', linewidth=1.2, 
            color='#888888', alpha=0.7, zorder=1)
    ax.grid(False, which='both', axis='x')  # Explicitly turn off x-axis gridlines
    
    # Plot each model's data
    for iter_idx, iteration in enumerate(all_iterations):
        iter_data = df[df['iteration'] == iteration]
        
        model_offset = -(n_models - 1) * point_spacing / 2
        
        for model_idx, model_name in enumerate(models):
            model_data = iter_data[iter_data['model'] == model_name]['test_auc'].values
            
            if len(model_data) > 0:
                # Get model type and color
                model_type, display_name = MODEL_CONFIG[model_name]
                color = MODEL_COLORS[model_type]
                
                # X position for this model at this iteration
                x_pos = iter_idx + model_offset
                
                # Add jitter to x positions for individual points (less jitter for cleaner look)
                jitter_strength = point_spacing * 0.15  # Reduced jitter
                x_jittered = x_pos + np.random.uniform(-jitter_strength, jitter_strength, len(model_data))
                
                # Plot individual points (all fold results)
                #ax.scatter(x_jittered, model_data, 
                #          color=color, alpha=0.4, s=40, zorder=10, 
                #          edgecolors='#333333', linewidths=0.5)
                
                # Calculate statistics
                mean_val = np.mean(model_data)
                sem = stats.sem(model_data)  # Standard error
                ci_95 = sem * stats.t.ppf(0.975, len(model_data) - 1)  # 95% CI
                
                # Plot mean as a larger marker
                ax.scatter(x_pos, mean_val, 
                          color=color, s=150, zorder=12, 
                          edgecolors='#333333', linewidths=2,
                          marker='D')  # Diamond marker for mean
                
                # Plot 95% confidence interval as error bar
                ax.errorbar(x_pos, mean_val, yerr=ci_95, 
                           color='#333333', linewidth=2, capsize=5, 
                           capthick=2, zorder=11, alpha=0.8)
                
                model_offset += point_spacing
    
    # Set x-axis ticks AND labels together
    ax.set_xticks(range(len(all_iterations)))
    ax.set_xticklabels([f'{int(it/1000)}k' if it >= 1000 else str(int(it)) 
                        for it in all_iterations], fontsize=22)
    
    # Add title if specified
    if add_title and title_text:
        ax.set_title(title_text, fontsize=26, fontweight='bold', pad=10)
    
    # Y-axis tick labels (conditional)
    if show_yticks:
        ax.set_yticklabels([f'{t:.2f}' for t in yticks], fontsize=22)
    else:
        ax.set_yticklabels([])  # Hide y-tick labels for fusion effect
    
    # Darken borders (highest zorder)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
        spine.set_zorder(20)
    
    # X-axis label (always show)
    ax.set_xlabel('Iteration', fontsize=24, fontweight='bold')
    
    # Y-axis label (conditional)
    if show_ylabel:
        ax.set_ylabel('AUC', fontsize=24, fontweight='bold')
    else:
        ax.set_ylabel('')  # Empty label

def main():
    # Base directories for different mutations as a list
    base_dirs = [
        "/data1/vanderbc/foundation_model_training_images/IMPACT/LUAD/checkpoints/EGFR",
        "/data1/vanderbc/foundation_model_training_images/IMPACT/BLCA/checkpoints/FGFR3"
    ]
    
    # Output directory
    output_dir = "/data1/vanderbc/nandas1/PostProc/mutation_benchmark_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Models to plot
    dinov1_models = [
        'TCGA_Dinov1_ViT-B_run2',
        'TCGA_TMEDinov1_version2_ViT-B',
        'TCGA_TMEDinov1_version3_ViT-B'
    ]
    
    dinov2_models = [
        'TCGA_Dinov2_ViT-B_run2',
        'TCGA_TMEDinov2_version2_ViT-B',
        'TCGA_TMEDinov2_version3_ViT-B'  # Note: this might be missing based on the directory listing
    ]
    
    # Calculate total number of subplots (2 dino versions × n_mutations)
    n_mutations = len(base_dirs)
    n_total_plots = n_mutations * 2  # One for Dinov1, one for Dinov2
    
    # Create figure with 1 row and n_total_plots columns
    # Adjust figure size - wider and less tall
    fig_width = 7 * n_total_plots  # 7 inches per subplot
    fig_height = 6  # Single row height
    
    # Create figure with GridSpec for custom spacing
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create GridSpec with custom width ratios and spacing
    # We want: [plot1][plot2] [gap] [plot3][plot4]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 5, figure=fig, 
                  width_ratios=[1, 1, 0.15, 1, 1],  # Middle column is a gap
                  wspace=0.02)  # Minimal space between columns
    
    # Create axes in the appropriate positions
    axes = [
        fig.add_subplot(gs[0, 0]),  # Plot 1: LUAD-EGFR-Dinov1
        fig.add_subplot(gs[0, 1]),  # Plot 2: LUAD-EGFR-Dinov2
        None,                        # Gap
        fig.add_subplot(gs[0, 3]),  # Plot 3: BLCA-FGFR3-Dinov1
        fig.add_subplot(gs[0, 4]),  # Plot 4: BLCA-FGFR3-Dinov2
    ]
    
    # First, collect all data for all mutations to calculate shared y-limits
    all_mutation_data = []
    mutation_info = []
    
    for base_dir in base_dirs:
        cancer_type, mutation_name = extract_cancer_and_mutation_from_path(base_dir)
        if cancer_type and mutation_name:
            all_models = dinov1_models + dinov2_models
            df = collect_data_for_models(base_dir, all_models, mutation_name)
            all_mutation_data.append(df)
            mutation_info.append((cancer_type, mutation_name))
    
    # Track which axes index to use (skipping the gap)
    axes_indices = [0, 1, 3, 4]  # Skip index 2 (the gap)
    subplot_idx = 0
    
    # Process each mutation with pre-calculated y-limits
    for mutation_idx, (df, (cancer_type, mutation_name)) in enumerate(zip(all_mutation_data, mutation_info)):
        
        # Create title from extracted values
        plot_title = f"{cancer_type}-{mutation_name}"
        
        print(f"\n{'='*60}")
        print(f"Processing {plot_title}...")
        print(f"{'='*60}")
        
        if df.empty:
            print(f"No valid data found for {plot_title}!")
            continue
        
        # Print summary
        print(f"\nData Summary for {plot_title}:")
        print(f"Total data points: {len(df)}")
        print(f"\nModels found:")
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            print(f"  - {model}: {len(model_data)} points")
        
        # Calculate shared y-limits for this mutation pair (Dinov1 and Dinov2)
        # This will now return limits that are multiples of 0.05
        ylim = calculate_shared_ylim(df, [dinov1_models, dinov2_models])
        print(f"Shared y-axis limits for {plot_title}: [{ylim[0]:.2f}, {ylim[1]:.2f}]")
        
        # Plot Dinov1 variants
        ax1 = axes[axes_indices[subplot_idx]]
        print(f"\nCreating Dinov1 plot for {plot_title}...")
        
        # Determine if this is the very first plot (LUAD-EGFR Dinov1)
        is_first_plot = (subplot_idx == 0)
        
        plot_model_group(df, dinov1_models, ax1, ylim, add_title=True, 
                        title_text=f"{plot_title} - Dinov1", 
                        show_ylabel=is_first_plot,  # Only show AUC label on first plot
                        show_yticks=True)            # Show y-ticks for first plot of each pair
        subplot_idx += 1
        
        # Plot Dinov2 variants
        ax2 = axes[axes_indices[subplot_idx]]
        print(f"Creating Dinov2 plot for {plot_title}...")
        
        plot_model_group(df, dinov2_models, ax2, ylim, add_title=True, 
                        title_text=f"{plot_title} - Dinov2",
                        show_ylabel=False,   # No AUC label for any other plot
                        show_yticks=False)   # Hide y-ticks for fusion effect
        subplot_idx += 1
        
        # Save summary statistics to CSV for this mutation
        summary_path = os.path.join(output_dir, f"{cancer_type}_{mutation_name}_summary_stats.csv")
        summary_stats = []
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            for iteration in sorted(model_data['iteration'].unique()):
                iter_data = model_data[model_data['iteration'] == iteration]
                summary_stats.append({
                    'model': model,
                    'iteration': iteration,
                    'mean_auc': iter_data['test_auc'].mean(),
                    'std_auc': iter_data['test_auc'].std(),
                    'min_auc': iter_data['test_auc'].min(),
                    'max_auc': iter_data['test_auc'].max(),
                    'n_splits': len(iter_data)
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved summary statistics: {summary_path}")
    
    # Create legend elements manually
    legend_elements = [
        Patch(facecolor=MODEL_COLORS['standard'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Standard'),
        Patch(facecolor=MODEL_COLORS['masked_only'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Masked Only'),
        Patch(facecolor=MODEL_COLORS['mixed'], alpha=0.7, edgecolor='#333333', 
              linewidth=1.5, label='Mixed')
    ]
    
    # Add figure-level legend at the top with more spacing
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12),
              ncol=3, fontsize=24, frameon=True, fancybox=False, 
              edgecolor='#333333', framealpha=0.95)
    
    # Adjust layout - leave more space for legend at top
    plt.tight_layout(rect=[0.02, 0, 1, 0.88])
    
    # Save plots
    output_path = os.path.join(output_dir, "mutation_benchmarks.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n✓ Saved PDF: {output_path}")
    
    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved PNG: {png_path}")
    
    plt.close()
    
    print(f"\n{'='*60}")
    print("✓ All visualizations complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
