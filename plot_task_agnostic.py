#!/usr/bin/env python3
"""
Script to create task-agnostic metrics plot comparing three model variants 
(Standard, Mixed, Masked) for Dinov1 and Dinov2 SSL methods.
Creates a 2 rows × 3 columns plot for RankMe, α-ReQ, and CLID metrics.
Uses ggplot styling with professional formatting.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob

# Set ggplot style and professional font
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.facecolor'] = '#F5F5F5'  # Lighter gray background
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

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
        
        # Plot task-agnostic metrics
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
                   markersize=5,
                   linewidth=2,
                   capsize=3,
                   capthick=1.2,
                   alpha=0.9,
                   label=label if show_legend else None)
        
        if aug_type not in lines_for_legend:
            lines_for_legend[aug_type] = (line, label)
    
    # Format x-axis
    format_iteration_axis(ax)
    
    # Darken borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return lines_for_legend

def create_task_agnostic_plot(comparisons, base_path, output_dir):
    """Create task-agnostic metrics plot with 2 rows × 3 columns"""
    
    print("\n" + "="*60)
    print("Creating Task-Agnostic Metrics Plot")
    print("="*60)
    
    # Define metrics
    metrics = [
        ('rankme', 'RankMe'),
        ('alphareq', r'$\alpha$-ReQ'),
        ('clid', 'CLID')
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    
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
            
            # Extract task-agnostic data for all three models
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
                ax.set_title(f'{metric_name}', fontsize=11, fontweight='bold', pad=10)
            
            # Add SSL method textbox in top-left corner of first column
            if col_idx == 0:
                ax.text(0.02, 0.98, comp['ssl_method'], transform=ax.transAxes,
                       fontsize=10, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333333',
                                linewidth=1.5, alpha=0.95))
            
            # Set y-label for first column
            if col_idx == 0:
                ax.set_ylabel('Score', fontsize=10, fontweight='bold')
            
            # Set x-label for bottom row
            if row_idx == 1:
                ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    
    # After all plotting is done, adjust y-axis for each column to fit data tightly
    for col_idx in range(3):
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
                  ncol=3, fontsize=11, frameon=True, fancybox=False,
                  edgecolor='#333333', framealpha=0.95)
    
    # Adjust layout with small gap between legend and plots
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "task_agnostic_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def main():
    # Base path for benchmark results
    base_path = "/data1/vanderbc/nandas1/PostProc/benchmark_results"
    output_dir = "patch_benchmark_plots"
    
    print("=" * 60)
    print("Creating Task-Agnostic Metrics Visualization")
    print("Using ggplot styling with professional formatting")
    print("=" * 60)
    
    # Define the 2 SSL methods with 3 models each
    comparisons = [
        {
            'model1_path': os.path.join(base_path, "TCGA_Dino_ViT-B_run2"),
            'model2_path': os.path.join(base_path, "TCGA_TMEDinov1_version2_ViT-B"),
            'model3_path': os.path.join(base_path, "TCGA_TMEDinov1_version3_ViT-B"),
            'model1_name': 'Dino_ViT-B',
            'model2_name': 'TMEDinov1_ViT-B_version2',
            'model3_name': 'TMEDinov1_ViT-B_version3',
            'ssl_method': 'Dinov1'
        },
        {
            'model1_path': os.path.join(base_path, "TCGA_Dinov2_ViT-B_run2"),
            'model2_path': os.path.join(base_path, "TCGA_TMEDinov2_version2_ViT-B"),
            'model3_path': os.path.join(base_path, "TCGA_TMEDinov2_version3_ViT-B"),
            'model1_name': 'Dinov2_ViT-B',
            'model2_name': 'TMEDinov2_ViT-B_version2',
            'model3_name': 'TMEDinov2_ViT-B_version3',
            'ssl_method': 'Dinov2'
        }
    ]
    
    # Create the task-agnostic metrics plot
    create_task_agnostic_plot(comparisons, base_path, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Task-Agnostic Metrics visualization complete!")
    print(f"✓ Output file: {output_dir}/task_agnostic_metrics.png (2×3 grid)")
    print("=" * 60)

if __name__ == "__main__":
    main()
