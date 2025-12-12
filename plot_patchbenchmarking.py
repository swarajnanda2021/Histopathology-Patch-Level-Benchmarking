#!/usr/bin/env python3
"""
Script to create three plots comparing three model variants (Standard, Mixed, Masked)
for Dinov1 and Dinov2 SSL methods:
1. Classification benchmarks plot: 2 rows × 5 columns
2. Segmentation benchmarks plot: 2 rows × 2 columns  
3. Task-agnostic metrics plot: 2 rows × 3 columns
Uses ggplot styling with professional formatting
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
from scipy import stats

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
        
        # MODIFIED: Handle 'iteration_final' as 300000
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

def extract_classification_data(metrics_data, dataset_name, metric='auc'):
    """Extract classification metrics from all_metrics.json"""
    data = []
    
    for entry in metrics_data:
        iteration = entry['iteration']
        metrics = entry['metrics']
        
        if dataset_name in metrics and 'monte_carlo' in metrics[dataset_name]:
            mc_data = metrics[dataset_name]['monte_carlo']
            
            if metric in mc_data and mc_data[metric] is not None:
                metric_dict = mc_data[metric]
                
                if isinstance(metric_dict, dict) and 'mean' in metric_dict:
                    row = {
                        'checkpoint_iteration': iteration,
                        'mean': metric_dict.get('mean'),
                        'ci_lower': None,
                        'ci_upper': None
                    }
                    
                    if 'ci_95' in metric_dict and metric_dict['ci_95'] is not None:
                        row['ci_lower'] = metric_dict['ci_95'][0]
                        row['ci_upper'] = metric_dict['ci_95'][1]
                    
                    data.append(row)
    
    return pd.DataFrame(data)

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



def load_segmentation_data(model_path, model_name, dataset):
    """Load segmentation data from CSV files and calculate mean with 95% CI"""
    csv_path = os.path.join(model_path, "extracted_trials", "segmentation_aji_trials.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['dataset'] == dataset].copy()
        
        if not df.empty:
            # Convert 'final' to 300000 before filtering
            df['checkpoint_iteration'] = df['checkpoint_iteration'].replace('final', '300000')
            
            # Convert to numeric
            df['checkpoint_numeric'] = pd.to_numeric(df['checkpoint_iteration'], errors='coerce')
            
            # Drop any rows where conversion failed
            df = df.dropna(subset=['checkpoint_numeric'])
            df['checkpoint_numeric'] = df['checkpoint_numeric'].astype(int)
            df['model'] = model_name
            
            # Calculate statistics with outlier removal
            stats_list = []
            
            for checkpoint in df['checkpoint_numeric'].unique():
                checkpoint_data = df[df['checkpoint_numeric'] == checkpoint]['aji_score'].values
                
                # Remove outliers using IQR method
                q1 = np.percentile(checkpoint_data, 25)
                q3 = np.percentile(checkpoint_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Filter out outliers
                filtered_data = checkpoint_data[(checkpoint_data >= lower_bound) & 
                                               (checkpoint_data <= upper_bound)]
                
                # Only proceed if we have enough data points after filtering
                if len(filtered_data) >= 3:  # Need at least 3 points for meaningful statistics
                    # Calculate mean
                    mean_val = np.mean(filtered_data)
                    
                    # Calculate 95% CI using standard error method
                    std_err = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data))
                    # Use t-distribution for small samples
                    from scipy import stats
                    confidence = 0.95
                    degrees_freedom = len(filtered_data) - 1
                    t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
                    
                    ci_lower = mean_val - t_value * std_err
                    ci_upper = mean_val + t_value * std_err
                    
                    stats_list.append({
                        'checkpoint_iteration': int(checkpoint),
                        'mean': mean_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_samples': len(filtered_data),
                        'n_outliers_removed': len(checkpoint_data) - len(filtered_data)
                    })
                else:
                    print(f"⚠️  Warning: Too few data points after outlier removal for checkpoint {checkpoint}")
            
            if stats_list:
                stats_df = pd.DataFrame(stats_list)
                # Sort by checkpoint for consistent plotting
                stats_df = stats_df.sort_values('checkpoint_iteration')
                
                # Print outlier removal summary if any outliers were removed
                total_outliers = stats_df['n_outliers_removed'].sum()
                if total_outliers > 0:
                    print(f"  Removed {total_outliers} outliers from {dataset} {model_name}")
                
                return stats_df, model_name
            else:
                print(f"⚠️  Warning: No valid data after outlier removal for {dataset} {model_name}")
    else:
        print(f"⚠️  Warning: Segmentation CSV not found at {csv_path}")
    
    return None, None


def plot_on_axis(ax, data_dict, dataset_name, metric_type, show_legend=False):
    """Plot data on a specific axis for three models - unified for all metric types"""
    
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
        
        # Unified plotting for all metric types (classification, segmentation, task-agnostic)
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
      

def create_classification_plot(comparisons, base_path, output_dir):
    """Create classification benchmarks plot with 2 rows × 5 columns"""
    
    print("\n" + "="*60)
    print("Creating Classification Benchmarks Plot")
    print("="*60)
    
    # Define datasets in order
    datasets = ['PCam', 'MiDOG', 'MHIST', 'CRC', 'BRACS']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    
    # Store lines for legend
    legend_lines = {}
    
    # Process each SSL method (row)
    for row_idx, comp in enumerate(comparisons):
        print(f"\nProcessing row {row_idx+1}: {comp['ssl_method']}")
        
        # Load metrics once per SSL method
        model1_metrics = load_all_metrics(comp['model1_path']) if os.path.exists(comp['model1_path']) else []
        model2_metrics = load_all_metrics(comp['model2_path']) if os.path.exists(comp['model2_path']) else []
        model3_metrics = load_all_metrics(comp['model3_path']) if os.path.exists(comp['model3_path']) else []
        
        # Process each dataset (column)
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            data_dict = {}
            
            # Extract classification data for all three models
            if model1_metrics:
                df1 = extract_classification_data(model1_metrics, dataset, 'auc')
                if not df1.empty:
                    data_dict[comp['model1_name']] = df1
            
            if model2_metrics:
                df2 = extract_classification_data(model2_metrics, dataset, 'auc')
                if not df2.empty:
                    data_dict[comp['model2_name']] = df2
            
            if model3_metrics:
                df3 = extract_classification_data(model3_metrics, dataset, 'auc')
                if not df3.empty:
                    data_dict[comp['model3_name']] = df3
            
            # Plot the data
            lines = plot_on_axis(ax, data_dict, dataset, 'classification', show_legend=False)
            legend_lines.update(lines)
            
            # Set column titles (dataset names) - only on top row
            if row_idx == 0:
                ax.set_title(f'{dataset}', fontsize=11, fontweight='bold', pad=10)
            
            # Add SSL method textbox in top-left corner of first column
            if col_idx == 0:
                ax.text(0.02, 0.98, comp['ssl_method'], transform=ax.transAxes,
                       fontsize=10, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333333',
                                linewidth=1.5, alpha=0.95))
            
            # Set y-label for first column
            if col_idx == 0:
                ax.set_ylabel('AUC', fontsize=10, fontweight='bold')
            
            # Set x-label for bottom row
            if row_idx == 1:
                ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    
    # After all plotting is done, adjust y-axis for each column to fit data tightly
    for col_idx in range(5):
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
            padding = 0.03 * y_range if y_range > 0 else 0.01
            
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
    output_path = os.path.join(output_dir, "classification_benchmarks.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def create_segmentation_plot(comparisons, base_path, output_dir):
    """Create segmentation benchmarks plot with 2 rows × 2 columns"""
    
    print("\n" + "="*60)
    print("Creating Segmentation Benchmarks Plot")
    print("="*60)
    
    # Define datasets
    datasets = ['MonuSeg', 'PanNuke']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    
    # Store lines for legend
    legend_lines = {}
    
    # Process each SSL method (row)
    for row_idx, comp in enumerate(comparisons):
        print(f"\nProcessing row {row_idx+1}: {comp['ssl_method']}")
        
        # Process each dataset (column)
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            data_dict = {}
            
            # Load segmentation data for all three models
            if os.path.exists(comp['model1_path']):
                stats1, name1 = load_segmentation_data(comp['model1_path'],
                                                       comp['model1_name'],
                                                       dataset)
                if stats1 is not None:
                    data_dict[name1] = stats1
            
            if os.path.exists(comp['model2_path']):
                stats2, name2 = load_segmentation_data(comp['model2_path'],
                                                       comp['model2_name'],
                                                       dataset)
                if stats2 is not None:
                    data_dict[name2] = stats2
            
            if os.path.exists(comp['model3_path']):
                stats3, name3 = load_segmentation_data(comp['model3_path'],
                                                       comp['model3_name'],
                                                       dataset)
                if stats3 is not None:
                    data_dict[name3] = stats3
            
            # Plot the data
            lines = plot_on_axis(ax, data_dict, dataset, 'segmentation', show_legend=False)
            legend_lines.update(lines)
            
            # Set column titles (dataset names) - only on top row
            if row_idx == 0:
                ax.set_title(f'{dataset}', fontsize=11, fontweight='bold', pad=10)
            
            # Add SSL method textbox in top-left corner of first column
            if col_idx == 0:
                ax.text(0.02, 0.98, comp['ssl_method'], transform=ax.transAxes,
                       fontsize=10, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333333',
                                linewidth=1.5, alpha=0.95))
            
            # Set y-label for first column
            if col_idx == 0:
                ax.set_ylabel('AJI', fontsize=10, fontweight='bold')
            
            # Set x-label for bottom row
            if row_idx == 1:
                ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    
    # After all plotting is done, adjust y-axis for each column to fit data tightly
    for col_idx in range(2):
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
            padding = 0.03 * y_range if y_range > 0 else 0.01
            
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
    output_path = os.path.join(output_dir, "segmentation_benchmarks.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

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
    print("Creating Three-Model Benchmark Visualizations")
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
    
    # Create the three plots
    create_classification_plot(comparisons, base_path, output_dir)
    create_segmentation_plot(comparisons, base_path, output_dir)
    create_task_agnostic_plot(comparisons, base_path, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ All visualizations complete!")
    print(f"✓ Output files:")
    print(f"  1. {output_dir}/classification_benchmarks.png (2×5 grid)")
    print(f"  2. {output_dir}/segmentation_benchmarks.png (2×2 grid)")
    print(f"  3. {output_dir}/task_agnostic_metrics.png (2×3 grid)")
    print("=" * 60)

if __name__ == "__main__":
    main()
