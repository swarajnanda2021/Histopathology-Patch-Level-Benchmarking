#!/usr/bin/env python3
"""
Script to create a unified patch benchmark plot comparing model variants 
for Dinov2 SSL method.
Combines classification benchmarks (PCam, MiDOG, MHIST, BRACS) with 
segmentation benchmark (PanNuke) in a single 1×5 grid.
Uses ggplot styling with professional formatting.
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
mpl.rcParams['font.size'] = 14
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
    'random_masking': '#43A047',  # Professional green
}

# Map model names to augmentation types
MODEL_TO_AUGMENTATION = {
    'Dino_ViT-B': 'standard',
    'TMEDinov1_ViT-B_version2': 'mixed',
    'TMEDinov1_ViT-B_version3': 'masked',
    'Dinov2_ViT-B': 'standard',
    'TMEDinov2_ViT-B_version2': 'mixed',
    'TMEDinov2_ViT-B_version3': 'masked',
    'TMEDinov2_ViT-B_random_masking': 'random_masking',
    # New model names
    'vanilla': 'vanilla',
    'masked': 'masked',
    'all': 'all',
    'unmasked': 'unmasked'
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

def load_segmentation_data(model_path, model_name, dataset='PanNuke'):
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
    """Plot data on a specific axis for multiple models
    data_dict now includes color info: {model_name: {'data': df, 'color': color_string, 'label': label_string}}
    """
    
    lines_for_legend = {}
    
    for model_name, model_info in data_dict.items():
        df = model_info['data']
        color = model_info['color']
        label = model_info['label']
        
        if df is None or df.empty:
            continue
        
        # Unified plotting for all metric types
        df_sorted = df.sort_values('checkpoint_iteration')
        x = df_sorted['checkpoint_iteration']
        y = df_sorted['mean']
        
        # Handle confidence intervals
        if 'ci_lower' in df_sorted.columns and 'ci_upper' in df_sorted.columns:
            yerr_lower = y - df_sorted['ci_lower'].fillna(y)
            yerr_upper = df_sorted['ci_upper'].fillna(y) - y
        else:
            yerr_lower = yerr_upper = 0
        
        line, = ax.plot(x, y,
                color=color,
                marker='o',
                markersize=5,
                linewidth=2,
                alpha=0.9,
                label=label if show_legend else None)

        # Add the error envelope (translucent shaded area)
        if 'ci_lower' in df_sorted.columns and 'ci_upper' in df_sorted.columns:
            ax.fill_between(x, 
                            df_sorted['ci_lower'].fillna(y),
                            df_sorted['ci_upper'].fillna(y),
                            color=color,
                            alpha=0.2,  # Translucent envelope
                            edgecolor='none')

        if label not in [l for _, l in lines_for_legend.values()]:
            lines_for_legend[label] = (line, label)
    
    # Format x-axis
    format_iteration_axis(ax)
    
    # Darken borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return lines_for_legend

def create_unified_patch_benchmark_plot(comparisons, base_path, output_dir):
    """Create unified patch benchmark plot with 1 row × 5 columns for Dinov2 only
    Columns: PCam, MiDOG, MHIST, BRACS, PanNuke"""
    
    print("\n" + "="*60)
    print("Creating Unified Patch Benchmark Plot (Dinov2 Only)")
    print("="*60)
    
    # Define datasets in order (4 classification + 1 segmentation)
    datasets = ['PCam', 'MiDOG', 'MHIST', 'BRACS', 'PanNuke']
    is_segmentation = [False, False, False, False, True]
    
    # Create figure with subplots - single row
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    # Store lines for legend
    legend_lines = {}
    
    # Process only the first comparison (should be just one now)
    comp = comparisons[0]
    print(f"\nProcessing: {comp['ssl_method']}")
    
    # Load metrics for all models
    models_info = []
    for i in range(1, 10):  # Support up to 9 models (model1 through model9)
        path_key = f'model{i}_path'
        name_key = f'model{i}_name'
        color_key = f'model{i}_color'
        
        if path_key in comp and name_key in comp and color_key in comp:
            if os.path.exists(comp[path_key]):
                metrics = load_all_metrics(comp[path_key])
                models_info.append({
                    'path': comp[path_key],
                    'name': comp[name_key],
                    'color': comp[color_key],
                    'metrics': metrics
                })
    
    # Process each dataset (column)
    for col_idx, dataset in enumerate(datasets):
        ax = axes[col_idx]
        data_dict = {}
        
        if is_segmentation[col_idx]:
            # Load segmentation data for PanNuke
            for model_info in models_info:
                stats, name = load_segmentation_data(model_info['path'],
                                                     model_info['name'],
                                                     'PanNuke')
                if stats is not None:
                    data_dict[name] = {
                        'data': stats,
                        'color': model_info['color'],
                        'label': model_info['name']
                    }
            
            metric_type = 'segmentation'
        else:
            # Extract classification data
            for model_info in models_info:
                if model_info['metrics']:
                    df = extract_classification_data(model_info['metrics'], dataset, 'auc')
                    if not df.empty:
                        data_dict[model_info['name']] = {
                            'data': df,
                            'color': model_info['color'],
                            'label': model_info['name']
                        }
            
            metric_type = 'classification'
        
        # Plot the data
        lines = plot_on_axis(ax, data_dict, dataset, metric_type, show_legend=False)
        legend_lines.update(lines)
        
        # Set column titles (dataset names)
        ax.set_title(f'{dataset}', fontsize=16, fontweight='bold', pad=10)
        
        # Set y-label for first column
        if col_idx == 0:
            if is_segmentation[col_idx]:
                ax.set_ylabel('AJI', fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('AUC', fontsize=14, fontweight='bold')
        
        # Set y-label for PanNuke column (since it's different metric)
        if col_idx == 4:  # PanNuke column
            ax.set_ylabel('AJI', fontsize=14, fontweight='bold')
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        
        # Set x-label for all columns
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    
    # Adjust y-axis for each column to fit data tightly
    for col_idx in range(5):
        ax = axes[col_idx]
        if len(ax.lines) > 0:  # Only if there's data
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            # Add just 3% padding around the data
            padding = 0.03 * y_range if y_range > 0 else 0.01
            
            ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add legend
    if legend_lines:
        handles = [line for line, _ in legend_lines.values()]
        labels = [label for _, label in legend_lines.values()]
        
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                  ncol=len(handles), fontsize=16, frameon=True, fancybox=False,
                  edgecolor='#333333', framealpha=0.95)
    
    # Adjust layout with small gap between legend and plots
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "patch_benchmarks_ICCV.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def main():
    # Base path for benchmark results
    base_path = "/data1/vanderbc/nandas1/PostProc/benchmark_results"
    output_dir = "patch_benchmark_plots"
    
    print("=" * 60)
    print("Creating Unified Patch Benchmark Visualization (Dinov2 Only)")
    print("Combining Classification (PCam, MiDOG, MHIST, BRACS) and Segmentation (PanNuke)")
    print("Using ggplot styling with professional formatting")
    print("=" * 60)
    
    # Define comparisons with color specifications in main
    comparisons = [
        {
            'model1_path': os.path.join(base_path, "TCGA_Dinov2_ViT-B_run2"),
            'model1_name': 'vanilla',
            'model1_color': '#1E88E5',  # Blue
            
            'model2_path': os.path.join(base_path, "TCGA_TMEDinov3_ViT-B_B2_seqpacking"),
            'model2_name': 'ibot-patches',
            'model2_color': '#DC3545',  # Red
            
            'model3_path': os.path.join(base_path, "TCGA_TMEDinov3_ViT-B_B3_seqpacking"),
            'model3_name': 'all-patches',
            'model3_color': '#FFA726',  # Orange
            
            'model4_path': os.path.join(base_path, "TCGA_TMEDinov3_ViT-B_B4_seqpacking"),
            'model4_name': '~ibot-patches',
            'model4_color': '#43A047',  # Green
            
            'ssl_method': 'Dinov2'
        }
    ]
    
    # Create the unified plot
    create_unified_patch_benchmark_plot(comparisons, base_path, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print(f"✓ Output file: {output_dir}/patch_benchmarks_ICCV.png (1×5 grid)")
    print("=" * 60)

if __name__ == "__main__":
    main()
