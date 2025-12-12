#!/usr/bin/env python3
"""
Script to extract final checkpoint results for all DINOv2 variants across
patch-level and slide-level benchmarks, and generate both console output
and LaTeX table for ICLR paper format.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import re

# Define the DINOv2 models
DINOV2_MODELS = {
    'Standard': 'TCGA_Dinov2_ViT-B_run2',
    'Mixed': 'TCGA_TMEDinov2_version2_ViT-B',
    'Masked Only': 'TCGA_TMEDinov2_version3_ViT-B'
}

# Define paths
PATCH_BASE_PATH = "/data1/vanderbc/nandas1/PostProc/benchmark_results"
SLIDE_BASE_PATHS = {
    'LUAD-EGFR': "/data1/vanderbc/foundation_model_training_images/IMPACT/LUAD/checkpoints/EGFR",
    'BLCA-FGFR3': "/data1/vanderbc/foundation_model_training_images/IMPACT/BLCA/checkpoints/FGFR3"
}

def load_patch_classification_metric(model_path, dataset_name, metric='auc'):
    """Load classification metrics from all_metrics.json for final checkpoint"""
    metrics_file = os.path.join(model_path, "iteration_300000", "all_metrics.json")
    
    # Also check for 'iteration_final' if 300000 doesn't exist
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(model_path, "iteration_final", "all_metrics.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            try:
                metrics = json.load(f)
                if dataset_name in metrics and 'monte_carlo' in metrics[dataset_name]:
                    mc_data = metrics[dataset_name]['monte_carlo']
                    
                    if metric in mc_data and mc_data[metric] is not None:
                        metric_dict = mc_data[metric]
                        
                        if isinstance(metric_dict, dict) and 'mean' in metric_dict:
                            result = {
                                'mean': metric_dict.get('mean'),
                                'ci_lower': None,
                                'ci_upper': None
                            }
                            
                            if 'ci_95' in metric_dict and metric_dict['ci_95'] is not None:
                                result['ci_lower'] = metric_dict['ci_95'][0]
                                result['ci_upper'] = metric_dict['ci_95'][1]
                            
                            return result
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {metrics_file}")
    
    return None

def load_pannuke_segmentation(model_path):
    """Load PanNuke segmentation results for final checkpoint"""
    csv_path = os.path.join(model_path, "extracted_trials", "segmentation_aji_trials.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['dataset'] == 'PanNuke'].copy()
        
        if not df.empty:
            # Filter for final checkpoint (300000 or 'final')
            df_final = df[(df['checkpoint_iteration'] == '300000') | 
                         (df['checkpoint_iteration'] == 'final')].copy()
            
            if df_final.empty:
                # If no exact match, get the highest checkpoint
                df['checkpoint_numeric'] = df['checkpoint_iteration'].replace('final', '300000')
                df['checkpoint_numeric'] = pd.to_numeric(df['checkpoint_numeric'], errors='coerce')
                df = df.dropna(subset=['checkpoint_numeric'])
                if not df.empty:
                    max_checkpoint = df['checkpoint_numeric'].max()
                    df_final = df[df['checkpoint_numeric'] == max_checkpoint]
            
            if not df_final.empty:
                aji_scores = df_final['aji_score'].values
                
                # Remove outliers using IQR
                q1 = np.percentile(aji_scores, 25)
                q3 = np.percentile(aji_scores, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_data = aji_scores[(aji_scores >= lower_bound) & 
                                          (aji_scores <= upper_bound)]
                
                if len(filtered_data) >= 3:
                    mean_val = np.mean(filtered_data)
                    std_err = np.std(filtered_data, ddof=1) / np.sqrt(len(filtered_data))
                    
                    # Calculate 95% CI
                    confidence = 0.95
                    degrees_freedom = len(filtered_data) - 1
                    t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
                    
                    return {
                        'mean': mean_val,
                        'ci_lower': mean_val - t_value * std_err,
                        'ci_upper': mean_val + t_value * std_err
                    }
    
    return None

def load_slide_mutation_results(base_dir, mutation_name, model_name):
    """Load slide-level mutation prediction results for highest iteration"""
    pattern = os.path.join(base_dir, 
                           f"{model_name}_checkpoint_iter_*",
                           f"{mutation_name}_gma",
                           f"{mutation_name}_split_*_set_{model_name}*_convergence_{mutation_name}_gma.csv")
    
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        return None
    
    # Extract data from all files
    all_data = []
    max_iteration = 0
    
    for csv_file in csv_files:
        # Extract iteration
        match = re.search(r'iter_(\d+)', csv_file)
        if match:
            iteration = int(match.group(1))
            max_iteration = max(max_iteration, iteration)
    
    # Now get only the data from max iteration
    max_iter_data = []
    
    for csv_file in csv_files:
        match = re.search(r'iter_(\d+)', csv_file)
        if match and int(match.group(1)) == max_iteration:
            try:
                df = pd.read_csv(csv_file)
                df = df.dropna(subset=['test_auc'])
                df = df[df['test_auc'] != '']
                
                if not df.empty:
                    highest_epoch = df['epoch'].max()
                    highest_epoch_data = df[df['epoch'] == highest_epoch]
                    
                    if not highest_epoch_data.empty:
                        max_iter_data.append(highest_epoch_data['test_auc'].iloc[0])
            except Exception as e:
                print(f"Warning: Error reading {csv_file}: {e}")
    
    if max_iter_data:
        mean_val = np.mean(max_iter_data)
        
        if len(max_iter_data) >= 2:
            std_err = np.std(max_iter_data, ddof=1) / np.sqrt(len(max_iter_data))
            
            # Calculate 95% CI
            confidence = 0.95
            degrees_freedom = len(max_iter_data) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
            
            return {
                'mean': mean_val,
                'ci_lower': mean_val - t_value * std_err,
                'ci_upper': mean_val + t_value * std_err,
                'iteration': max_iteration
            }
        else:
            return {
                'mean': mean_val,
                'ci_lower': mean_val,
                'ci_upper': mean_val,
                'iteration': max_iteration
            }
    
    return None

def format_result(result):
    """Format result as mean ± CI"""
    if result is None:
        return "N/A"
    
    mean = result['mean']
    
    if result['ci_lower'] is not None and result['ci_upper'] is not None:
        # Calculate the CI range (symmetric around mean for display)
        ci_range = (result['ci_upper'] - result['ci_lower']) / 2
        return f"{mean:.3f} ± {ci_range:.3f}"
    else:
        return f"{mean:.3f}"

def format_latex_result(result, is_best=False):
    """Format result for LaTeX with proper number formatting"""
    if result is None:
        return "---"
    
    mean = result['mean']
    
    if result['ci_lower'] is not None and result['ci_upper'] is not None:
        ci_range = (result['ci_upper'] - result['ci_lower']) / 2
        if is_best:
            # Bold the numbers separately, keep ± in math mode
            formatted = f"\\textbf{{{mean:.3f}}} $\\pm$ \\textbf{{{ci_range:.3f}}}"
        else:
            formatted = f"{mean:.3f} $\\pm$ {ci_range:.3f}"
    else:
        if is_best:
            formatted = f"\\textbf{{{mean:.3f}}}"
        else:
            formatted = f"{mean:.3f}"
    
    return formatted

def main():
    print("="*80)
    print("DINOV2 FINAL CHECKPOINT RESULTS EXTRACTION")
    print("="*80)
    print()
    
    # Initialize results dictionary
    results = {variant: {} for variant in DINOV2_MODELS.keys()}
    
    # Process patch-level benchmarks
    print("Loading Patch-Level Benchmarks...")
    print("-"*40)
    
    patch_benchmarks = ['PCam', 'MiDOG', 'MHIST', 'BRACS']
    
    for variant_name, model_name in DINOV2_MODELS.items():
        model_path = os.path.join(PATCH_BASE_PATH, model_name)
        
        print(f"\n{variant_name} ({model_name}):")
        
        # Classification benchmarks
        for benchmark in patch_benchmarks:
            result = load_patch_classification_metric(model_path, benchmark, 'auc')
            results[variant_name][f"{benchmark} (AUC)"] = result
            print(f"  {benchmark}: {format_result(result)}")
        
        # PanNuke segmentation
        result = load_pannuke_segmentation(model_path)
        results[variant_name]["PanNuke (AJI)"] = result
        print(f"  PanNuke: {format_result(result)}")
    
    # Process slide-level mutation benchmarks
    print("\n" + "="*80)
    print("Loading Slide-Level Mutation Benchmarks...")
    print("-"*40)
    
    for mutation_key, base_dir in SLIDE_BASE_PATHS.items():
        print(f"\n{mutation_key}:")
        
        # Extract mutation name for file matching
        mutation_name = mutation_key.split('-')[1]  # EGFR or FGFR3
        
        for variant_name, model_name in DINOV2_MODELS.items():
            result = load_slide_mutation_results(base_dir, mutation_name, model_name)
            results[variant_name][f"{mutation_key} (AUC)"] = result
            
            if result:
                print(f"  {variant_name} (iter {result['iteration']}): {format_result(result)}")
            else:
                print(f"  {variant_name}: N/A")
    
    # Find best performing model for each benchmark
    benchmarks = ["PCam (AUC)", "MiDOG (AUC)", "MHIST (AUC)", "BRACS (AUC)", 
                  "PanNuke (AJI)", "LUAD-EGFR (AUC)", "BLCA-FGFR3 (AUC)"]
    
    best_performers = {}
    for benchmark in benchmarks:
        best_variant = None
        best_score = -float('inf')
        
        for variant in DINOV2_MODELS.keys():
            if benchmark in results[variant] and results[variant][benchmark] is not None:
                score = results[variant][benchmark]['mean']
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        best_performers[benchmark] = best_variant
    
    # Print console table with best performers in bold (using ANSI codes)
    print("\n" + "="*80)
    print("SUMMARY TABLE (Best performers marked with *)")
    print("="*80)
    
    # Prepare column headers
    headers = ["Model Variant", "PCam (AUC)", "MiDOG (AUC)", "MHIST (AUC)", 
               "BRACS (AUC)", "PanNuke (AJI)", "LUAD-EGFR (AUC)", "BLCA-FGFR3 (AUC)"]
    
    # Print header
    print("\n" + "\t".join(headers))
    print("-" * 120)
    
    # Print results with best performers marked
    for variant in DINOV2_MODELS.keys():
        row = [variant]
        for benchmark in benchmarks:
            result_str = format_result(results[variant].get(benchmark))
            # Mark best performer with asterisk
            if best_performers.get(benchmark) == variant:
                result_str = f"*{result_str}*"
            row.append(result_str)
        print("\t".join(row))
    
    print("\n* = Best performing model for that benchmark")
    
    # Generate LaTeX table with bold best performers
    print("\n" + "="*80)
    print("LATEX TABLE (ICLR Format) - Best performers in bold")
    print("="*80)
    print()
    
    latex_code = r"""
\begin{table}[t]
\centering
\caption{Performance comparison of DINOv2 training strategies across patch-level and slide-level benchmarks. Results show mean ± 95\% CI on test sets. Best results for each benchmark are shown in \textbf{bold}.}
\label{tab:dinov2_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cccc|c|cc}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{4}{c|}{\textbf{Patch Classification (AUC)}} & \textbf{Patch Segmentation} & \multicolumn{2}{c}{\textbf{Slide Mutation (AUC)}} \\
\cmidrule{2-8}
& PCam & MiDOG & MHIST & BRACS & PanNuke (AJI) & LUAD-EGFR & BLCA-FGFR3 \\
\midrule"""
    
    # Add data rows with bold formatting for best results
    for variant in DINOV2_MODELS.keys():
        row_data = [variant]
        for benchmark in benchmarks:
            is_best = (best_performers.get(benchmark) == variant)
            row_data.append(format_latex_result(results[variant].get(benchmark), is_best))
        
        latex_code += f"\n{row_data[0]} & {' & '.join(row_data[1:])} \\\\"
    
    latex_code += r"""
\bottomrule
\end{tabular}%
}
\end{table}"""
    
    print(latex_code)
    
    # Print best performing models summary
    print("\n" + "="*80)
    print("BEST PERFORMING MODELS SUMMARY")
    print("="*80)
    
    for benchmark in benchmarks:
        if best_performers[benchmark]:
            best_variant = best_performers[benchmark]
            best_score = results[best_variant][benchmark]['mean']
            print(f"{benchmark:20s}: {best_variant:12s} ({best_score:.3f})")

if __name__ == "__main__":
    main()
