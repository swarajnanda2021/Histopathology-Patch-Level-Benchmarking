#!/usr/bin/env python3
"""
Convert benchmarking JSON results to .dat files for LaTeX/pgfplots.

Usage:
    python convert_to_dat.py

This script reads the Monte Carlo classification metrics from all checkpoint iterations
for multiple training runs and produces .dat files organized by run name.

Output structure:
    dat_files/
    ├── FoundationModel_ViT-B_p16_b1024/
    │   ├── PCam.dat
    │   ├── MHIST.dat
    │   └── ...
    ├── FoundationModel_ViT-B_p16_b1024_adios/
    │   └── ...
    └── ...
"""

import os
import json
from pathlib import Path


# =============================================================================
# CONFIGURATION - Edit the list of benchmark directories here
# =============================================================================

BENCHMARK_BASE_DIR = "benchmark_results"
OUTPUT_BASE_DIR = "dat_files"

# List all your training run names here
# These correspond to subdirectories in benchmark_results/
TRAINING_RUNS = [
    "FoundationModel_ViT-B_p16_b1024",
    "FoundationModel_ViT-B_p16_b1024_adios",
    "FoundationModel_ViT-B_p16_b1024_adios_phaseB",
    "FoundationModel_ViT-B_p16_b1024_cellvit",
    "FoundationModel_ViT-B_p16_b1024_cellvit_phaseB",
    "FoundationModel_ViT-B_p16_b1024_phaseB",
    "FoundationModel_ViT-B_p16_b1024_random",
    "FoundationModel_ViT-B_p16_b1024_random_phaseB",
    "FoundationModel_ViT-L_p16_b2048",
    "FoundationModel_ViT-L_p16_b2048_adios",  # Uncomment and edit when ready
]

# Decimal precision for floating point values
PRECISION = 6

# =============================================================================
# END CONFIGURATION
# =============================================================================


def extract_iteration_number(iteration_dir):
    """Extract numeric iteration from directory name like 'iteration_25000' or 'iteration_final'."""
    name = os.path.basename(iteration_dir)
    if name == 'iteration_final':
        return float('inf'), 'final'
    else:
        try:
            iter_num = int(name.split('_')[1])
            return iter_num, str(iter_num)
        except (IndexError, ValueError):
            return None, None


def load_classification_metrics(benchmark_dir):
    """
    Load all classification metrics from a benchmark results directory.
    
    Returns:
        dict: {dataset_name: [(iteration, metrics_dict), ...]}
    """
    results = {}
    
    if not os.path.exists(benchmark_dir):
        return results
    
    # Find all iteration directories
    iteration_dirs = []
    for d in os.listdir(benchmark_dir):
        if d.startswith('iteration_') and os.path.isdir(os.path.join(benchmark_dir, d)):
            full_path = os.path.join(benchmark_dir, d)
            iter_num, iter_str = extract_iteration_number(full_path)
            if iter_num is not None:
                iteration_dirs.append((iter_num, iter_str, full_path))
    
    # Sort by iteration number
    iteration_dirs.sort(key=lambda x: x[0])
    
    # Process each iteration
    for iter_num, iter_str, iter_path in iteration_dirs:
        # Look for Monte Carlo metrics files
        for filename in os.listdir(iter_path):
            if filename.endswith('_monte_carlo_metrics.json'):
                dataset_name = filename.replace('_monte_carlo_metrics.json', '')
                
                filepath = os.path.join(iter_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        metrics = json.load(f)
                    
                    if dataset_name not in results:
                        results[dataset_name] = []
                    
                    results[dataset_name].append((iter_str, metrics))
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"  Warning: Could not load {filepath}: {e}")
    
    return results


def format_value(value, precision=6):
    """Format a numeric value for .dat file, handling None values."""
    if value is None:
        return 'nan'
    elif isinstance(value, float):
        return f"{value:.{precision}f}"
    else:
        return str(value)


def write_dat_file(dataset_name, iteration_metrics, output_dir, precision=6):
    """
    Write a .dat file for a single dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'PCam')
        iteration_metrics: List of (iteration_str, metrics_dict) tuples
        output_dir: Directory to write .dat files
        precision: Decimal precision for floating point values
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.dat")
    
    # Define columns
    columns = [
        'iteration',
        'acc_mean', 'acc_std', 'acc_ci_low', 'acc_ci_high',
        'auc_mean', 'auc_std', 'auc_ci_low', 'auc_ci_high',
        'f1_mean', 'f1_std', 'f1_ci_low', 'f1_ci_high',
        'n_iterations'
    ]
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Classification metrics for " + dataset_name + "\n")
        f.write("# Generated from Monte Carlo cross-validation results\n")
        f.write("# Columns: " + "  ".join(columns) + "\n")
        f.write("# CI values are 95% confidence intervals\n")
        f.write("#\n")
        
        # Write column headers (tab-separated for pgfplots compatibility)
        f.write("\t".join(columns) + "\n")
        
        # Write data rows
        for iter_str, metrics in iteration_metrics:
            row = []
            
            # Iteration
            row.append(iter_str)
            
            # Accuracy
            if 'accuracy' in metrics and isinstance(metrics['accuracy'], dict):
                acc = metrics['accuracy']
                row.append(format_value(acc.get('mean'), precision))
                row.append(format_value(acc.get('std'), precision))
                ci = acc.get('ci_95', [None, None])
                row.append(format_value(ci[0] if ci else None, precision))
                row.append(format_value(ci[1] if ci else None, precision))
            else:
                row.extend(['nan', 'nan', 'nan', 'nan'])
            
            # AUC
            if 'auc' in metrics and isinstance(metrics['auc'], dict):
                auc = metrics['auc']
                row.append(format_value(auc.get('mean'), precision))
                row.append(format_value(auc.get('std'), precision))
                ci = auc.get('ci_95', [None, None])
                row.append(format_value(ci[0] if ci else None, precision))
                row.append(format_value(ci[1] if ci else None, precision))
            else:
                row.extend(['nan', 'nan', 'nan', 'nan'])
            
            # F1
            if 'f1' in metrics and isinstance(metrics['f1'], dict):
                f1 = metrics['f1']
                row.append(format_value(f1.get('mean'), precision))
                row.append(format_value(f1.get('std'), precision))
                ci = f1.get('ci_95', [None, None])
                row.append(format_value(ci[0] if ci else None, precision))
                row.append(format_value(ci[1] if ci else None, precision))
            else:
                row.extend(['nan', 'nan', 'nan', 'nan'])
            
            # Number of completed iterations
            row.append(str(metrics.get('completed_iterations', 'nan')))
            
            f.write("\t".join(row) + "\n")
    
    return output_path


def process_training_run(run_name, benchmark_base_dir, output_base_dir, precision):
    """
    Process a single training run and generate .dat files.
    
    Args:
        run_name: Name of the training run (e.g., 'FoundationModel_ViT-B_p16_b1024')
        benchmark_base_dir: Base directory containing benchmark results
        output_base_dir: Base directory for output .dat files
        precision: Decimal precision
        
    Returns:
        dict: Summary of what was processed
    """
    benchmark_dir = os.path.join(benchmark_base_dir, run_name)
    output_dir = os.path.join(output_base_dir, run_name)
    
    summary = {
        'run_name': run_name,
        'exists': False,
        'datasets': {},
        'total_checkpoints': 0
    }
    
    # Check if directory exists
    if not os.path.exists(benchmark_dir):
        return summary
    
    summary['exists'] = True
    
    # Load metrics
    all_results = load_classification_metrics(benchmark_dir)
    
    if not all_results:
        return summary
    
    # Write .dat files for each dataset
    for dataset_name, metrics_list in all_results.items():
        output_path = write_dat_file(dataset_name, metrics_list, output_dir, precision)
        summary['datasets'][dataset_name] = len(metrics_list)
        summary['total_checkpoints'] = max(summary['total_checkpoints'], len(metrics_list))
    
    return summary


def main():
    print("=" * 70)
    print("Converting benchmark JSON results to .dat files")
    print("=" * 70)
    print()
    print(f"Benchmark base directory: {BENCHMARK_BASE_DIR}")
    print(f"Output base directory:    {OUTPUT_BASE_DIR}")
    print(f"Number of training runs:  {len(TRAINING_RUNS)}")
    print()
    
    # Create output base directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Process each training run
    all_summaries = []
    
    for run_name in TRAINING_RUNS:
        print(f"Processing: {run_name}")
        
        summary = process_training_run(
            run_name, 
            BENCHMARK_BASE_DIR, 
            OUTPUT_BASE_DIR, 
            PRECISION
        )
        
        all_summaries.append(summary)
        
        if not summary['exists']:
            print(f"  [SKIP] Directory not found")
        elif not summary['datasets']:
            print(f"  [SKIP] No classification metrics found")
        else:
            print(f"  [OK] {len(summary['datasets'])} datasets, {summary['total_checkpoints']} checkpoints")
            for dataset_name, n_checkpoints in summary['datasets'].items():
                print(f"       - {dataset_name}: {n_checkpoints} checkpoints")
        
        print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    processed = [s for s in all_summaries if s['datasets']]
    skipped_not_found = [s for s in all_summaries if not s['exists']]
    skipped_no_data = [s for s in all_summaries if s['exists'] and not s['datasets']]
    
    print(f"Processed:           {len(processed)}")
    print(f"Skipped (not found): {len(skipped_not_found)}")
    print(f"Skipped (no data):   {len(skipped_no_data)}")
    print()
    
    if processed:
        print("Processed runs:")
        for s in processed:
            print(f"  - {s['run_name']}: {len(s['datasets'])} datasets")
    
    if skipped_not_found:
        print()
        print("Not found (add later when benchmarking completes):")
        for s in skipped_not_found:
            print(f"  - {s['run_name']}")
    
    if skipped_no_data:
        print()
        print("No data yet (benchmarking in progress?):")
        for s in skipped_no_data:
            print(f"  - {s['run_name']}")
    
    print()
    print("=" * 70)
    print("Done!")
    print()
    print("Output structure:")
    print(f"  {OUTPUT_BASE_DIR}/")
    for s in processed:
        print(f"  ├── {s['run_name']}/")
        datasets = list(s['datasets'].keys())
        for i, dataset in enumerate(datasets):
            prefix = "│   └──" if i == len(datasets) - 1 else "│   ├──"
            print(f"  {prefix} {dataset}.dat")
    print()
    print("Example pgfplots usage:")
    print("  \\addplot table[x=iteration, y=acc_mean] {FoundationModel_ViT-B_p16_b1024/PCam.dat};")
    print()
    print("With error bars:")
    print("  \\addplot+[error bars/.cd, y dir=both, y explicit]")
    print("    table[x=iteration, y=acc_mean,")
    print("          y error minus expr={\\thisrow{acc_mean}-\\thisrow{acc_ci_low}},")
    print("          y error plus expr={\\thisrow{acc_ci_high}-\\thisrow{acc_mean}}]")
    print("    {FoundationModel_ViT-B_p16_b1024/PCam.dat};")


if __name__ == '__main__':
    main()
