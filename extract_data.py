import json
import pandas as pd
import os
from pathlib import Path
import numpy as np

def extract_benchmark_trials(output_dir, save_dir=None):
    """
    Extract all individual trial results from benchmark JSON files and save to separate CSVs.
    
    Args:
        output_dir: Directory containing the benchmark outputs (with iteration_* folders)
        save_dir: Directory to save CSV files (if None, creates 'extracted_trials' in output_dir)
    """
    if save_dir is None:
        save_dir = os.path.join(output_dir, "extracted_trials")
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all iteration directories
    iteration_dirs = []
    for d in os.listdir(output_dir):
        if d.startswith('iteration_') and os.path.isdir(os.path.join(output_dir, d)):
            iteration_dirs.append(d)
    
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # 1. Extract Monte Carlo classification AUC values only
    classification_auc_trials = []
    
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        # Look for Monte Carlo metrics files
        for file in os.listdir(iter_path):
            if file.endswith('_monte_carlo_metrics.json'):
                dataset_name = file.replace('_monte_carlo_metrics.json', '')
                
                with open(os.path.join(iter_path, file), 'r') as f:
                    data = json.load(f)
                
                # Extract individual iterations - AUC only
                if 'all_iterations' in data:
                    for trial in data['all_iterations']:
                        if trial.get('test_auc') is not None:  # Only include trials with valid AUC
                            record = {
                                'checkpoint_iteration': iteration,
                                'dataset': dataset_name,
                                'trial_iteration': trial['iteration'],
                                'test_auc': trial['test_auc']
                            }
                            classification_auc_trials.append(record)
    
    # Save classification AUC trials to CSV
    if classification_auc_trials:
        df_classification_auc = pd.DataFrame(classification_auc_trials)
        csv_path = os.path.join(save_dir, 'classification_auc_trials.csv')
        df_classification_auc.to_csv(csv_path, index=False)
        print(f"Saved {len(classification_auc_trials)} classification AUC trials to {csv_path}")
    
    # 2. Extract segmentation trials (individual AJI scores)
    segmentation_aji_trials = []
    
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        # Look for segmentation metrics files
        for file in os.listdir(iter_path):
            if file.endswith('_segmentation_metrics.json'):
                dataset_name = file.replace('_segmentation_metrics.json', '')
                
                with open(os.path.join(iter_path, file), 'r') as f:
                    data = json.load(f)
                
                # Extract individual AJI scores
                if 'aji' in data:
                    for idx, aji_score in enumerate(data['aji']):
                        record = {
                            'checkpoint_iteration': iteration,
                            'dataset': dataset_name,
                            'sample_idx': idx,
                            'aji_score': aji_score
                        }
                        segmentation_aji_trials.append(record)
    
    # Save segmentation AJI trials to CSV
    if segmentation_aji_trials:
        df_segmentation = pd.DataFrame(segmentation_aji_trials)
        csv_path = os.path.join(save_dir, 'segmentation_aji_trials.csv')
        df_segmentation.to_csv(csv_path, index=False)
        print(f"Saved {len(segmentation_aji_trials)} segmentation AJI trials to {csv_path}")
    
    # 3. Extract task-agnostic metrics - each metric in separate CSV
    
    # RankMe bootstrap samples
    rankme_trials = []
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        rankme_file = os.path.join(iter_path, 'task_agnostic_metrics', 'rankme_metric.json')
        if os.path.exists(rankme_file):
            with open(rankme_file, 'r') as f:
                data = json.load(f)
            
            if 'bootstrap_statistics' in data:
                for stat in data['bootstrap_statistics']:
                    if 'values' in stat:
                        for idx, value in enumerate(stat['values']):
                            record = {
                                'checkpoint_iteration': iteration,
                                'sample_size': stat['size'],
                                'bootstrap_idx': idx,
                                'rankme_value': value
                            }
                            rankme_trials.append(record)
    
    if rankme_trials:
        df_rankme = pd.DataFrame(rankme_trials)
        csv_path = os.path.join(save_dir, 'rankme_bootstrap_trials.csv')
        df_rankme.to_csv(csv_path, index=False)
        print(f"Saved {len(rankme_trials)} RankMe bootstrap trials to {csv_path}")
    
    # CLID bootstrap samples
    clid_trials = []
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        clid_file = os.path.join(iter_path, 'task_agnostic_metrics', 'clid_metric.json')
        if os.path.exists(clid_file):
            with open(clid_file, 'r') as f:
                data = json.load(f)
            
            if 'bootstrap_statistics' in data:
                for stat in data['bootstrap_statistics']:
                    if 'values' in stat:
                        for idx, value in enumerate(stat['values']):
                            record = {
                                'checkpoint_iteration': iteration,
                                'sample_size': stat['size'],
                                'bootstrap_idx': idx,
                                'clid_value': value
                            }
                            clid_trials.append(record)
    
    if clid_trials:
        df_clid = pd.DataFrame(clid_trials)
        csv_path = os.path.join(save_dir, 'clid_bootstrap_trials.csv')
        df_clid.to_csv(csv_path, index=False)
        print(f"Saved {len(clid_trials)} CLID bootstrap trials to {csv_path}")
    
    # Alpha-ReQ bootstrap samples
    alphareq_trials = []
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        alphareq_file = os.path.join(iter_path, 'task_agnostic_metrics', 'alphareq_metric.json')
        if os.path.exists(alphareq_file):
            with open(alphareq_file, 'r') as f:
                data = json.load(f)
            
            if 'bootstrap_statistics' in data:
                for stat in data['bootstrap_statistics']:
                    if 'values' in stat:
                        for idx, value in enumerate(stat['values']):
                            record = {
                                'checkpoint_iteration': iteration,
                                'sample_size': stat['size'],
                                'bootstrap_idx': idx,
                                'alphareq_value': value
                            }
                            alphareq_trials.append(record)
    
    if alphareq_trials:
        df_alphareq = pd.DataFrame(alphareq_trials)
        csv_path = os.path.join(save_dir, 'alphareq_bootstrap_trials.csv')
        df_alphareq.to_csv(csv_path, index=False)
        print(f"Saved {len(alphareq_trials)} Alpha-ReQ bootstrap trials to {csv_path}")
    
    # LiDAR bootstrap samples
    lidar_trials = []
    for iter_dir in iteration_dirs:
        iter_path = os.path.join(output_dir, iter_dir)
        # MODIFIED: Handle 'iteration_final' as 300000
        iteration = iter_dir.replace('iteration_', '')
        if iteration == 'final':
            iteration = '300000'
        
        lidar_file = os.path.join(iter_path, 'task_agnostic_metrics', 'lidar_metric.json')
        if os.path.exists(lidar_file):
            with open(lidar_file, 'r') as f:
                data = json.load(f)
            
            if 'bootstrap_statistics' in data:
                for stat in data['bootstrap_statistics']:
                    if 'values' in stat:
                        for idx, value in enumerate(stat['values']):
                            record = {
                                'checkpoint_iteration': iteration,
                                'n_images': stat['images'],
                                'n_augs': stat['augs'],
                                'bootstrap_idx': idx,
                                'lidar_value': value
                            }
                            lidar_trials.append(record)
    
    if lidar_trials:
        df_lidar = pd.DataFrame(lidar_trials)
        csv_path = os.path.join(save_dir, 'lidar_bootstrap_trials.csv')
        df_lidar.to_csv(csv_path, index=False)
        print(f"Saved {len(lidar_trials)} LiDAR bootstrap trials to {csv_path}")
    
    # 4. Create a summary of what was extracted
    summary = {
        'classification_auc_trials': len(classification_auc_trials),
        'segmentation_aji_trials': len(segmentation_aji_trials),
        'rankme_bootstrap_trials': len(rankme_trials),
        'clid_bootstrap_trials': len(clid_trials),
        'alphareq_bootstrap_trials': len(alphareq_trials),
        'lidar_bootstrap_trials': len(lidar_trials),
        'total_trials': len(classification_auc_trials) + len(segmentation_aji_trials) + 
                       len(rankme_trials) + len(clid_trials) + len(alphareq_trials) + len(lidar_trials)
    }
    
    # Save summary as both JSON and CSV for convenience
    with open(os.path.join(save_dir, 'extraction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save summary as CSV
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(save_dir, 'extraction_summary.csv'), index=False)
    
    print("\nExtraction Summary:")
    print("-" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nAll CSV files saved to: {save_dir}")
    
    # Print the list of created files
    print("\nCreated files:")
    for file in sorted(os.listdir(save_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(save_dir, file)
            df = pd.read_csv(file_path)
            print(f"  - {file} ({len(df)} rows)")
    
    return save_dir


# MODIFIED: Main execution block to handle multiple directories
if __name__ == "__main__":
    # Base directory
    base_dir = "/data1/vanderbc/nandas1/PostProc/benchmark_results"
    
    # List of models to process
    models = [
        #"TCGA_Dino_ViT-B_run2",
        #"TCGA_Dinov2_ViT-B_run2",
        #"TCGA_TMEDinov1_version2_ViT-B",
        #"TCGA_TMEDinov1_version3_ViT-B",
        #"TCGA_TMEDinov2_version2_ViT-B",
        #"TCGA_TMEDinov2_version3_ViT-B",
        #"TCGA_TMEDinov2_version4_random_masking_ViT-B"
        "TCGA_TMEDinov3_ViT-B_B4_seqpacking"
    ]
    
    # Process each model directory
    for model in models:
        output_directory = os.path.join(base_dir, model)
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        
        if os.path.exists(output_directory):
            # Extract all trials for this model
            extracted_dir = extract_benchmark_trials(output_directory)
        else:
            print(f"WARNING: Directory not found: {output_directory}")
