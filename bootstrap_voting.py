#!/usr/bin/env python3
"""
Simple Checkpoint Selector with Bootstrap Voting
Non-consensus aware version using actual trial resampling
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class DualMetricSelector:
    """Original dual-metric selection algorithm without consensus tracking"""
    
    def __init__(self):
        pass
    
    def normalize_metrics(self, P_ts, P_ta, ts_minimize=None, ta_minimize=None):
        """Normalize metrics to [0,1] range, handling minimize/maximize"""
        # Initialize
        n_epochs, n_ts = P_ts.shape
        _, n_ta = P_ta.shape
        
        if ts_minimize is None:
            ts_minimize = [False] * n_ts
        if ta_minimize is None:
            ta_minimize = [False] * n_ta
        
        # Normalize task-specific metrics
        P_ts_norm = np.zeros_like(P_ts)
        for i in range(n_ts):
            col = P_ts[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 0:
                if ts_minimize[i]:
                    # For minimization: invert so lower values become higher normalized values
                    P_ts_norm[:, i] = (col_max - col) / (col_max - col_min)
                else:
                    # For maximization: standard normalization
                    P_ts_norm[:, i] = (col - col_min) / (col_max - col_min)
            else:
                P_ts_norm[:, i] = 0.5  # Constant metric
        
        # Normalize task-agnostic metrics
        P_ta_norm = np.zeros_like(P_ta)
        for i in range(n_ta):
            col = P_ta[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 0:
                if ta_minimize[i]:
                    P_ta_norm[:, i] = (col_max - col) / (col_max - col_min)
                else:
                    P_ta_norm[:, i] = (col - col_min) / (col_max - col_min)
            else:
                P_ta_norm[:, i] = 0.5  # Constant metric
        
        return P_ts_norm, P_ta_norm
    
    def select_best(self, P_ts, P_ta, ts_minimize=None, ta_minimize=None):
        """
        Original dual-metric selection algorithm
        Returns: (best_checkpoint_idx, score)
        """
        # Normalize metrics
        P_ts_norm, P_ta_norm = self.normalize_metrics(P_ts, P_ta, ts_minimize, ta_minimize)
        
        n_epochs = P_ts_norm.shape[0]
        n_ts = P_ts_norm.shape[1]
        n_ta = P_ta_norm.shape[1]
        
        # Dictionary to store checkpoint -> score
        checkpoint_scores = {}
        
        # For each metric pair
        for i in range(n_ts):
            for j in range(n_ta):
                # Find best checkpoint for this pair
                combined = P_ts_norm[:, i] + P_ta_norm[:, j]
                best_idx = np.argmax(combined)
                
                # Calculate total score at this checkpoint
                total_score = P_ts_norm[best_idx, :].sum() + P_ta_norm[best_idx, :].sum()
                
                # Store (overwrites if checkpoint already exists)
                checkpoint_scores[best_idx] = total_score
        
        # Find checkpoint with highest score
        if not checkpoint_scores:
            return None, None
        
        best_idx = max(checkpoint_scores.items(), key=lambda x: x[1])[0]
        best_score = checkpoint_scores[best_idx]
        
        return best_idx, best_score


class CheckpointSelector:
    """Main class for checkpoint selection with bootstrap voting"""
    
    def __init__(self, data_dir):
        """Load all CSV files from extracted_trials directory"""
        self.data_dir = Path(data_dir)
        
        print(f"Loading data from: {self.data_dir}")
        
        # Load all CSVs
        self.classification_df = pd.read_csv(self.data_dir / 'classification_auc_trials.csv')
        self.segmentation_df = pd.read_csv(self.data_dir / 'segmentation_aji_trials.csv')
        self.rankme_df = pd.read_csv(self.data_dir / 'rankme_bootstrap_trials.csv')
        self.clid_df = pd.read_csv(self.data_dir / 'clid_bootstrap_trials.csv')
        self.alphareq_df = pd.read_csv(self.data_dir / 'alphareq_bootstrap_trials.csv')
        self.lidar_df = pd.read_csv(self.data_dir / 'lidar_bootstrap_trials.csv')
        
        # Extract unique checkpoints
        self.checkpoints = sorted(self.classification_df['checkpoint_iteration'].unique())
        print(f"Found {len(self.checkpoints)} checkpoints: {self.checkpoints}")
        
        # Extract unique datasets for classification and segmentation
        self.classification_datasets = sorted(self.classification_df['dataset'].unique())
        self.segmentation_datasets = sorted(self.segmentation_df['dataset'].unique())
        
        print(f"Classification datasets: {self.classification_datasets}")
        print(f"Segmentation datasets: {self.segmentation_datasets}")
        
        # Create metric names
        self.task_specific_metrics = []
        self.task_specific_metrics.extend([f"{ds}_AUC" for ds in self.classification_datasets])
        self.task_specific_metrics.extend([f"{ds}_AJI" for ds in self.segmentation_datasets])
        
        self.task_agnostic_metrics = ['RANKME', 'CLID', 'ALPHAREQ', 'LIDAR']
        
        print(f"\nTask-specific metrics: {self.task_specific_metrics}")
        print(f"Task-agnostic metrics: {self.task_agnostic_metrics}")
        
        # Metric type mapping for resampling
        self.metric_info = {}
        for ds in self.classification_datasets:
            self.metric_info[f"{ds}_AUC"] = ('classification', ds)
        for ds in self.segmentation_datasets:
            self.metric_info[f"{ds}_AJI"] = ('segmentation', ds)
    
    def resample_classification(self, checkpoint, dataset):
        """Resample from classification AUC trials"""
        mask = (self.classification_df['checkpoint_iteration'] == checkpoint) & \
               (self.classification_df['dataset'] == dataset)
        values = self.classification_df[mask]['test_auc'].values
        
        if len(values) > 0:
            resampled = np.random.choice(values, size=len(values), replace=True)
            return np.mean(resampled)
        return np.nan
    
    def resample_segmentation(self, checkpoint, dataset):
        """Resample from segmentation AJI trials"""
        mask = (self.segmentation_df['checkpoint_iteration'] == checkpoint) & \
               (self.segmentation_df['dataset'] == dataset)
        values = self.segmentation_df[mask]['aji_score'].values
        
        if len(values) > 0:
            resampled = np.random.choice(values, size=len(values), replace=True)
            return np.mean(resampled)
        return np.nan
    
    def resample_task_agnostic(self, checkpoint, metric):
        """Resample from task-agnostic bootstrap trials"""
        if metric == 'RANKME':
            df = self.rankme_df
            value_col = 'rankme_value'
        elif metric == 'CLID':
            df = self.clid_df
            value_col = 'clid_value'
        elif metric == 'ALPHAREQ':
            df = self.alphareq_df
            value_col = 'alphareq_value'
        elif metric == 'LIDAR':
            df = self.lidar_df
            value_col = 'lidar_value'
        else:
            return np.nan
        
        mask = df['checkpoint_iteration'] == checkpoint
        values = df[mask][value_col].values
        
        if len(values) > 0:
            resampled = np.random.choice(values, size=len(values), replace=True)
            return np.mean(resampled)
        return np.nan
    
    def resample_bootstrap_iteration(self, selected_ts_metrics, selected_ta_metrics):
        """Resample all metrics for one bootstrap iteration"""
        n_checkpoints = len(self.checkpoints)
        n_ts = len(selected_ts_metrics)
        n_ta = len(selected_ta_metrics)
        
        P_ts = np.zeros((n_checkpoints, n_ts))
        P_ta = np.zeros((n_checkpoints, n_ta))
        
        # Resample task-specific metrics
        for j, metric_name in enumerate(selected_ts_metrics):
            metric_type, dataset = self.metric_info[metric_name]
            
            for i, checkpoint in enumerate(self.checkpoints):
                if metric_type == 'classification':
                    P_ts[i, j] = self.resample_classification(checkpoint, dataset)
                else:  # segmentation
                    P_ts[i, j] = self.resample_segmentation(checkpoint, dataset)
        
        # Resample task-agnostic metrics
        for j, metric_name in enumerate(selected_ta_metrics):
            for i, checkpoint in enumerate(self.checkpoints):
                P_ta[i, j] = self.resample_task_agnostic(checkpoint, metric_name)
        
        return P_ts, P_ta
    
    def run_bootstrap_voting(self, selected_ts_metrics, selected_ta_metrics, 
                           ts_minimize=None, ta_minimize=None, n_bootstrap=5000):
        """Run bootstrap voting with dual-metric selection"""
        print(f"\nRunning bootstrap voting with {n_bootstrap} iterations...")
        print(f"Selected task-specific metrics: {selected_ts_metrics}")
        print(f"Selected task-agnostic metrics: {selected_ta_metrics}")
        
        # Initialize selector
        selector = DualMetricSelector()
        
        # Vote tracking
        votes = defaultdict(int)
        valid_iterations = 0
        
        # Progress reporting
        report_interval = n_bootstrap // 10 if n_bootstrap >= 10 else 1
        
        # Bootstrap iterations
        for iter_num in range(n_bootstrap):
            if (iter_num + 1) % report_interval == 0:
                print(f"  Progress: {iter_num + 1}/{n_bootstrap} iterations")
            
            # Resample metrics
            P_ts, P_ta = self.resample_bootstrap_iteration(selected_ts_metrics, selected_ta_metrics)
            
            # Check for valid data
            if np.isnan(P_ts).all() or np.isnan(P_ta).all():
                continue
            
            # Run dual-metric selection
            best_idx, best_score = selector.select_best(P_ts, P_ta, ts_minimize, ta_minimize)
            
            if best_idx is not None:
                best_checkpoint = self.checkpoints[best_idx]
                votes[best_checkpoint] += 1
                valid_iterations += 1
        
        # Calculate vote percentages
        vote_percentages = {cp: (count / valid_iterations * 100) if valid_iterations > 0 else 0 
                           for cp, count in votes.items()}
        
        # Sort by vote percentage
        sorted_votes = sorted(vote_percentages.items(), key=lambda x: x[1], reverse=True)
        
        results = {
            'votes': dict(votes),
            'vote_percentages': vote_percentages,
            'sorted_votes': sorted_votes,
            'valid_iterations': valid_iterations,
            'total_iterations': n_bootstrap,
            'selected_ts_metrics': selected_ts_metrics,
            'selected_ta_metrics': selected_ta_metrics
        }
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        print("\n" + "="*60)
        print("BOOTSTRAP VOTING RESULTS")
        print("="*60)
        print(f"Valid iterations: {results['valid_iterations']}/{results['total_iterations']}")
        print(f"Success rate: {results['valid_iterations']/results['total_iterations']*100:.1f}%")
        
        print(f"\nTop 10 Checkpoints by Vote Percentage:")
        print("-" * 50)
        print(f"{'Rank':<6} {'Checkpoint':<20} {'Votes':<10} {'Percentage':<10}")
        print("-" * 50)
        
        for i, (checkpoint, percentage) in enumerate(results['sorted_votes'][:10]):
            vote_count = results['votes'][checkpoint]
            print(f"{i+1:<6} {checkpoint:<20} {vote_count:<10} {percentage:>6.2f}%")
        
        if results['sorted_votes']:
            winner, winner_pct = results['sorted_votes'][0]
            print(f"\nWINNER: Checkpoint {winner} with {winner_pct:.2f}% of votes")
            
            if len(results['sorted_votes']) > 1:
                runner_up, runner_up_pct = results['sorted_votes'][1]
                print(f"Runner-up: Checkpoint {runner_up} with {runner_up_pct:.2f}%")
                print(f"Winning margin: {winner_pct - runner_up_pct:.2f} percentage points")
    
    def plot_results(self, results, save_path=None):
        """Create visualization of voting results"""
        plt.style.use('default')
        
        # Get top 10 checkpoints
        top_checkpoints = results['sorted_votes'][:10]
        if not top_checkpoints:
            print("No results to plot")
            return
        
        checkpoints, percentages = zip(*top_checkpoints)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(checkpoints)), percentages, alpha=0.7, color='steelblue')
        
        plt.xlabel('Checkpoint')
        plt.ylabel('Vote Percentage (%)')
        plt.title('Bootstrap Voting Results - Top 10 Checkpoints')
        plt.xticks(range(len(checkpoints)), checkpoints, rotation=45, ha='right')
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Checkpoint selection with bootstrap voting')
    parser.add_argument('--data_dir', type=str, default='/data1/vanderbc/nandas1/PostProc/benchmark_results/TCGA_Dinov2_ViT-B_run2/extracted_trials',
                       help='Directory containing extracted trial CSVs')
    parser.add_argument('--n_bootstrap', type=int, default=5000,
                       help='Number of bootstrap iterations')
    parser.add_argument('--plot', action='store_true',
                       help='Create visualization of results')
    parser.add_argument('--save_plot', type=str, default='/data1/vanderbc/nandas1/PostProc/benchmark_results/TCGA_Dinov2_ViT-B_run2/extracted_trials/bootstrap_voting_plot.png',
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = CheckpointSelector(args.data_dir)
    
    # Define metrics to use (can be customized)
    selected_ts_metrics = [
        'CRC_AUC', 
        'MHIST_AUC',
        'PCam_AUC',
        'BRACS_AUC',
        'MiDOG_AUC',
        'MonuSeg_AJI',
        'PanNuke_AJI'
    ]
    
    selected_ta_metrics = [
        'RANKME',
        'LIDAR', 
        'ALPHAREQ',
        'CLID'
    ]
    
    # Define which metrics to minimize (True) vs maximize (False)
    ts_minimize = [False] * len(selected_ts_metrics)  # All maximize
    ta_minimize = [False, False, True, False]  # ALPHAREQ minimize, others maximize
    
    # Run bootstrap voting
    results = selector.run_bootstrap_voting(
        selected_ts_metrics=selected_ts_metrics,
        selected_ta_metrics=selected_ta_metrics,
        ts_minimize=ts_minimize,
        ta_minimize=ta_minimize,
        n_bootstrap=args.n_bootstrap
    )
    
    # Print results
    selector.print_results(results)
    
    # Plot if requested
    if args.plot or args.save_plot:
        selector.plot_results(results, args.save_plot)


if __name__ == "__main__":
    main()
