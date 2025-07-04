#!/usr/bin/env python3
"""
Analyze W&B sweep results and extract best hyperparameters
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
from pathlib import Path
import numpy as np


def get_sweep_results(project_name, sweep_id):
    """Fetch sweep results from W&B"""
    api = wandb.Api()
    
    # Get sweep
    sweep = api.sweep(f"{api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Get all runs
    runs = sweep.runs
    
    # Extract run data
    run_data = []
    for run in runs:
        if run.state == "finished":
            config = run.config
            summary = run.summary._json_dict
            
            # Extract key metrics
            run_info = {
                'run_id': run.id,
                'run_name': run.name,
                'val_hand_mpjpe': summary.get('val/hand_mpjpe', float('inf')),
                'best_val_mpjpe': summary.get('val/best_mpjpe', float('inf')),
                'final_train_loss': summary.get('train/loss', float('inf')),
                'epochs_trained': summary.get('epoch', 0),
            }
            
            # Add all config parameters
            run_info.update(config)
            run_data.append(run_info)
    
    return pd.DataFrame(run_data)


def analyze_results(df, output_dir):
    """Analyze sweep results and create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sort by best validation MPJPE
    df_sorted = df.sort_values('best_val_mpjpe')
    
    print("\n=== TOP 10 RUNS ===")
    print(df_sorted[['run_name', 'best_val_mpjpe', 'learning_rate', 'batch_size', 
                     'hidden_dim', 'dropout', 'num_refinement_steps']].head(10))
    
    # Save best config
    best_run = df_sorted.iloc[0]
    best_config = {k: v for k, v in best_run.items() 
                   if k not in ['run_id', 'run_name', 'val_hand_mpjpe', 
                               'best_val_mpjpe', 'final_train_loss', 'epochs_trained']}
    
    with open(output_dir / 'best_config.yaml', 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"\nBest configuration saved to: {output_dir / 'best_config.yaml'}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Learning rate vs MPJPE
    ax = axes[0, 0]
    ax.scatter(df['learning_rate'], df['best_val_mpjpe'], alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Val MPJPE (mm)')
    ax.set_title('Learning Rate vs Performance')
    ax.grid(True, alpha=0.3)
    
    # 2. Batch size vs MPJPE
    ax = axes[0, 1]
    batch_sizes = df['batch_size'].unique()
    mpjpe_by_batch = [df[df['batch_size'] == bs]['best_val_mpjpe'].values 
                      for bs in sorted(batch_sizes)]
    ax.boxplot(mpjpe_by_batch, labels=sorted(batch_sizes))
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Best Val MPJPE (mm)')
    ax.set_title('Batch Size vs Performance')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Hidden dim vs MPJPE
    ax = axes[0, 2]
    hidden_dims = df['hidden_dim'].unique()
    mpjpe_by_hidden = [df[df['hidden_dim'] == hd]['best_val_mpjpe'].values 
                       for hd in sorted(hidden_dims)]
    ax.boxplot(mpjpe_by_hidden, labels=sorted(hidden_dims))
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Best Val MPJPE (mm)')
    ax.set_title('Hidden Dimension vs Performance')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Loss weight analysis - Hand losses
    ax = axes[1, 0]
    ax.scatter(df['loss_weight_hand_refined'], df['best_val_mpjpe'], 
               c=df['loss_weight_hand_coarse'], cmap='viridis', alpha=0.6)
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Coarse Hand Weight')
    ax.set_xlabel('Refined Hand Weight')
    ax.set_ylabel('Best Val MPJPE (mm)')
    ax.set_title('Hand Loss Weights vs Performance')
    ax.grid(True, alpha=0.3)
    
    # 5. Dropout vs MPJPE
    ax = axes[1, 1]
    dropout_rates = df['dropout'].unique()
    mpjpe_by_dropout = [df[df['dropout'] == dr]['best_val_mpjpe'].values 
                        for dr in sorted(dropout_rates)]
    ax.boxplot(mpjpe_by_dropout, labels=sorted(dropout_rates))
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Best Val MPJPE (mm)')
    ax.set_title('Dropout vs Performance')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 6. Parameter importance (correlation with performance)
    ax = axes[1, 2]
    
    # Select numeric columns for correlation
    numeric_cols = [col for col in df.columns 
                   if col not in ['run_id', 'run_name'] 
                   and df[col].dtype in ['float64', 'int64']]
    
    # Calculate correlations with best_val_mpjpe
    correlations = {}
    for col in numeric_cols:
        if col != 'best_val_mpjpe' and not df[col].isna().all():
            corr = df[[col, 'best_val_mpjpe']].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    # Sort and plot top 10
    top_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    params, importance = zip(*top_params)
    
    y_pos = np.arange(len(params))
    ax.barh(y_pos, importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel('Absolute Correlation with Val MPJPE')
    ax.set_title('Top 10 Most Important Parameters')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create parameter importance heatmap
    plt.figure(figsize=(12, 10))
    
    # Select top 20 runs
    top_runs = df_sorted.head(20)
    
    # Select important numeric parameters
    important_params = ['learning_rate', 'batch_size', 'hidden_dim', 'dropout',
                       'num_refinement_steps', 'freeze_layers',
                       'loss_weight_hand_coarse', 'loss_weight_hand_refined',
                       'loss_weight_contact', 'loss_weight_diversity']
    
    # Filter available parameters
    available_params = [p for p in important_params if p in top_runs.columns]
    
    # Normalize parameters for visualization
    param_data = top_runs[available_params].copy()
    for col in param_data.columns:
        if param_data[col].std() > 0:
            param_data[col] = (param_data[col] - param_data[col].mean()) / param_data[col].std()
    
    # Create heatmap
    sns.heatmap(param_data.T, 
                xticklabels=[f"Run {i+1}" for i in range(len(top_runs))],
                yticklabels=available_params,
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Normalized Value'})
    
    plt.title('Top 20 Runs - Parameter Values (Normalized)')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_run


def create_training_script(best_config, output_dir):
    """Create a training script with the best hyperparameters"""
    output_dir = Path(output_dir)
    
    script_content = f"""#!/usr/bin/env python3
# Auto-generated training script with best hyperparameters from sweep

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_advanced import main as train_main
from omegaconf import OmegaConf

# Best hyperparameters from sweep
best_config = {best_config}

# Load base config and update with best parameters
config = OmegaConf.load('configs/default_config.yaml')

# Update configuration
config.training.learning_rate = best_config.get('learning_rate', config.training.learning_rate)
config.training.batch_size = best_config.get('batch_size', config.training.batch_size)
config.model.hidden_dim = best_config.get('hidden_dim', config.model.hidden_dim)
config.model.dropout = best_config.get('dropout', config.model.dropout)
# ... add more parameter updates as needed

# Run training
train_main(config)
"""
    
    script_path = output_dir / 'train_best_config.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\nTraining script created: {script_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze W&B sweep results')
    parser.add_argument('--project', type=str, required=True, help='W&B project name')
    parser.add_argument('--sweep-id', type=str, required=True, help='W&B sweep ID')
    parser.add_argument('--output-dir', type=str, default='sweep_analysis', 
                       help='Output directory for analysis results')
    parser.add_argument('--top-n', type=int, default=10, 
                       help='Number of top runs to analyze in detail')
    
    args = parser.parse_args()
    
    print(f"Fetching sweep results from project: {args.project}, sweep: {args.sweep_id}")
    
    # Get sweep results
    df = get_sweep_results(args.project, args.sweep_id)
    
    if df.empty:
        print("No finished runs found in sweep!")
        return
    
    print(f"Found {len(df)} finished runs")
    
    # Analyze results
    best_run = analyze_results(df, args.output_dir)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Best MPJPE: {df['best_val_mpjpe'].min():.2f} mm")
    print(f"Median MPJPE: {df['best_val_mpjpe'].median():.2f} mm")
    print(f"Worst MPJPE: {df['best_val_mpjpe'].max():.2f} mm")
    print(f"Std Dev: {df['best_val_mpjpe'].std():.2f} mm")
    
    # Save full results
    df.to_csv(Path(args.output_dir) / 'sweep_results.csv', index=False)
    print(f"\nFull results saved to: {Path(args.output_dir) / 'sweep_results.csv'}")


if __name__ == "__main__":
    main()