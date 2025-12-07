import os
import json
import argparse
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from train import train, MODELS

# Copmrehensible run
# python grid_search.py \
#   --models BasicCNN DeeperCNN WiderCNN PyramidCNN \
#   --learning_rates 1e-4 1e-5 \
#   --batch_sizes 16 32 64 128 \
#   --optimizers adamw sgd \
#   --loss_functions mse mae smoothl1 \
#   --epochs 100

def run_grid_search(
    search_space: Dict[str, List[Any]],
    dataset_path: str,
    training_dir: str,
    augmented_dir: str,
    max_epochs: int = 50,
    val_split: float = 0.2,
    early_stop_patience: int = 10,
    results_dir: str = "grid_search_results",
) -> pd.DataFrame:

    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    total_runs = len(all_combinations)
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(f"Total configurations to test: {total_runs}")
    print(f"Search space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    print("=" * 80)
    print()
    
    results = []
    
    for idx, combination in enumerate(all_combinations, 1):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))
        
        print("\n" + "=" * 80)
        print(f"RUN {idx}/{total_runs}")
        print("=" * 80)
        print("Configuration:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print("=" * 80)
        
        try:
            # Run training with this configuration
            result = train(
                model_name=params.get('model', 'BasicCNN'),
                dataset_path=dataset_path,
                training_dir=training_dir,
                augmented_dir=augmented_dir,
                epochs=max_epochs,
                batch_size=params.get('batch_size', 32),
                lr=params.get('learning_rate', 1e-3),
                optimizer_name=params.get('optimizer', 'adam'),
                loss_fn_name=params.get('loss_function', 'mse'),
                val_split=val_split,
                early_stop_patience=early_stop_patience,
                plot_results=False,
                auto_visualize=False,
            )
            
            # Store results
            run_result = {
                'run': idx,
                **params,
                'best_val_loss': result['best_val_loss'],
                'final_epoch': result['final_epoch'],
                'final_train_loss': result['train_losses'][-1],
                'final_val_loss': result['val_losses'][-1],
                'status': 'completed',
            }
            
        except Exception as e:
            print(f"ERROR in run {idx}: {str(e)}")
            run_result = {
                'run': idx,
                **params,
                'best_val_loss': float('inf'),
                'final_epoch': 0,
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'status': f'failed: {str(e)[:100]}',
            }
        
        results.append(run_result)
        
        # Save intermediate results after each run
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(results_dir, f"grid_search_results_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"Run {idx} completed. Best val loss so far: {min([r['best_val_loss'] for r in results]):.4f}")
        print(f"Intermediate results saved to: {csv_path}")
        print(f"{'=' * 80}\n")
    
    # Final results summary
    df_results = pd.DataFrame(results)
    
    # Sort by best validation loss
    df_results_sorted = df_results.sort_values('best_val_loss')
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {len(df_results[df_results['status'] == 'completed'])}")
    print(f"Failed runs: {len(df_results[df_results['status'] != 'completed'])}")
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS:")
    print("=" * 80)
    
    # Display top 5 results
    for idx, row in df_results_sorted.head(5).iterrows():
        print(f"\nRank {list(df_results_sorted.index).index(idx) + 1}:")
        print(f"  Best Val Loss: {row['best_val_loss']:.4f}")
        for param in param_names:
            print(f"  {param}: {row[param]}")
        print(f"  Final Epoch: {row['final_epoch']}")
    
    # Save final results
    final_csv = os.path.join(results_dir, "grid_search_final_results.csv")
    df_results_sorted.to_csv(final_csv, index=False)
    print(f"\n{'=' * 80}")
    print(f"Final results saved to: {final_csv}")
    
    # Save best configuration as JSON
    best_config = df_results_sorted.iloc[0]
    best_config_dict = {param: best_config[param] for param in param_names}
    best_config_dict['best_val_loss'] = float(best_config['best_val_loss'])
    best_config_dict['final_epoch'] = int(best_config['final_epoch'])
    
    json_path = os.path.join(results_dir, "best_config.json")
    with open(json_path, 'w') as f:
        json.dump(best_config_dict, indent=2, fp=f)
    
    print(f"Best configuration saved to: {json_path}")
    print("=" * 80)
    
    return df_results_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for hyperparameter tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "-d", "--dataset",
        default="train_final.csv",
        help="Path to training CSV file"
    )
    parser.add_argument(
        "-td", "--training_dir",
        default="training",
        help="Directory containing original training images"
    )
    parser.add_argument(
        "-ad", "--augmented_dir",
        default="augmented",
        help="Directory containing augmented training images"
    )
    
    # Grid search configuration
    parser.add_argument(
        "--models",
        nargs='+',
        default=["BasicCNN"],
        choices=list(MODELS.keys()),
        help="Models to search over"
    )
    parser.add_argument(
        "--learning_rates",
        nargs='+',
        type=float,
        default=[1e-3, 5e-4, 1e-4],
        help="Learning rates to search over"
    )
    parser.add_argument(
        "--batch_sizes",
        nargs='+',
        type=int,
        default=[32, 64],
        help="Batch sizes to search over"
    )
    parser.add_argument(
        "--optimizers",
        nargs='+',
        default=["adam"],
        choices=["adam", "sgd", "adamw"],
        help="Optimizers to search over"
    )
    parser.add_argument(
        "--loss_functions",
        nargs='+',
        default=["mse"],
        choices=["mse", "mae", "smoothl1"],
        help="Loss functions to search over"
    )
    
    # Training configuration
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=50,
        help="Maximum epochs per configuration"
    )
    parser.add_argument(
        "-vs", "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "-es", "--early_stop",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    
    # Output
    parser.add_argument(
        "-o", "--output_dir",
        default="grid_search_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.training_dir):
        raise ValueError(f"Training directory not found: {args.training_dir}")
    if not os.path.exists(args.augmented_dir):
        raise ValueError(f"Augmented directory not found: {args.augmented_dir}")
    
    # Build search space
    search_space = {
        'model': args.models,
        'learning_rate': args.learning_rates,
        'batch_size': args.batch_sizes,
        'optimizer': args.optimizers,
        'loss_function': args.loss_functions,
    }
    
    # Run grid search
    results = run_grid_search(
        search_space=search_space,
        dataset_path=args.dataset,
        training_dir=args.training_dir,
        augmented_dir=args.augmented_dir,
        max_epochs=args.epochs,
        val_split=args.val_split,
        early_stop_patience=args.early_stop,
        results_dir=args.output_dir,
    )
    
    print("\nGrid search completed successfully!")


if __name__ == "__main__":
    main()
