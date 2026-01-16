"""
Main Pipeline Script
Orchestrates running multiple models and generating comparison reports
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json


def get_enabled_models(models_config: Dict) -> List[Dict]:
    """Extract all enabled models from config"""
    enabled = []
    
    for category in ['open_source_models', 'closed_source_models', 'baseline_models']:
        if category in models_config:
            for model in models_config[category]:
                if model.get('enabled', False):
                    enabled.append(model)
    
    return enabled


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run shell command and handle errors
    
    Args:
        cmd: Command as list of strings
        description: Description for logging
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running command: {e}")
        return False


def run_single_model(model_config: Dict, eval_config_path: str, 
                    models_config_path: str) -> Path:
    """
    Run reranking and evaluation for a single model
    
    Args:
        model_config: Model configuration
        eval_config_path: Path to evaluation config
        models_config_path: Path to models config
    
    Returns:
        Path to metrics file
    """
    model_name = model_config['name']
    
    # Run reranking
    cmd = [
        'python', 'scripts/run_rerank.py',
        '--model', model_name,
        '--config', eval_config_path,
        '--models_config', models_config_path
    ]
    
    success = run_command(cmd, f"Running reranking for: {model_name}")
    
    if not success:
        print(f"⚠️  Skipping evaluation for {model_name} due to reranking failure")
        return None
    
    # Run evaluation
    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    results_dir = Path(eval_config['output']['results_dir'])
    eval_file = results_dir / model_name / 'eval_data.json'
    
    if not eval_file.exists():
        print(f"⚠️  Eval file not found: {eval_file}")
        return None
    
    # Get k parameter from config
    k = eval_config.get('evaluation', {}).get('k', 10)
    
    cmd = [
        'python', 'scripts/evaluate.py',
        '--input', str(eval_file),
        '--k', str(k)
    ]
    
    success = run_command(cmd, f"Evaluating: {model_name}")
    
    if success:
        metrics_file = results_dir / model_name / 'metrics.json'
        return metrics_file
    
    return None


def generate_comparison_report(metrics_files: Dict[str, Path], output_file: Path):
    """
    Generate comparison report across all models
    
    Args:
        metrics_files: Dict mapping model names to metrics files
        output_file: Where to save comparison report
    """
    print(f"\n{'='*60}")
    print("Generating Comparison Report")
    print(f"{'='*60}\n")
    
    # Load all metrics
    all_metrics = {}
    for model_name, metrics_file in metrics_files.items():
        if metrics_file and metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                all_metrics[model_name] = data['metrics']
    
    if not all_metrics:
        print("⚠️  No metrics to compare")
        return
    
    # Get all metric names
    all_metric_names = set()
    for metrics in all_metrics.values():
        all_metric_names.update(metrics.keys())
    
    # Create comparison table
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'models': list(all_metrics.keys()),
        'metrics': {}
    }
    
    for metric_name in sorted(all_metric_names):
        comparison['metrics'][metric_name] = {}
        for model_name in all_metrics:
            value = all_metrics[model_name].get(metric_name, None)
            comparison['metrics'][metric_name][model_name] = value
    
    # Save as JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate text report
    text_file = output_file.with_suffix('.txt')
    with open(text_file, 'w') as f:
        f.write("LLM RANKER COMPARISON REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Generated: {comparison['timestamp']}\n")
        f.write(f"Models Compared: {len(comparison['models'])}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("RESULTS BY METRIC\n")
        f.write(f"{'='*80}\n\n")
        
        for metric_name in sorted(all_metric_names):
            f.write(f"\n{metric_name}\n")
            f.write("-" * 80 + "\n")
            
            # Get values and sort models by performance
            values = comparison['metrics'][metric_name]
            sorted_models = sorted(values.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
            
            for rank, (model_name, value) in enumerate(sorted_models, 1):
                if value is not None:
                    f.write(f"  #{rank:2d}  {model_name:30s}  {value:.4f}\n")
                else:
                    f.write(f"       {model_name:30s}  N/A\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Find best model for each metric
    for metric_name in sorted(all_metric_names):
        values = comparison['metrics'][metric_name]
        best_model = max(values.items(), key=lambda x: x[1] if x[1] is not None else -1)
        
        if best_model[1] is not None:
            print(f"{metric_name:25s}  Best: {best_model[0]:20s}  ({best_model[1]:.4f})")
    
    print(f"\n{'='*80}")
    print(f"✓ Comparison saved to: {output_file}")
    print(f"✓ Report saved to: {text_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run LLM reranking pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to evaluation config')
    parser.add_argument('--models_config', type=str, default='configs/models.yaml',
                       help='Path to models config')
    parser.add_argument('--model', type=str, default=None,
                       help='Run specific model only (default: all enabled models)')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip generating comparison report')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.models_config, 'r') as f:
        models_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    # Get models to run
    if args.model:
        # Run specific model
        all_models = get_enabled_models(models_config)
        models_to_run = [m for m in all_models if m['name'] == args.model]
        
        if not models_to_run:
            print(f"Error: Model '{args.model}' not found or not enabled")
            return
    else:
        # Run all enabled models
        models_to_run = get_enabled_models(models_config)
    
    if not models_to_run:
        print("No enabled models found in config")
        return
    
    print(f"\n{'='*80}")
    print("LLM RANKER EVALUATION PIPELINE")
    print(f"{'='*80}")
    print(f"\nModels to evaluate: {len(models_to_run)}")
    for model in models_to_run:
        print(f"  - {model['name']} ({model['type']})")
    print()
    
    # Run each model
    metrics_files = {}
    
    for i, model_config in enumerate(models_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {i}/{len(models_to_run)}: {model_config['name']}")
        print(f"{'#'*80}\n")
        
        metrics_file = run_single_model(
            model_config,
            args.config,
            args.models_config
        )
        
        if metrics_file:
            metrics_files[model_config['name']] = metrics_file
            print(f"\n✓ {model_config['name']} completed successfully")
        else:
            print(f"\n⚠️  {model_config['name']} failed or incomplete")
    
    # Generate comparison report
    if not args.skip_comparison and len(metrics_files) > 1:
        results_dir = Path(eval_config['output']['results_dir'])
        comparison_file = results_dir / 'comparison.json'
        
        generate_comparison_report(metrics_files, comparison_file)
    
    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nSuccessful: {len(metrics_files)}/{len(models_to_run)}")
    print(f"\nResults directory: {eval_config['output']['results_dir']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
