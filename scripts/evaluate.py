"""
Evaluation Script
Calculates ranking metrics from saved reranking results
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import (
    calculate_reranking_metrics,
    use_pytrec_eval,
    convert_scores_to_ranking
)


def load_eval_data(eval_file: Path) -> Dict:
    """Load evaluation data from JSON file"""
    with open(eval_file, 'r') as f:
        return json.load(f)


def calculate_metrics(eval_data: Dict, use_pytrec: bool = True, k: int = 10) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        eval_data: Evaluation data dictionary
        use_pytrec: Whether to use pytrec_eval (if available)
        k: Cutoff for @k metrics
    
    Returns:
        Dictionary of metric scores
    """
    rankings = eval_data['rankings']
    qrels = eval_data['qrels']
    
    # Try pytrec_eval first
    if use_pytrec:
        try:
            metrics = use_pytrec_eval(qrels, rankings, k=k)
            if metrics:
                # Rename to standard format
                renamed = {}
                for key, val in metrics.items():
                    if f'ndcg_cut_{k}' in key:
                        renamed[f'NDCG@{k}'] = val
                    elif f'map_cut_{k}' in key:
                        renamed[f'MAP@{k}'] = val
                    else:
                        renamed[key] = val
                return renamed
        except Exception as e:
            print(f"Warning: pytrec_eval failed ({e}), using fallback metrics")
    
    # Convert score dicts to ranked lists for fallback metrics
    ranked_lists = {
        qid: convert_scores_to_ranking(scores)
        for qid, scores in rankings.items()
    }
    
    # Calculate metrics using our implementation
    metrics = calculate_reranking_metrics(ranked_lists, qrels, k=k)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}\n")
    
    # Group metrics
    primary_metrics = {}
    precision_metrics = {}
    ndcg_metrics = {}
    
    for name, value in metrics.items():
        name_lower = name.lower()
        if 'map' in name_lower:
            primary_metrics[name] = value
        elif 'mrr' in name_lower or 'recip' in name_lower:
            primary_metrics[name] = value
        elif 'p_' in name_lower or 'p@' in name_lower:
            precision_metrics[name] = value
        elif 'ndcg' in name_lower:
            ndcg_metrics[name] = value
    
    # Print primary metrics
    if primary_metrics:
        print("Primary Metrics:")
        for name, value in primary_metrics.items():
            print(f"  {name:20s}: {value:.4f}")
        print()
    
    # Print NDCG metrics
    if ndcg_metrics:
        print("NDCG Metrics:")
        for name, value in ndcg_metrics.items():
            print(f"  {name:20s}: {value:.4f}")
        print()
    
    # Print Precision metrics
    if precision_metrics:
        print("Precision Metrics:")
        for name, value in precision_metrics.items():
            print(f"  {name:20s}: {value:.4f}")
        print()
    
    print(f"{'='*60}\n")


def save_metrics_report(metrics: Dict[str, float], eval_data: Dict, output_file: Path):
    """Save metrics to file"""
    report = {
        'model_name': eval_data.get('model_config', {}).get('name', 'unknown'),
        'model_id': eval_data.get('model_config', {}).get('model_id', 'unknown'),
        'dataset': eval_data.get('eval_config', {}).get('dataset', {}),
        'timestamp': eval_data.get('timestamp', datetime.now().isoformat()),
        'metrics': metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as text
    text_file = output_file.with_suffix('.txt')
    with open(text_file, 'w') as f:
        f.write(f"EVALUATION REPORT\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {report['model_name']}\n")
        f.write(f"Model ID: {report['model_id']}\n")
        f.write(f"Dataset: {report['dataset'].get('name', 'unknown')}\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"METRICS\n")
        f.write(f"{'='*60}\n\n")
        
        for name, value in metrics.items():
            f.write(f"{name:25s}: {value:.4f}\n")
    
    return text_file


def compare_with_baseline(eval_file: Path, baseline_file: Path) -> Dict:
    """Compare results with baseline"""
    eval_data = load_eval_data(eval_file)
    baseline_data = load_eval_data(baseline_file)
    
    eval_metrics = calculate_metrics(eval_data)
    baseline_metrics = calculate_metrics(baseline_data)
    
    comparison = {}
    for metric in eval_metrics:
        if metric in baseline_metrics:
            eval_val = eval_metrics[metric]
            baseline_val = baseline_metrics[metric]
            
            improvement = ((eval_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison[metric] = {
                'model': eval_val,
                'baseline': baseline_val,
                'improvement_%': improvement
            }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate reranking results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to eval_data.json file')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline eval_data.json for comparison')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save metrics report (default: same dir as input)')
    parser.add_argument('--no_pytrec', action='store_true',
                       help='Disable pytrec_eval, use fallback metrics')
    parser.add_argument('--k', type=int, default=10,
                       help='Cutoff for @k metrics (default: 10)')
    
    args = parser.parse_args()
    
    # Load data
    eval_file = Path(args.input)
    if not eval_file.exists():
        print(f"Error: Input file not found: {eval_file}")
        return
    
    print(f"Loading evaluation data from: {eval_file}")
    eval_data = load_eval_data(eval_file)
    
    # Calculate metrics
    print(f"Calculating metrics (k={args.k})...")
    metrics = calculate_metrics(eval_data, use_pytrec=not args.no_pytrec, k=args.k)
    
    # Print results
    model_name = eval_data.get('model_config', {}).get('name', 'Unknown')
    print_metrics(metrics, model_name)
    
    # Save report
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = eval_file.parent / 'metrics.json'
    
    text_file = save_metrics_report(metrics, eval_data, output_file)
    print(f"✓ Metrics saved to: {output_file}")
    print(f"✓ Report saved to: {text_file}")
    
    # Compare with baseline if provided
    if args.baseline:
        baseline_file = Path(args.baseline)
        if baseline_file.exists():
            print(f"\nComparing with baseline: {baseline_file}")
            comparison = compare_with_baseline(eval_file, baseline_file)
            
            print(f"\n{'='*60}")
            print("COMPARISON WITH BASELINE")
            print(f"{'='*60}\n")
            
            for metric, values in comparison.items():
                print(f"{metric:20s}")
                print(f"  Model:       {values['model']:.4f}")
                print(f"  Baseline:    {values['baseline']:.4f}")
                print(f"  Improvement: {values['improvement_%']:+.2f}%")
                print()
            
            # Save comparison
            comparison_file = output_file.parent / 'comparison.json'
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"✓ Comparison saved to: {comparison_file}")
        else:
            print(f"Warning: Baseline file not found: {baseline_file}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
