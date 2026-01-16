"""
Main Reranking Inference Script
Performs LLM-based reranking and saves results to disk
"""

import os
import re
import json
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from datetime import datetime

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import get_data_loader
from scripts.prompts import create_rankgpt_messages


class ModelInference:
    """Handles LLM inference for different model types"""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.model_type = model_config['type']
        self.model = None
        
        if self.model_type == 'vllm':
            self._init_vllm()
        elif self.model_type == 'openai':
            self._init_openai()
        elif self.model_type == 'anthropic':
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _init_vllm(self):
        """Initialize vLLM model"""
        from vllm import LLM, SamplingParams
        
        self.model = LLM(
            model=self.config['model_id'],
            tensor_parallel_size=self.config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=self.config.get('gpu_memory_utilization', 0.85),
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.config.get('temperature', 0.0),
            max_tokens=self.config.get('max_tokens', 300),
            top_p=0.95
        )
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        
        api_key = os.environ.get(self.config.get('api_key_env', 'OPENAI_API_KEY'))
        if not api_key:
            raise ValueError(f"API key not found: {self.config.get('api_key_env')}")
        
        self.model = OpenAI(api_key=api_key)
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        from anthropic import Anthropic
        
        api_key = os.environ.get(self.config.get('api_key_env', 'ANTHROPIC_API_KEY'))
        if not api_key:
            raise ValueError(f"API key not found: {self.config.get('api_key_env')}")
        
        self.model = Anthropic(api_key=api_key)
    
    def generate(self, messages: List[Dict]) -> str:
        """
        Generate response from model
        
        Args:
            messages: List of chat messages
        
        Returns:
            Generated text
        """
        if self.model_type == 'vllm':
            return self._generate_vllm(messages)
        elif self.model_type == 'openai':
            return self._generate_openai(messages)
        elif self.model_type == 'anthropic':
            return self._generate_anthropic(messages)
    
    def _generate_vllm(self, messages: List[Dict]) -> str:
        """Generate using vLLM"""
        outputs = self.model.chat([messages], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def _generate_openai(self, messages: List[Dict]) -> str:
        """Generate using OpenAI API"""
        response = self.model.chat.completions.create(
            model=self.config['model_id'],
            messages=messages,
            temperature=self.config.get('temperature', 0.0),
            max_tokens=self.config.get('max_tokens', 300)
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, messages: List[Dict]) -> str:
        """Generate using Anthropic API"""
        # Separate system messages
        system = ' '.join([m['content'] for m in messages if m['role'] == 'system'])
        user_messages = [m for m in messages if m['role'] != 'system']
        
        response = self.model.messages.create(
            model=self.config['model_id'],
            system=system if system else None,
            messages=user_messages,
            max_tokens=self.config.get('max_tokens', 300),
            temperature=self.config.get('temperature', 0.0)
        )
        return response.content[0].text
    
    def batch_generate(self, all_messages: List[List[Dict]]) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            all_messages: List of message lists
        
        Returns:
            List of generated texts
        """
        if self.model_type == 'vllm':
            # Batch processing for vLLM
            outputs = self.model.chat(all_messages, sampling_params=self.sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            # Sequential for API-based models
            results = []
            for messages in tqdm(all_messages, desc="Generating", leave=False):
                result = self.generate(messages)
                results.append(result)
            return results


def parse_ranking_output(text: str, num_passages: int) -> List[int]:
    """
    Parse LLM ranking output to extract ranked indices
    
    Args:
        text: Raw LLM output (e.g., "[2] > [0] > [1]")
        num_passages: Total number of passages
    
    Returns:
        List of 0-indexed passage positions
    """
    # Extract all numbers in brackets or standalone
    matches = re.findall(r'\[?(\d+)\]?', text)
    
    if not matches:
        # Fallback to original order
        return list(range(num_passages))
    
    # Convert to integers
    ranking = []
    for m in matches:
        try:
            idx = int(m)
            if 0 <= idx < num_passages and idx not in ranking:
                ranking.append(idx)
        except ValueError:
            continue
    
    # Add missing indices at the end
    for i in range(num_passages):
        if i not in ranking:
            ranking.append(i)
    
    return ranking


def run_reranking(model_config: Dict, eval_config: Dict, output_dir: Path):
    """
    Main reranking function
    
    Args:
        model_config: Model configuration dictionary
        eval_config: Evaluation configuration dictionary
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"Running reranking with: {model_config['name']}")
    print(f"{'='*60}\n")
    
    # Load data
    print("[1/4] Loading dataset...")
    dataset_config = eval_config['dataset']
    loader = get_data_loader(
        dataset_config['name'],
        cache_dir=dataset_config.get('cache_dir')
    )
    
    data = loader.load(
        split=dataset_config.get('split', 'validation'),
        num_queries=dataset_config.get('num_queries'),
        num_passages=dataset_config.get('num_passages', 10)
    )
    
    print(f"Loaded {len(data)} queries\n")
    
    # Initialize model
    print("[2/4] Initializing model...")
    model = ModelInference(model_config)
    print("Model ready\n")
    
    # Generate prompts
    print("[3/4] Creating prompts...")
    
    all_prompts = []
    for item in tqdm(data, desc="Building prompts"):
        passages = [p['text'] for p in item['passages']]
        prompts = create_rankgpt_messages(item['query'], passages)
        all_prompts.append(prompts)
    
    print(f"Created {len(all_prompts)} prompts\n")
    
    # Run inference
    print("[4/4] Running inference...")
    outputs = model.batch_generate(all_prompts)
    
    # Process results
    print("\nProcessing results...")
    results = []
    all_rankings = {}  # For evaluation
    all_qrels = {}
    
    for i, (item, output_text) in enumerate(zip(data, outputs)):
        qid = item['qid']
        num_passages = len(item['passages'])
        
        # Parse ranking
        ranking_indices = parse_ranking_output(output_text, num_passages)
        
        # Reorder passages
        ranked_passages = [item['passages'][idx] for idx in ranking_indices]
        
        # Store ranking for evaluation (doc_id -> score)
        all_rankings[qid] = {
            p['pid']: float(num_passages - i)
            for i, p in enumerate(ranked_passages)
        }
        
        # Store qrels
        all_qrels[qid] = item['qrels']
        
        # Save detailed result
        result = {
            'qid': qid,
            'query': item['query'],
            'raw_output': output_text,
            'ranking_indices': ranking_indices,
            'ranked_passage_ids': [p['pid'] for p in ranked_passages]
        }
        
        if eval_config.get('evaluation', {}).get('save_raw_outputs', True):
            result['passages'] = ranked_passages
        
        results.append(result)
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / 'rankings.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save ranking and qrels for evaluation
    eval_data = {
        'rankings': all_rankings,
        'qrels': all_qrels,
        'model_config': model_config,
        'eval_config': eval_config,
        'timestamp': datetime.now().isoformat()
    }
    
    eval_file = output_dir / 'eval_data.json'
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ Eval data saved to: {eval_file}")
    
    return eval_file


def main():
    parser = argparse.ArgumentParser(description='Run LLM-based reranking')
    parser.add_argument('--model', type=str, help='Model name from config')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to evaluation config')
    parser.add_argument('--models_config', type=str, default='configs/models.yaml',
                       help='Path to models config')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    with open(args.models_config, 'r') as f:
        models_config = yaml.safe_load(f)
    
    # Find model config
    model_config = None
    for category in ['open_source_models', 'closed_source_models', 'baseline_models']:
        if category in models_config:
            for model in models_config[category]:
                if model['name'] == args.model:
                    model_config = model
                    break
        if model_config:
            break
    
    if not model_config:
        print(f"Error: Model '{args.model}' not found in {args.models_config}")
        return
    
    if not model_config.get('enabled', False):
        print(f"Warning: Model '{args.model}' is disabled in config")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(eval_config['output']['results_dir']) / model_config['name']
    
    # Run reranking
    eval_file = run_reranking(model_config, eval_config, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Reranking complete!")
    print(f"Run evaluation with: python scripts/evaluate.py --input {eval_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
