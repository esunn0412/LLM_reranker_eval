"""
Custom MS MARCO Reranker Evaluation Script
Evaluates reranker performance on top-10 passages from MS MARCO
"""

import os
import json
import copy
from tqdm import tqdm
from datasets import load_dataset

# Set cache directories
os.environ['HF_HOME'] = '/local/scratch/tkim462/.cache/huggingface'
os.environ['XDG_CACHE_HOME'] = '/local/scratch/tkim462/.cache'

# Configuration
NUM_PASSAGES = 10  # Number of passages to rerank
NUM_QUERIES = 100  # Number of queries to evaluate (set to None for all)
MODEL_NAME = 'gpt-3.5-turbo'  # LLM model for reranking


def create_permutation_instruction(query, passages):
    """
    Create prompt for LLM to rank passages
    
    Args:
        query: The search query string
        passages: List of passage texts (up to 10)
    
    Returns:
        List of messages for LLM chat completion
    """
    num = len(passages)
    
    # Build passage list for prompt
    passage_text = "\n".join([f"[{i+1}] {passage}" for i, passage in enumerate(passages)])
    
    messages = [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        },
        {
            "role": "user",
            "content": f"""I will provide you with {num} passages, each indicated by number identifier []. 
Rank the passages based on their relevance to query: {query}

{passage_text}

Search Query: {query}.

Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."""
        }
    ]
    
    return messages


def call_llm(messages, model_name='gpt-3.5-turbo', api_key=None):
    """
    Call LLM API to get ranking
    
    Args:
        messages: Chat messages
        model_name: Model to use
        api_key: API key
    
    Returns:
        String response with ranking (e.g., "[3] > [1] > [2]")
    """
    if 'gpt' in model_name.lower():
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content
    
    elif 'claude' in model_name.lower():
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        system = ' '.join([turn['content'] for turn in messages if turn['role'] == 'system'])
        user_messages = [turn for turn in messages if turn['role'] != 'system']
        
        response = client.messages.create(
            model=model_name,
            system=system,
            messages=user_messages,
            max_tokens=300
        )
        return response.content[0].text
    
    else:
        # Use litellm for other models
        from litellm import completion
        response = completion(api_key=api_key, model=model_name, messages=messages)
        return response.choices[0].message.content


def parse_ranking(ranking_str, num_passages):
    """
    Parse LLM ranking output into list of indices
    
    Args:
        ranking_str: String like "[3] > [1] > [2]"
        num_passages: Total number of passages
    
    Returns:
        List of 0-indexed positions, e.g., [2, 0, 1] for "[3] > [1] > [2]"
    """
    import re
    
    # Extract numbers from brackets
    matches = re.findall(r'\[(\d+)\]', ranking_str)
    
    if not matches:
        print(f"Warning: Could not parse ranking '{ranking_str}', using original order")
        return list(range(num_passages))
    
    # Convert to 0-indexed
    ranking = [int(m) - 1 for m in matches]
    
    # Remove duplicates while preserving order
    seen = set()
    ranking = [x for x in ranking if not (x in seen or seen.add(x))]
    
    # Filter invalid indices
    ranking = [x for x in ranking if 0 <= x < num_passages]
    
    # Add missing indices at the end
    all_indices = set(range(num_passages))
    missing = [x for x in range(num_passages) if x not in ranking]
    ranking.extend(missing)
    
    return ranking


def rerank_passages(query, passages, model_name='gpt-3.5-turbo', api_key=None):
    """
    Rerank passages using LLM
    
    Args:
        query: Search query
        passages: List of passage dictionaries with 'text' and 'pid' keys
        model_name: LLM model name
        api_key: API key
    
    Returns:
        List of reranked passage dictionaries
    """
    # Extract just the text for ranking
    passage_texts = [p['text'] for p in passages]
    
    # Get ranking from LLM
    messages = create_permutation_instruction(query, passage_texts)
    ranking_str = call_llm(messages, model_name=model_name, api_key=api_key)
    
    # Parse ranking
    ranking = parse_ranking(ranking_str, len(passages))
    
    # Reorder passages
    reranked = [passages[i] for i in ranking]
    
    return reranked, ranking_str


def load_msmarco_data(num_queries=None):
    """
    Load MS MARCO dataset
    
    Args:
        num_queries: Number of queries to load (None for all)
    
    Returns:
        Dictionary with queries, passages, and qrels
    """
    print("Loading MS MARCO dataset...")
    
    # Load dataset
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    
    if num_queries:
        dataset = dataset.select(range(min(num_queries, len(dataset))))
    
    print(f"Loaded {len(dataset)} queries")
    
    # Process into query-passage-qrel format
    data = []
    for item in tqdm(dataset, desc="Processing queries"):
        query_id = str(item['query_id'])
        query = item['query']
        passages = item['passages']
        
        # Get top-10 passages
        passage_list = []
        qrels = {}
        
        for i, passage in enumerate(passages[:NUM_PASSAGES]):
            pid = f"{query_id}_p{i}"
            passage_list.append({
                'pid': pid,
                'text': passage['passage_text'],
                'is_selected': passage['is_selected']
            })
            
            # Qrels: 1 if selected (relevant), 0 otherwise
            qrels[pid] = 1 if passage['is_selected'] else 0
        
        data.append({
            'qid': query_id,
            'query': query,
            'passages': passage_list,
            'qrels': qrels
        })
    
    return data


def calculate_metrics(all_qrels, all_rankings):
    """
    Calculate ranking metrics
    
    Args:
        all_qrels: Dict of {qid: {docid: relevance}}
        all_rankings: Dict of {qid: {docid: score}}
    
    Returns:
        Dictionary of metric scores
    """
    try:
        import pytrec_eval
        
        evaluator = pytrec_eval.RelevanceEvaluator(
            all_qrels, 
            {'ndcg_cut.10', 'map_cut.10', 'recip_rank', 'P.1', 'P.5', 'P.10'}
        )
        
        results = evaluator.evaluate(all_rankings)
        
        # Average across queries
        metrics = {}
        for measure in ['ndcg_cut_10', 'map_cut_10', 'recip_rank', 'P_1', 'P_5', 'P_10']:
            scores = [query_measures[measure] for query_measures in results.values()]
            metrics[measure] = sum(scores) / len(scores) if scores else 0.0
        
        return metrics
    
    except ImportError:
        print("pytrec_eval not installed. Installing simple metrics...")
        # Simple MRR and Precision@k calculation
        metrics = {'mrr': 0.0, 'p@1': 0.0, 'p@5': 0.0, 'p@10': 0.0}
        
        for qid in all_qrels:
            qrel = all_qrels[qid]
            ranking = all_rankings.get(qid, {})
            
            # Sort by score
            ranked_docs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
            
            # MRR - position of first relevant doc
            for i, (docid, _) in enumerate(ranked_docs):
                if qrel.get(docid, 0) > 0:
                    metrics['mrr'] += 1.0 / (i + 1)
                    break
            
            # Precision at k
            for k in [1, 5, 10]:
                top_k = ranked_docs[:k]
                relevant_in_k = sum(1 for docid, _ in top_k if qrel.get(docid, 0) > 0)
                metrics[f'p@{k}'] += relevant_in_k / k
        
        # Average
        num_queries = len(all_qrels)
        for key in metrics:
            metrics[key] /= num_queries
        
        return metrics


def main():
    print('=' * 60)
    print('MS MARCO Reranker Evaluation')
    print(f'Model: {MODEL_NAME}')
    print(f'Evaluating top-{NUM_PASSAGES} passages per query')
    print('=' * 60)
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if not api_key and 'gpt' in MODEL_NAME.lower():
        print("\nWARNING: OPENAI_API_KEY not found!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Load MS MARCO data
    print("\n[Step 1/3] Loading MS MARCO dataset...")
    data = load_msmarco_data(num_queries=NUM_QUERIES)
    print(f"Loaded {len(data)} queries")
    
    # Step 2: Rerank with LLM
    print(f"\n[Step 2/3] Reranking with {MODEL_NAME}...")
    
    # Store original and reranked results
    original_qrels = {}  # {qid: {docid: relevance}}
    original_rankings = {}  # {qid: {docid: score}}
    reranked_rankings = {}  # {qid: {docid: score}}
    
    for item in tqdm(data, desc="Reranking queries"):
        qid = item['qid']
        query = item['query']
        passages = item['passages']
        qrels = item['qrels']
        
        # Original BM25-like ranking (by position)
        original_rankings[qid] = {
            p['pid']: float(len(passages) - i) for i, p in enumerate(passages)
        }
        
        # Rerank with LLM
        try:
            reranked_passages, ranking_str = rerank_passages(
                query, passages, model_name=MODEL_NAME, api_key=api_key
            )
            
            # Store reranked scores
            reranked_rankings[qid] = {
                p['pid']: float(len(reranked_passages) - i) 
                for i, p in enumerate(reranked_passages)
            }
            
        except Exception as e:
            print(f"\nError reranking query {qid}: {e}")
            # Fall back to original ranking
            reranked_rankings[qid] = original_rankings[qid]
        
        # Store qrels
        original_qrels[qid] = qrels
    
    # Step 3: Calculate metrics
    print("\n[Step 3/3] Calculating metrics...")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nOriginal Ranking (BM25-like):")
    original_metrics = calculate_metrics(original_qrels, original_rankings)
    for metric, score in original_metrics.items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    print("\nReranked with LLM:")
    reranked_metrics = calculate_metrics(original_qrels, reranked_rankings)
    for metric, score in reranked_metrics.items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    print("\nImprovement:")
    for metric in original_metrics:
        original = original_metrics[metric]
        reranked = reranked_metrics[metric]
        improvement = ((reranked - original) / original * 100) if original > 0 else 0
        print(f"  {metric.upper()}: {improvement:+.2f}%")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

