"""
Evaluation Metrics for Reranking
Core metrics: NDCG@k and MAP for evaluating ranking quality
"""

import numpy as np
from typing import Dict, List, Optional


def average_precision(ranking: List[str], qrels: Dict[str, int], k: Optional[int] = None) -> float:
    """
    Calculate Average Precision (AP), optionally at cutoff k
    
    Args:
        ranking: List of document IDs in ranked order
        qrels: Dictionary mapping doc IDs to relevance scores (1 = relevant)
        k: Cutoff rank (None = use all rankings)
    
    Returns:
        Average Precision score
    """
    if not ranking or not qrels:
        return 0.0
    
    # Apply cutoff if specified
    if k is not None:
        ranking = ranking[:k]
    
    # Count total relevant documents
    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    
    if total_relevant == 0:
        return 0.0
    
    ap = 0.0
    relevant_found = 0
    
    for i, doc_id in enumerate(ranking, start=1):
        if qrels.get(doc_id, 0) > 0:
            relevant_found += 1
            precision_at_i = relevant_found / i
            ap += precision_at_i
    
    ap /= total_relevant
    return ap


def mean_average_precision(all_rankings: Dict[str, List[str]], 
                           all_qrels: Dict[str, Dict[str, int]],
                           k: Optional[int] = None) -> float:
    """
    Calculate Mean Average Precision (MAP), optionally at cutoff k
    
    Args:
        all_rankings: Dict mapping query IDs to ranked doc ID lists
        all_qrels: Dict mapping query IDs to qrels dicts
        k: Cutoff rank (None = adaptive per query based on passage count)
    
    Returns:
        MAP score (or MAP@k if k specified)
    """
    if not all_rankings or not all_qrels:
        return 0.0
    
    aps = []
    for qid in all_rankings:
        if qid in all_qrels:
            ap = average_precision(all_rankings[qid], all_qrels[qid], k)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def ndcg_at_k(ranking: List[str], qrels: Dict[str, int], k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k (NDCG@k)
    
    Args:
        ranking: List of document IDs in ranked order
        qrels: Dictionary mapping doc IDs to relevance scores
        k: Cutoff position (None = use all available passages)
    
    Returns:
        NDCG@k score
    """
    if not ranking or not qrels:
        return 0.0
    
    # Truncate to k (or use all if k is None)
    if k is not None:
        ranking = ranking[:k]
        k_actual = k
    else:
        k_actual = len(ranking)
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(ranking, start=1):
        rel = qrels.get(doc_id, 0)
        dcg += rel / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    ideal_ranking = sorted(qrels.values(), reverse=True)[:k_actual]
    idcg = 0.0
    for i, rel in enumerate(ideal_ranking, start=1):
        idcg += rel / np.log2(i + 1)
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_ndcg_at_k(all_rankings: Dict[str, List[str]], 
                   all_qrels: Dict[str, Dict[str, int]], 
                   k: Optional[int] = None) -> float:
    """
    Calculate Mean NDCG@k across all queries
    
    Args:
        all_rankings: Dict mapping query IDs to ranked doc ID lists
        all_qrels: Dict mapping query IDs to qrels dicts
        k: Cutoff position (None = adaptive per query based on passage count)
    
    Returns:
        Mean NDCG@k score
    """
    if not all_rankings or not all_qrels:
        return 0.0
    
    ndcgs = []
    for qid in all_rankings:
        if qid in all_qrels:
            ndcg = ndcg_at_k(all_rankings[qid], all_qrels[qid], k)
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def calculate_reranking_metrics(all_rankings: Dict[str, List[str]], 
                               all_qrels: Dict[str, Dict[str, int]],
                               k: int = 10) -> Dict[str, float]:
    """
    Calculate core reranking metrics: NDCG@k and MAP@k
    
    These are the most important metrics for evaluating reranking:
    - NDCG@k: Normalized Discounted Cumulative Gain (accounts for position and relevance)
    - MAP@k: Mean Average Precision (binary relevance quality)
    - NDCG/MAP (full): Evaluated on all passages adaptively per query
    
    Args:
        all_rankings: Dict mapping query IDs to ranked doc ID lists
        all_qrels: Dict mapping query IDs to qrels dicts
        k: Cutoff rank (will use min(k, actual_passage_count) per query)
    
    Returns:
        Dictionary with 'NDCG@k', 'MAP@k', 'NDCG', 'MAP' (full ranking)
    """
    metrics = {}
    
    # Metrics at cutoff k (standard comparison point)
    metrics[f'NDCG@{k}'] = mean_ndcg_at_k(all_rankings, all_qrels, k)
    metrics[f'MAP@{k}'] = mean_average_precision(all_rankings, all_qrels, k)
    
    # Full ranking metrics (adaptive to each query's passage count)
    metrics['NDCG'] = mean_ndcg_at_k(all_rankings, all_qrels, k=None)
    metrics['MAP'] = mean_average_precision(all_rankings, all_qrels, k=None)
    
    return metrics


def convert_scores_to_ranking(scores: Dict[str, float]) -> List[str]:
    """
    Convert score dictionary to ranked list of document IDs
    
    Args:
        scores: Dictionary mapping doc IDs to scores
    
    Returns:
        List of doc IDs sorted by score (descending)
    """
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def use_pytrec_eval(all_qrels: Dict[str, Dict[str, int]], 
                    all_rankings: Dict[str, Dict[str, float]],
                    k: int = 10) -> Dict[str, float]:
    """
    Calculate metrics using pytrec_eval (if available)
    
    Args:
        all_qrels: Dict mapping query IDs to qrels dicts
        all_rankings: Dict mapping query IDs to score dicts
        k: Cutoff for @k metrics
    
    Returns:
        Dictionary of averaged metric scores
    """
    try:
        import pytrec_eval
    except ImportError:
        print("Warning: pytrec_eval not installed. Using fallback metrics.")
        return {}
    
    measures = {f'ndcg_cut_{k}', f'map_cut_{k}'}
    
    evaluator = pytrec_eval.RelevanceEvaluator(all_qrels, measures)
    results = evaluator.evaluate(all_rankings)
    
    # Average across queries
    averaged_metrics = {}
    for measure in measures:
        scores = [query_results.get(measure, 0.0) for query_results in results.values()]
        averaged_metrics[measure] = np.mean(scores) if scores else 0.0
    
    return averaged_metrics
