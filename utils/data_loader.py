"""
Data Loader for IR Benchmarks
Supports MS MARCO, BEIR, and TREC DL datasets
"""

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm


class MSMARCOLoader:
    """Loader for MS MARCO dataset"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
            os.environ['XDG_CACHE_HOME'] = cache_dir
    
    def load(self, split: str = "validation", num_queries: Optional[int] = None, 
             num_passages: int = 10) -> List[Dict]:
        """
        Load MS MARCO dataset
        
        Args:
            split: Dataset split ("train", "validation", "test")
            num_queries: Number of queries to load (None for all)
            num_passages: Number of passages per query to include
        
        Returns:
            List of query dictionaries with passages and qrels
        """
        print(f"Loading MS MARCO {split} split...")
        
        # Load dataset
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
        
        if num_queries:
            dataset = dataset.select(range(min(num_queries, len(dataset))))
        
        print(f"Processing {len(dataset)} queries...")
        
        # Process into structured format
        data = []
        for item in tqdm(dataset, desc="Processing queries"):
            query_id = str(item['query_id'])
            query = item['query']
            passages = item['passages']
            
            # Extract passages
            passage_list = []
            qrels = {}
            
            for i in range(min(num_passages, len(passages['passage_text']))):
                pid = f"{query_id}_p{i}"
                passage_text = passages['passage_text'][i]
                is_selected = passages['is_selected'][i]
                
                passage_list.append({
                    'pid': pid,
                    'text': passage_text,
                    'is_selected': is_selected
                })
                
                # Qrels: 1 if selected (relevant), 0 otherwise
                qrels[pid] = 1 if is_selected else 0
            
            data.append({
                'qid': query_id,
                'query': query,
                'passages': passage_list,
                'qrels': qrels
            })
        
        return data


class BEIRLoader:
    """Loader for BEIR benchmark datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
    
    def load(self, dataset_name: str = "scifact", split: str = "test", 
             num_queries: Optional[int] = None) -> List[Dict]:
        """
        Load BEIR dataset
        
        Args:
            dataset_name: BEIR dataset name (e.g., "scifact", "nfcorpus", "fiqa")
            split: Dataset split
            num_queries: Number of queries to load (None for all)
        
        Returns:
            List of query dictionaries with passages and qrels
        """
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
        except ImportError:
            raise ImportError("BEIR not installed. Install with: pip install beir")
        
        print(f"Loading BEIR dataset: {dataset_name}")
        
        # Download and load dataset
        data_path = util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
            self.cache_dir or "datasets"
        )
        
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        
        # Convert to common format
        data = []
        query_ids = list(queries.keys())
        
        if num_queries:
            query_ids = query_ids[:num_queries]
        
        for qid in tqdm(query_ids, desc="Processing queries"):
            query = queries[qid]
            
            # Get relevant documents
            relevant_docs = qrels.get(qid, {})
            doc_ids = list(relevant_docs.keys())
            
            # Get passages
            passage_list = []
            query_qrels = {}
            
            for i, doc_id in enumerate(doc_ids[:20]):  # Top 20 candidates
                pid = f"{qid}_p{i}"
                passage_text = corpus[doc_id].get('text', '')
                relevance = relevant_docs.get(doc_id, 0)
                
                passage_list.append({
                    'pid': pid,
                    'text': passage_text,
                    'original_docid': doc_id
                })
                
                query_qrels[pid] = relevance
            
            if passage_list:  # Only add if we have passages
                data.append({
                    'qid': qid,
                    'query': query,
                    'passages': passage_list,
                    'qrels': query_qrels
                })
        
        return data


class TRECDLLoader:
    """Loader for TREC Deep Learning Track datasets"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
    
    def load(self, year: str = "2019", split: str = "test", 
             num_queries: Optional[int] = None) -> List[Dict]:
        """
        Load TREC DL dataset
        
        Args:
            year: Dataset year ("2019" or "2020")
            split: Dataset split
            num_queries: Number of queries to load
        
        Returns:
            List of query dictionaries
        """
        # This is a placeholder - implement based on your TREC DL data access
        raise NotImplementedError("TREC DL loader not yet implemented. Use MS MARCO or BEIR.")


def get_data_loader(dataset_name: str, cache_dir: Optional[str] = None):
    """
    Factory function to get appropriate data loader
    
    Args:
        dataset_name: Name of dataset ("msmarco", "beir", "trec-dl")
        cache_dir: Cache directory for datasets
    
    Returns:
        Data loader instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "msmarco" or dataset_name == "ms_marco":
        return MSMARCOLoader(cache_dir=cache_dir)
    elif dataset_name == "beir":
        return BEIRLoader(cache_dir=cache_dir)
    elif dataset_name == "trec-dl" or dataset_name == "trec_dl":
        return TRECDLLoader(cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
