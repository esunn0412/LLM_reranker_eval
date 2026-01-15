import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_QUERIES = 10  # Adjust as needed
SCRATCH_PATH = "/local/scratch/tkim462"
OUTPUT_FILE = f"{SCRATCH_PATH}/eval_results_{MODEL_NAME.split('/')[-1]}.json"
REPORT_FILE = f"{SCRATCH_PATH}/final_report_{MODEL_NAME.split('/')[-1]}.txt"

# --- 1. Robust Parser ---
def parse_ranking(text, num_candidates):
    # 1. Find all sequences of digits (e.g., "0", "23", etc.)
    found = re.findall(r'\d+', text)
    
    try:
        # 2. Convert to integers 
        ranks = [int(x) for x in found if 0 <= int(x) < num_candidates]
        
        # 3. Remove duplicates while preserving the order
        seen = set()
        clean_ranks = [x for x in ranks if not (x in seen or seen.add(x))]
        
        return clean_ranks
    except Exception:
        return []

# --- 2. Metrics Calculation ---
def get_metrics(predicted_indices, ground_truth_indices):
    if not predicted_indices or not ground_truth_indices:
        return 0.0, 0.0
    
    # Average Precision (AP)
    ap = 0.0
    rel_found = 0
    for i, p in enumerate(predicted_indices, start=1):
        if p in ground_truth_indices:
            rel_found += 1
            ap += rel_found / i # number of relevant docs found so far / rank position of doc that's also in gt
    ap /= len(ground_truth_indices)
    
    # Discounted Cumulative Gain (NDCG)
    dcg = 0.0
    for i, p in enumerate(predicted_indices, start=1):
        if p in ground_truth_indices:
            dcg += 1.0 / np.log2(i + 1)
            
    idcg = 0.0
    for i, _ in enumerate(predicted_indices, start=1):
        idcg += 1.0 / np.log2(i + 1)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ap, ndcg

# --- 3. Prompt Logic ---
def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be the ranking as a list of identifiers, e.g., [3, 1, 2]. Only response the ranking results, do not say any word or explain."


def create_ranking_prompt(item):
    num = len(item['passages']['passage_text'])
    query = item['query']
    
    messages = get_prefix_prompt(query, num)
    for i, passage in enumerate(item['passages']['passage_text']):
        messages.append({'role': 'user', 'content': f'[{i}] {passage}'})
        messages.append({'role': 'assistant', 'content': f'Received passage [{i}].'})
    
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})
    return messages

# --- 4. Main Execution ---
def run_eval():
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(NUM_QUERIES))
    
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    
    prompts = [create_ranking_prompt(item) for item in dataset]
    outputs = llm.chat(messages=prompts, sampling_params=sampling_params) # list of RequestOutput
    
    all_ap, all_ndcg = [], []
    detailed_results = []
    
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text
        num_docs = len(dataset[i]['passages']['is_selected'])
        
        gt_indices = [idx for idx, val in enumerate(dataset[i]['passages']['is_selected']) if val == 1]
        pred_indices = parse_ranking(raw_text, num_docs)
        
        ap, ndcg = get_metrics(pred_indices, gt_indices)
        all_ap.append(ap)
        all_ndcg.append(ndcg)
        
        detailed_results.append({
            "qid": str(dataset[i]['query_id']),
            "raw_output": raw_text,
            "parsed": pred_indices,
            "gt": gt_indices,
            "ap": ap,
            "ndcg": ndcg
        })

    # Generate Report
    report = (
        f"LLM RERANKING REPORT\n"
        f"Model: {MODEL_NAME}\n"
        f"Dataset: MS MARCO v1.1 (Validation)\n"
        f"Queries: {NUM_QUERIES}\n"
        f"{'-'*30}\n"
        f"MEAN AVERAGE PRECISION (MAP): {np.mean(all_ap):.4f}\n"
        f"NDCG@10: {np.mean(all_ndcg):.4f}\n"
    )
    
    print(report)
    with open(REPORT_FILE, "w") as f: f.write(report)
    with open(OUTPUT_FILE, "w") as f: json.dump(detailed_results, f, indent=2)

if __name__ == "__main__":
    run_eval()