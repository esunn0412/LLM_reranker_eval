# LLM Ranker Evaluation Pipeline

A comprehensive framework for evaluating Large Language Model (LLM) rankers on standard Information Retrieval (IR) benchmarks.

## Overview

This pipeline evaluates which LLM ranker can be trusted for training and evaluating preference-based ranking tasks. It validates ranking capabilities on independent, standard IR benchmarks (MS MARCO, BEIR, TREC DL) before using them for downstream tasks.

### Key Features

- **Multiple Model Support**: Evaluate both open-source (LLaMA, Gemma, Qwen) and closed-source (GPT-4, Claude) models
- **Standard Benchmarks**: MS MARCO, BEIR datasets, TREC DL
- **Flexible Prompting**: RankGPT-style, listwise, and pairwise ranking prompts
- **Comprehensive Metrics**: MAP, NDCG@k, MRR, Precision@k
- **Automated Pipeline**: Run multiple models sequentially with automatic comparison
- **Organized Output**: Results saved per model with JSON and text reports

## Project Structure

```
rerank/
├── configs/
│   ├── models.yaml          # Model specifications
│   └── config.yaml          # Evaluation configuration
├── scripts/
│   ├── run_rerank.py        # Main inference script (The "Doer")
│   ├── evaluate.py          # Metric calculation script (The "Grader")
│   └── prompts.py           # Prompt templates (RankGPT style)
├── utils/
│   ├── data_loader.py       # Dataset loading (MS MARCO/BEIR)
│   └── metrics.py           # MAP, NDCG calculation logic
├── results/                 # Output directory
│   ├── {model_name}/
│   │   ├── rankings.json    # Detailed ranking outputs
│   │   ├── eval_data.json   # Evaluation data
│   │   ├── metrics.json     # Computed metrics
│   │   └── metrics.txt      # Human-readable report
│   └── comparison.json      # Cross-model comparison
├── main.py                  # Pipeline orchestration
└── requirements_new.txt     # Core dependencies
```

## Installation

### 1. Create Virtual Environment

```bash
cd /local/scratch/tkim462/rerank
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements_new.txt
```

### 3. Optional: Install pytrec_eval for Official TREC Metrics

```bash
pip install pytrec_eval
```

### 4. Set API Keys (for closed-source models)

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Configuration

### Model Configuration (`configs/models.yaml`)

Define which models to evaluate:

```yaml
open_source_models:
  - name: "llama-8b"
    model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    type: "vllm"
    enabled: true  # Set to true to run

closed_source_models:
  - name: "gpt-4o-mini"
    model_id: "gpt-4o-mini"
    type: "openai"
    enabled: true
```

**Enable/disable models** by setting `enabled: true/false`.

### Evaluation Configuration (`configs/config.yaml`)

```yaml
dataset:
  name: "msmarco"
  num_queries: 100
  num_passages: 10

evaluation:
  metrics:
    - "ndcg_cut_10"
    - "map_cut_10"
    - "recip_rank"

prompt:
  style: "rankgpt"  # Options: rankgpt, listwise, pairwise
```

## Usage

### Option 1: Run All Enabled Models (Recommended)

```bash
python main.py
```

This will:
1. Run reranking for all enabled models in `configs/models.yaml`
2. Compute metrics for each model
3. Generate a comparison report

### Option 2: Run Single Model

```bash
python main.py --model llama-8b
```

### Option 3: Manual Step-by-Step

#### Step 1: Run Reranking

```bash
python scripts/run_rerank.py --model llama-8b
```

Output: `results/llama-8b/eval_data.json`

#### Step 2: Evaluate

```bash
python scripts/evaluate.py --input results/llama-8b/eval_data.json
```

Output: `results/llama-8b/metrics.json` and `metrics.txt`

#### Step 3: Compare with Baseline (Optional)

```bash
python scripts/evaluate.py \
    --input results/llama-8b/eval_data.json \
    --baseline results/gpt-4o-mini/eval_data.json
```

## Output Files

### Per-Model Results (`results/{model_name}/`)

- **`rankings.json`**: Full reranking results with raw LLM outputs
- **`eval_data.json`**: Structured data for evaluation (rankings + qrels)
- **`metrics.json`**: Computed IR metrics
- **`metrics.txt`**: Human-readable metrics report

### Comparison Report (`results/comparison.json`)

Aggregates all model results for easy comparison:

```json
{
  "models": ["llama-8b", "gpt-4o-mini", "claude-haiku-4"],
  "metrics": {
    "NDCG@10": {
      "llama-8b": 0.4523,
      "gpt-4o-mini": 0.4891,
      "claude-haiku-4": 0.4672
    },
    ...
  }
}
```

## Task Definition

**Input**: (Query, Documents)

**Output**: Ranked document indices (e.g., [7, 2, 1, 4, 0, 3, 6, 5])

**Evaluation**: Whether relevant documents are ranked highly
- **Metrics**: MAP, NDCG@k, MRR, Precision@k

## Supported Models

### Open-Source (via vLLM)
- LLaMA 3.2 3B / 3.1 8B / 3.1 70B
- Gemma 2 27B
- Qwen 2.5 72B
- Any HuggingFace model compatible with vLLM

### Closed-Source (via API)
- GPT-4, GPT-4o, GPT-4o-mini
- Claude Sonnet 4, Claude Haiku 4
- Any model supported by OpenAI or Anthropic APIs

## Supported Datasets

- **MS MARCO** (v1.1): Passage ranking
- **BEIR**: Multiple domain-specific datasets
- **TREC DL**: Deep Learning Track (planned)

## Prompt Styles

### RankGPT (Default)
Interactive chat format with passage-by-passage presentation:
```
System: You are RankGPT...
User: I will provide you with 10 passages...
User: [0] First passage...
Assistant: Received passage [0].
...
User: Rank the passages...
```

### Listwise
Single-turn prompt with all passages:
```
Given these 10 passages, rank them by relevance...
```

### Pairwise
Compare passages pairwise (for pairwise ranking algorithms)

## Customization

### Add New Model

Edit `configs/models.yaml`:

```yaml
open_source_models:
  - name: "my-custom-model"
    model_id: "organization/model-name"
    type: "vllm"
    tensor_parallel_size: 1
    enabled: true
```

### Add New Dataset

Extend `utils/data_loader.py` with a new loader class.

### Modify Prompts

Edit `scripts/prompts.py` to customize prompt templates.

## Advanced Usage

### Run on Subset for Quick Testing

```bash
# Edit configs/config.yaml
dataset:
  num_queries: 10  # Test on 10 queries only
```

### Run on Different Datasets

```bash
# For BEIR datasets
dataset:
  name: "beir"
  beir_dataset: "scifact"
```

### Custom Output Directory

```bash
python scripts/run_rerank.py --model llama-8b --output_dir custom_results/
```

## Troubleshooting

### GPU Memory Issues

For large models (70B), reduce `gpu_memory_utilization`:

```yaml
gpu_memory_utilization: 0.70  # Lower from 0.85
```

### API Rate Limits

For closed-source models, add delays between requests (modify `scripts/run_rerank.py`).

### Import Errors

Ensure virtual environment is activated:
```bash
source rerank_env/bin/activate
```

## Contributing

To extend the pipeline:

1. **New metrics**: Add to `utils/metrics.py`
2. **New prompts**: Add to `scripts/prompts.py`
3. **New datasets**: Add loader to `utils/data_loader.py`
4. **New model types**: Extend `ModelInference` class in `scripts/run_rerank.py`

## Citation

Based on RankGPT methodology:
```
@article{rankgpt,
  title={Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent},
  author={Sun, Weiwei and others},
  year={2023}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue in the repository.
