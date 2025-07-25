# RAG-Powered LLM Agents for Banking Chatbots and Assistants

This convoluted mess contains various solutions for SMILES 2025 project focused on building a **Retrieval-Augmented Generation (RAG)** system for banking chatbots using Large Language Models (LLMs), utilizing hybrid search and semantic splitting.

| Configuration                  | Factual Correctness | METEOR |
|--------------------------------|---------------------|--------|
| **Semantic Chunking**          |                     |        |
| - BM25S Only                   | 0.38                | 0.37   |
| - Chroma Only                  | 0.33                | 0.33   |
| - Hybrid (BM25S + Chroma)      | 0.37                | 0.36   |
| **Recursive Chunking**         |                     |        |
| - BM25S Only                   | 0.41                | 0.38   |
| - Chroma Only                  | 0.38                | 0.37   |
| - Hybrid (BM25S + Chroma)      | **0.41**            | **0.39** |

## 📁 Structure

### Files

- `baseline_*.py` — create disk-persisted knowledge bases and run RAG to generate JSONs `valid_predictions_*.json`.
- `metrics.py` — Evaluates predictions using the **METEOR** metric on `valid_predictions_*.json` files.
- `evaluate_factual_correctness` — calculates Factual Correctness metric from RAGAS on `valid_predictions_*.json` files.

### Key Components

- **Retrievers**: Custom BM25S Langchain Integration and Langchain Chroma
- **LLM**: `Qwen2.5` served via Ollama.
- **Prompt**: Instructional prompt ensuring concise and grounded answers

---

### Run Baseline QA Generation

```bash
python3 baseline_*.py valid_dataset.json valid_predictions.json knowledge/
```

- `valid_dataset.json`: input JSON with customer queries
- `valid_predictions_*.json`: output file with predicted responses
- `knowledge/`: folder containing `.txt` documents used as the knowledge base

### Run Evaluation

```bash
python3 metrics.py valid_predictions_*.json
```
This will print the **average METEOR score** for the predicted answers.

```bash
python3 evaluate_factual_correctness.py'
```
This will run **Factual Correctness** on a list of JSON files with predictions specified in `evaluate_factual_correctness.py`

---

## 📊 Metrics

- The available evaluation metrics are **METEOR** and **Factual Correctness**
- Reference answers are compared to model outputs

  
---

## 📚 Dataset & Knowledge Base

Datasets are available: https://huggingface.co/datasets/mllab/smiles-2025

- `valid_dataset.json` — Validation examples with ground-truth responses
- `test_dataset.json` — Unlabeled questions for final evaluation
- `knowledge/` — Directory with knowledge base files in `.txt` format

---

## :page_with_curl: Report

This repo also contains the `.pdf` file — a project report paper for SMILES2025. You can read it if you want. 

---

## 📎 Example

### Input JSON (`valid_dataset.json`)
```json
[
  {
    "query": "How can I apply for a mortgage?",
    "response": "You can apply for a mortgage through Alfa-Bank's online platform or by visiting a branch..."
  }
]
```

### Output JSON (`valid_predictions.json`)
```json
[
  {
    "query": "How can I apply for a mortgage?",
    "response": "You can apply for a mortgage through Alfa-Bank's online platform...",
    "pred_response": "You can apply for a mortgage on Alfa-Bank’s website or in a local branch."
  }
]
```
## :rocket: Reproduce the results

Make sure to build a virtual environment and do `pip install -r requirements.txt`, pretty basic stuff.
