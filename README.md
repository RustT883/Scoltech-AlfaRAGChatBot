# RAG-Powered LLM Agents for Banking Chatbots and Assistants

This repository contains the hybrid retrieval solutions for SMILES 2025 project focused on building a **Retrieval-Augmented Generation (RAG)** system for banking chatbots using Large Language Models (LLMs).



## 📁 Structure

### Files

- `baseline.py` — Loads the knowledge base and runs a BM25 + LLM (ChatGroq) retrieval-augmented QA system.
- `metrics.py` — Evaluates predictions using the **METEOR** metric.

### Key Components

- **Retriever**: Custom BM25S Langchain Integration
- **LLM**: `Qwen2.5` served via Ollama. You are free to use alternative LLMs and APIs.
- **Prompt**: Instructional prompt ensuring concise and grounded answers

---

### Run Baseline QA Generation

```bash
python3 baseline.py valid_dataset.json valid_predictions.json knowledge/
```

- `valid_dataset.json`: input JSON with customer queries
- `valid_predictions.json`: output file with predicted responses
- `knowledge/`: folder containing `.txt` documents used as the knowledge base

### Run Evaluation

```bash
python3 metrics.py valid_predictions.json
```

This will print the **average METEOR score** for the predicted answers.

---

## 📊 Metric

- The available evaluation metrics are **METEOR**, **Context Precision**, **Context Recall** and **Faithfulness**
- Reference answers are compared to model outputs

  
---

## 📚 Dataset & Knowledge Base

Datasets are available: https://huggingface.co/datasets/mllab/smiles-2025

- `valid_dataset.json` — Validation examples with ground-truth responses
- `test_dataset.json` — Unlabeled questions for final evaluation
- `knowledge/` — Directory with knowledge base files in `.txt` format

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

---

## 🧠 Future Work

You can go beyond this baseline by:
- Improving retrieval quality
- Tuning prompts and model configuration
- Adding fallback mechanisms or filtering
- Using alternative LLMs methods

---

## 📩 Contact

For questions or clarifications, please contact your project mentor (Telegram: @aaasenin).

---
