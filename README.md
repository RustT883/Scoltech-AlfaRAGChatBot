# RAG-Powered LLM Agents for Banking Chatbots and Assistants

This repository contains the baseline solution for SMILES 2025 project focused on building a **Retrieval-Augmented Generation (RAG)** system for banking chatbots using Large Language Models (LLMs). The goal is to create an assistant capable of producing accurate and complete answers using Alfa-Bank’s internal knowledge base.

---

## 🚀 Project Overview

### Scope

Participants are provided with:

- A **knowledge base** of articles describing Alfa-Bank’s financial products (as `.txt` files)
- A **validation dataset**: pairs of *customer question – reference answer*
- A **test set** of customer questions
- A **simple baseline solution** (this repo)

Your task is to develop a system that uses the provided documents to accurately respond to customer questions, optimizing answer quality.

### Why It Matters

Traditional rule-based chatbots are being replaced by intelligent LLM-driven assistants. However, LLMs do not have access to private, up-to-date data like banking product terms, making them unsuitable for real-world financial services without augmentation.

This project supports the industry shift toward **retrieval-augmented assistants**, and provides valuable hands-on experience in building production-grade LLM systems.

---

## 🎯 Goals

- **Build an LLM-Based Chatbot**  
  Develop a chatbot using a retrieval-augmented pipeline and Alfa-Bank’s internal knowledge base.

- **Optimize Answer Quality**  
  Improve the pipeline to achieve top performance on the METEOR metric.

- **Present Your Solution**  
  Deliver a clear report or presentation outlining your approach, innovations, and results.

---

## 📦 Expected Deliverables

- ✅ Source code of the LLM-based assistant (e.g., this GitHub repo)
- ✅ A JSON file with generated answers to test set questions
- ✅ A report or slide deck summarizing your methodology and evaluation

---

## 📁 Baseline Structure

### Files

- `baseline.py` — Loads the knowledge base and runs a BM25 + LLM (ChatGroq) retrieval-augmented QA system.
- `metrics.py` — Evaluates predictions using the **METEOR** metric.

### Key Components

- **Retriever**: BM25 (via `langchain_community.retrievers`)
- **LLM**: `gemma2-9b-it` served via Groq. You are free to use alternative LLMs and APIs.
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

- The main evaluation metric is **METEOR**
- Reference answers are compared to model outputs
- The goal is to **maximize the METEOR score**

- The **main problem** is to generate answers that are **relevant**, **precise**, and **complete** based on the provided knowledge base.
- **METEOR** is used as a **proxy metric** to approximate answer quality. It compares generated responses to reference answers using lexical and semantic overlap.
- Participants may also use additional evaluation metrics that better capture factual correctness, completeness, or other qualitative aspects — especially if they highlight limitations in METEOR.
  
---

## 📚 Dataset & Knowledge Base

Datasets will be provided via a separate download link and should include:

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
