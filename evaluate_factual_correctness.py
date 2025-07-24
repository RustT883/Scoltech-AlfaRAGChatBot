import json
import asyncio
from pathlib import Path
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

# Configuration
JSON_FILES = [
    "valid_predictions_bm25s_recursive.json",
    "valid_predictions_chroma_recursive.json",
    "valid_predictions_hybrid_recursive.json",
    "valid_predictions_hybrid_semantic.json"
]
OUTPUT_SUMMARY = "factual_correctness_summary.txt"
LLM_MODEL = "qwen2.5:latest"

# Initialize LLM once
evaluator_llm = LangchainLLMWrapper(ChatOllama(
    base_url="http://localhost:11434",
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=4096
))

async def evaluate_file(file_path):
    """Evaluate a single JSON file and return mean score"""
    print(f"Processing {file_path}...")
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    scorer = FactualCorrectness(llm=evaluator_llm)
    scores = []
    
    # Process each item
    for item in data:
        try:
            sample = SingleTurnSample(
                response=item["pred_response"],
                reference=item["response"]
            )
            score = await scorer.single_turn_ascore(sample)
            item["factual_correctness_score"] = score
            scores.append(score)
        except Exception as e:
            print(f"Error processing item in {file_path}: {str(e)}")
            item["factual_correctness_score"] = None
    
    # Save scored version
    output_path = f"scored_{file_path}"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Calculate mean (ignore None scores)
    valid_scores = [s for s in scores if s is not None]
    mean_score = sum(valid_scores)/len(valid_scores) if valid_scores else 0
    
    print(f"Completed {file_path}. Mean score: {mean_score:.3f}")
    return file_path, mean_score, len(valid_scores)

async def main():
    # Process all files concurrently
    results = await asyncio.gather(*[evaluate_file(f) for f in JSON_FILES])
    
    # Generate summary
    summary_lines = [
        "Factual Correctness Evaluation Summary",
        "=====================================",
        f"{'File':<40} {'Mean Score':>12} {'Samples':>10}",
        "-"*64
    ]
    
    for file_path, mean_score, count in results:
        summary_lines.append(
            f"{Path(file_path).name:<40} {mean_score:>12.3f} {count:>10}"
        )
    
    # Save summary
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nSummary saved to {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    asyncio.run(main())
