import json
import asyncio
from pathlib import Path
import os
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

# Configuration
JSON_FILES = [
    "valid_predictions_bm25s_semantic.json",
    "valid_predictions_chroma_semantic.json"
]
OUTPUT_SUMMARY = "factual_correctness_summary.txt"
CHECKPOINT_FILE = "evaluation_checkpoint.json"
LLM_MODEL = "qwen2.5:latest"

class EvaluationCheckpoint:
    def __init__(self):
        self.state = {
            'completed_files': [],
            'current_file': None,
            'current_file_progress': 0,
            'results': []
        }
        
    def load(self):
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                self.state = json.load(f)
            print(f"Resuming from checkpoint. Already completed: {len(self.state['completed_files'])} files")
            
    def save(self):
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
            
    def mark_file_complete(self, file_path, result):
        self.state['completed_files'].append(file_path)
        self.state['results'].append(result)
        self.state['current_file'] = None
        self.state['current_file_progress'] = 0
        self.save()
        
    def update_progress(self, file_path, progress):
        self.state['current_file'] = file_path
        self.state['current_file_progress'] = progress
        self.save()

async def evaluate_file(file_path, checkpoint):
    """Evaluate a single JSON file with checkpointing"""
    print(f"\nProcessing {file_path}...")
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if we're resuming this file
    start_idx = 0
    if checkpoint.state['current_file'] == file_path:
        start_idx = checkpoint.state['current_file_progress']
        print(f"Resuming from item {start_idx}/{len(data)}")
    
    scorer = FactualCorrectness(llm=evaluator_llm)
    scores = []
    output_path = f"scored_{file_path}"
    
    # Load existing results if resuming
    if os.path.exists(output_path) and start_idx > 0:
        with open(output_path, 'r') as f:
            scored_data = json.load(f)
    else:
        scored_data = data.copy()
    
    for i in range(start_idx, len(data)):
        try:
            sample = SingleTurnSample(
                response=data[i]["pred_response"],
                reference=data[i]["response"]
            )
            score = await scorer.single_turn_ascore(sample)
            scored_data[i]["factual_correctness_score"] = score
            scores.append(score)
            
            if i % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(scored_data, f, indent=2)
                checkpoint.update_progress(file_path, i)
                print(f"Progress: {i+1}/{len(data)} | Current score: {score:.3f}")
                
        except Exception as e:
            print(f"Error processing item {i} in {file_path}: {str(e)}")
            scored_data[i]["factual_correctness_score"] = None
    
    with open(output_path, 'w') as f:
        json.dump(scored_data, f, indent=2)
    
    valid_scores = [s for s in scores if s is not None]
    mean_score = sum(valid_scores)/len(valid_scores) if valid_scores else 0
    
    result = (file_path, mean_score, len(valid_scores))
    checkpoint.mark_file_complete(file_path, result)
    
    print(f"Completed {file_path}. Mean score: {mean_score:.3f}")
    return result

async def main():
    global evaluator_llm
    
    # Initialize LLM
    evaluator_llm = LangchainLLMWrapper(ChatOllama(
        base_url="http://localhost:11434",
        model=LLM_MODEL,
        temperature=0.0,
        num_ctx=4096
    ))
    
    checkpoint = EvaluationCheckpoint()
    checkpoint.load()
    
    files_to_process = [f for f in JSON_FILES if f not in checkpoint.state['completed_files']]
    
    if not files_to_process:
        print("All files already processed according to checkpoint")
    else:
        # Process remaining files
        results = await asyncio.gather(*[
            evaluate_file(f, checkpoint) for f in files_to_process
        ])
        checkpoint.state['results'].extend(results)
    
    # Generate summary
    summary_lines = [
        "Factual Correctness Evaluation Summary",
        "=====================================",
        f"{'File':<40} {'Mean Score':>12} {'Samples':>10}",
        "-"*64
    ]
    
    for file_path, mean_score, count in checkpoint.state['results']:
        summary_lines.append(
            f"{Path(file_path).name:<40} {mean_score:>12.3f} {count:>10}"
        )
    
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("\n".join(summary_lines))
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print(f"\nEvaluation complete. Summary saved to {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    asyncio.run(main())
