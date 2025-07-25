import os
import json
import argparse
import torch
import gc
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from custom_langchain.retrievers import BM25SRetriever

# Configuration
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 50
LLM_MODEL = "qwen2.5:latest"

def load_and_split_documents(knowledge_base):
    """Load and split documents"""
    print("Loading documents...")
    loader = DirectoryLoader(knowledge_base, glob="**/*.txt")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n"],
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_bm25_retriever(documents):
    """Create and persist BM25S Index"""
    print("Creating BM25S Index...")
    
    if not os.path.exists('bm25s_index'):
        texts = [doc.page_content for doc in documents]
        BM25SRetriever.from_texts(
            texts=texts,
            k=4,
            persist_directory='bm25s_index'
        )
    
    return BM25SRetriever.from_persisted_directory("bm25s_index", k=4)

def build_qa_chain(retriever):
    """Build QA chain with BM25 retriever"""
    print("Building QA chain...")
    
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model=LLM_MODEL,
        temperature=0.0,
        num_ctx=4096,
    )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the client's question concisely, clearly, and to the point—without speculation or digressions.\n"
            "You can not ask for more context or ask questions, just answer.\n"
            "Use information from given context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )
   
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

def load_checkpoint(output_file):
    """Load existing checkpoint if available"""
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_checkpoint(data, output_file):
    """Save progress to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run retrieval-based QA using BM25S"
    )
    parser.add_argument("input_json", help="Path to input JSON file with queries.")
    parser.add_argument("output_json", help="Path to output JSON file for predictions.")
    parser.add_argument("knowledge_base", help="Path to directory with .txt knowledge files.")
    args = parser.parse_args()

    # Verify Ollama model is available
    assert "qwen2.5:latest" in os.popen("ollama list").read(), "Missing LLM model - run: ollama pull qwen2.5:latest"

    # Load documents and create retriever
    documents = load_and_split_documents(args.knowledge_base)
    retriever = create_bm25_retriever(documents)
    qa_chain = build_qa_chain(retriever)

    # Load input data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(args.output_json)
    if checkpoint_data:
        print(f"Resuming from checkpoint with {len(checkpoint_data)} processed items")
        data = checkpoint_data
    else:
        data = input_data

    # Process queries with checkpointing
    try:
        for i, sample in enumerate(tqdm(data, desc="Generating answers")):
            if "pred_response" not in sample:  # Only process if not already done
                try:
                    result = qa_chain.invoke(sample["query"])
                    sample["pred_response"] = result["result"]
                    
                    # Save checkpoint every 10 items
                    if i % 10 == 0:
                        save_checkpoint(data, args.output_json)
                        
                except Exception as e:
                    print(f"Error processing query: {sample['query']} - {str(e)}")
                    sample["pred_response"] = f"Error: {str(e)}"
                    
                torch.cuda.empty_cache()
                gc.collect()
                
    finally:
        # Ensure final save even if interrupted
        save_checkpoint(data, args.output_json)
        print(f"Predictions saved to {args.output_json}")

if __name__ == "__main__":
    main()
