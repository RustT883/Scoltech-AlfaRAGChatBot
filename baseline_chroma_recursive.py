__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import json
import argparse
import torch
import gc
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# Configuration
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 110
EMBEDDING_MODEL = "/home/rust/Downloads/IBMC/NLP/embed/nomic-embed-text-v1.5"
LLM_MODEL = "qwen2.5:latest"
PERSIST_DIRECTORY = "./chroma_db_ollama"

def load_and_split_documents(knowledge_base):
    """Load and split documents with memory optimization"""
    print("Loading documents...")
    loader = DirectoryLoader(knowledge_base, glob="**/*.txt")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n"], # ["\n\n\n\n", "\n\n\n", "\n\n", "\n"]
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(documents):
    """Create Chroma vectorstore with memory optimizations"""
    print("Creating embeddings...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda", 'trust_remote_code': True},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
    )
    
    # Process in chunks to avoid memory overload
    def chunked_documents(docs, chunk_size=1000):
        for i in range(0, len(docs), chunk_size):
            yield docs[i:i + chunk_size]
    
    # Create new vectorstore in chunks
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading existing vectorstore...")
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("Creating new vectorstore...")
        # First chunk creates the persistent store
        first_chunk = True
        for doc_chunk in tqdm(chunked_documents(documents), desc="Processing documents"):
            if first_chunk:
                vectorstore = Chroma.from_documents(
                    documents=doc_chunk,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                first_chunk = False
            else:
                # Subsequent chunks are added to the existing store
                Chroma.from_documents(
                    documents=doc_chunk,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
            
            torch.cuda.empty_cache()
            gc.collect()
        
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

def build_qa_chain(vectorstore):
    """Build memory-efficient QA chain"""
    print("Building QA chain...")
    
    # Initialize Ollama LLM
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
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
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
        description="Run retrieval-based QA using Chroma vectorstore"
    )
    parser.add_argument("input_json", help="Path to input JSON file with queries.")
    parser.add_argument("output_json", help="Path to output JSON file for predictions.")
    parser.add_argument("knowledge_base", help="Path to directory with .txt knowledge files.")
    args = parser.parse_args()

    # Verify Ollama models are available
    assert "nomic-embed-text:v1.5" in os.popen("ollama list").read(), "Missing embedding model - run: ollama pull nomic-embed-text:v1.5"
    assert "qwen2.5:latest" in os.popen("ollama list").read(), "Missing LLM model - run: ollama pull qwen2.5:latest"

    # Load or create vectorstore
    documents = load_and_split_documents(args.knowledge_base)
    vectorstore = create_vectorstore(documents)
    qa_chain = build_qa_chain(vectorstore)

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
