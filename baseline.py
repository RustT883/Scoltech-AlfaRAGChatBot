import os
import sys
import json
import argparse

from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


def load_documents(path):
    """
    Loads and splits text documents from the specified directory.
    """
    loader = DirectoryLoader(path, glob="**/*.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="")
    documents = splitter.split_documents(documents)

    return documents


def build_qa_chain(documents):
    """
    Builds a retrieval-augmented QA pipeline using BM25 retriever.
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 3

    llm = ChatGroq(model="gemma2-9b-it")

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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run retrieval-based QA baseline using BM25 retriever"
    )
    parser.add_argument("input_json", help="Path to input JSON file with queries.")
    parser.add_argument("output_json", help="Path to output JSON file for predictions.")
    parser.add_argument("knowledge_base", help="Path to directory with .txt knowledge files.")
    args = parser.parse_args()

    # WARNING: Never commit real API keys to public repos. This key is a placeholder.
    os.environ["GROQ_API_KEY"] = "your-api-key"

    # Load documents and build QA chain
    documents = load_documents(args.knowledge_base)
    qa = build_qa_chain(documents)

    # Load input JSON file
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate predictions
    for sample in tqdm(data, file=sys.stdout, desc="Generating answers", ncols=80):
        sample["pred_response"] = qa.invoke(sample["query"])["result"]

    # Save results to output JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
