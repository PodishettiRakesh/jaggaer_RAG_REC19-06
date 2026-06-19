import argparse
import os
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pdfplumber


def load_pdf_text(pdf_path: str) -> List[Dict[str, object]]:
    """Read all text pages from a PDF and preserve page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            pages.append({"page": page_number, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def build_documents(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, object]]:
    """Load one PDF and create chunk metadata for Chroma ingestion."""
    source_name = Path(pdf_path).stem.replace(" ", "_")
    pages = load_pdf_text(pdf_path)

    documents = []
    for page_info in pages:
        page_number = page_info["page"]
        chunks = chunk_text(page_info["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, chunk in enumerate(chunks, start=1):
            documents.append(
                {
                    "id": f"{source_name}-p{page_number}-c{chunk_index}",
                    "document": chunk,
                    "metadata": {
                        "source": source_name,
                        "page": page_number,
                        "chunk": chunk_index,
                    },
                }
            )

    return documents


def create_chroma_collection(
    collection_name: str,
    documents: List[Dict[str, object]],
    persist_directory: str,
    model_name: str,
):
    """Create or replace a Chroma collection and persist embeddings locally."""
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name)

    texts = [doc["document"] for doc in documents]
    ids = [doc["id"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=texts,
        embeddings=embeddings,
    )
    client.persist()
    return client, collection


def load_chroma_collection(collection_name: str, persist_directory: str):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    if collection_name not in [col.name for col in client.list_collections()]:
        raise ValueError(f"Collection '{collection_name}' not found in '{persist_directory}'")
    collection = client.get_collection(name=collection_name)
    return client, collection


def collection_to_documents(collection) -> List[Dict[str, object]]:
    records = collection.get(include=["ids", "documents", "metadatas"])
    return [
        {"id": doc_id, "document": text, "metadata": metadata}
        for doc_id, text, metadata in zip(records["ids"], records["documents"], records["metadatas"])
    ]
