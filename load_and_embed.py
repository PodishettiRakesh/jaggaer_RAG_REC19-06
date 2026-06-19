import os
import re
from pathlib import Path
from typing import List, Dict

import chromadb
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


def parse_source_metadata(pdf_path: str) -> Dict[str, object]:
    """Extract document, company, and quarter metadata from a PDF filename."""
    filename = Path(pdf_path).name
    stem = Path(pdf_path).stem.replace(" ", "_")
    parts = [part for part in stem.split("_") if part]

    metadata: Dict[str, object] = {
        "document": filename,
        "source": stem,
    }

    quarter_match = re.match(r"^(\d{4})[_ ]?(Q[1-4])", stem, re.IGNORECASE)
    if quarter_match:
        metadata["quarter"] = f"{quarter_match.group(1)}_{quarter_match.group(2).upper()}"

    if len(parts) >= 3 and parts[0].isdigit() and parts[1].upper().startswith("Q"):
        metadata["company"] = parts[2].upper()
    elif parts:
        metadata["company"] = parts[-1].upper()

    return metadata


def recursive_character_split(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Split text with a recursive character splitter (LangChain-style separators)."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    separators = ["\n\n", "\n", ". ", " ", ""]
    for separator in separators:
        if separator and separator not in text:
            continue

        if separator == "":
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end].strip())
                if end >= len(text):
                    break
                start = max(end - chunk_overlap, start + 1)
            return [chunk for chunk in chunks if chunk]

        parts = text.split(separator)
        chunks: List[str] = []
        current = ""

        for index, part in enumerate(parts):
            segment = part if index == len(parts) - 1 else part + separator
            if len(current) + len(segment) <= chunk_size:
                current += segment
                continue

            if current.strip():
                chunks.append(current.strip())

            if len(segment) > chunk_size:
                chunks.extend(recursive_character_split(segment, chunk_size, chunk_overlap))
                current = ""
            elif chunk_overlap and current:
                current = current[-chunk_overlap:] + segment
            else:
                current = segment

        if current.strip():
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    return [text.strip()]


def build_documents(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, object]]:
    """Load one PDF and create enriched chunk metadata for Chroma ingestion."""
    source_metadata = parse_source_metadata(pdf_path)
    source_name = str(source_metadata["source"])
    pages = load_pdf_text(pdf_path)

    documents = []
    chunk_counter = 0
    for page_info in pages:
        page_number = page_info["page"]
        chunks = recursive_character_split(
            page_info["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter}"
            documents.append(
                {
                    "id": f"{source_name}-p{page_number}-c{chunk_index}",
                    "document": chunk,
                    "metadata": {
                        **source_metadata,
                        "page": page_number,
                        "chunk": chunk_index,
                        "chunk_id": chunk_id,
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
    client = chromadb.PersistentClient(path=persist_directory)

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
    return client, collection


def load_chroma_collection(collection_name: str, persist_directory: str):
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    if collection_name not in [col.name for col in client.list_collections()]:
        raise ValueError(f"Collection '{collection_name}' not found in '{persist_directory}'")
    collection = client.get_collection(name=collection_name)
    return client, collection


def collection_to_documents(collection) -> List[Dict[str, object]]:
    records = collection.get(include=["documents", "metadatas"])
    return [
        {"id": doc_id, "document": text, "metadata": metadata}
        for doc_id, text, metadata in zip(records["ids"], records["documents"], records["metadatas"])
    ]
