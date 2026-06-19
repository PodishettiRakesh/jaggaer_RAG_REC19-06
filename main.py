import argparse
from pathlib import Path
from typing import List, Dict

from load_and_embed import build_documents, create_chroma_collection, load_chroma_collection, collection_to_documents
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize_query(query: str) -> List[str]:
    return query.lower().split()


def rank_bm25_documents(documents: List[Dict[str, object]], query: str, top_k: int = 5) -> List[Dict[str, object]]:
    texts = [doc["document"] for doc in documents]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize_query(query))
    ranked = sorted(
        zip(documents, scores), key=lambda item: item[1], reverse=True
    )
    return [doc for doc, score in ranked[:top_k]]


def dense_search(collection, query: str, model_name: str, top_k: int = 5):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def hybrid_search(collection, query: str, model_name: str, top_k: int = 5, dense_weight: float = 0.5, sparse_weight: float = 0.5):
    dense_results = dense_search(collection, query, model_name, top_k=top_k)
    dense_documents = [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(dense_results["documents"][0], dense_results["metadatas"][0], dense_results["distances"][0])
    ]

    documents = collection_to_documents(collection)
    sparse_documents = rank_bm25_documents(documents, query, top_k=top_k)

    sparse_ids = {doc["id"] for doc in sparse_documents}
    merged = []
    for idx, doc in enumerate(dense_documents):
        merged.append(
            {
                "id": None,
                "document": doc["document"],
                "metadata": doc["metadata"],
                "score": dense_weight * (1.0 / (1.0 + doc["distance"])),
                "source": "dense",
            }
        )
    for idx, doc in enumerate(sparse_documents):
        merged.append(
            {
                "id": doc["id"],
                "document": doc["document"],
                "metadata": doc["metadata"],
                "score": sparse_weight * (idx + 1),
                "source": "sparse",
            }
        )

    ranked = sorted(merged, key=lambda item: item["score"], reverse=True)
    unique = []
    seen_text = set()
    for item in ranked:
        if item["document"] in seen_text:
            continue
        seen_text.add(item["document"])
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def run_ingest(args):
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = build_documents(
        str(pdf_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Created {len(documents)} chunks from '{pdf_path.name}'")

    create_chroma_collection(
        collection_name=args.collection_name,
        documents=documents,
        persist_directory=args.persist_dir,
        model_name=args.model_name,
    )
    print(f"Chroma collection '{args.collection_name}' created at '{args.persist_dir}'")


def run_query(args):
    _, collection = load_chroma_collection(args.collection_name, args.persist_dir)
    results = hybrid_search(
        collection,
        query=args.query,
        model_name=args.model_name,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    print(f"Hybrid search results for: {args.query}\n")
    for idx, result in enumerate(results, start=1):
        source = result["source"]
        metadata = result["metadata"]
        print(f"{idx}. Source: {source} | Page: {metadata.get('page')} | Chunk: {metadata.get('chunk')}")
        print(result["document"][:600].replace("\n", " "))
        print("---")


def main():
    parser = argparse.ArgumentParser(description="Simple document ingestion and hybrid retrieval for single-document RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a single PDF and create embeddings.")
    ingest.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    ingest.add_argument("--collection-name", default="kg_rag_single_doc", help="Chroma collection name.")
    ingest.add_argument("--persist-dir", default="./chroma_db", help="Chroma persistence directory.")
    ingest.add_argument("--chunk-size", type=int, default=300, help="Chunk size in words.")
    ingest.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in words.")
    ingest.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Sentence Transformers model name.")

    query = subparsers.add_parser("query", help="Run hybrid search against an existing collection.")
    query.add_argument("query", help="Search query text.")
    query.add_argument("--collection-name", default="kg_rag_single_doc", help="Chroma collection name.")
    query.add_argument("--persist-dir", default="./chroma_db", help="Chroma persistence directory.")
    query.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    query.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Sentence Transformers model name.")
    query.add_argument("--dense-weight", type=float, default=0.5, help="Weight for dense retrieval scoring.")
    query.add_argument("--sparse-weight", type=float, default=0.5, help="Weight for sparse retrieval scoring.")

    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)


if __name__ == "__main__":
    main()
