import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  

import argparse
# from pathlib import Path

# from context_builder import build_context
...
from context_builder import build_context
from generation import format_sources_section, generate_answer
from load_and_embed import build_documents, create_chroma_collection, load_chroma_collection
from retrieval import hybrid_search


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--collection-name", default="kg_rag_single_doc", help="Chroma collection name.")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Chroma persistence directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks to return.")
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=20,
        help="Number of candidates to retrieve from each search method before fusion.",
    )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Sentence Transformers model name.")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Weight for dense retrieval scoring.")
    parser.add_argument("--sparse-weight", type=float, default=0.5, help="Weight for sparse retrieval scoring.")


def retrieve_chunks(args, collection):
    return hybrid_search(
        collection,
        query=args.query,
        model_name=args.model_name,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )


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
    results = retrieve_chunks(args, collection)

    print(f"Hybrid search results for: {args.query}\n")
    for idx, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"{idx}. Score: {result['score']:.4f} "
            f"(dense={result['dense_score']:.4f}, bm25={result['bm25_score']:.4f})"
        )
        print(
            f"   Document: {metadata.get('document')} | "
            f"Company: {metadata.get('company')} | "
            f"Quarter: {metadata.get('quarter')} | "
            f"Page: {metadata.get('page')} | "
            f"Chunk: {metadata.get('chunk_id')}"
        )
        print(result["document"][:600].replace("\n", " "))
        print("---")


def run_ask(args):
    _, collection = load_chroma_collection(args.collection_name, args.persist_dir)
    chunks = retrieve_chunks(args, collection)
    context, citations = build_context(chunks)
    answer = generate_answer(args.query, context, model=args.gemini_model)

    print(f"Question: {args.query}\n")
    print("Answer:")
    print(answer)
    print()
    print(format_sources_section(citations))


def main():
    parser = argparse.ArgumentParser(description="Document ingestion and hybrid retrieval for RAG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a single PDF and create embeddings.")
    ingest.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    ingest.add_argument("--collection-name", default="kg_rag_single_doc", help="Chroma collection name.")
    ingest.add_argument("--persist-dir", default="./chroma_db", help="Chroma persistence directory.")
    ingest.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters.")
    ingest.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    ingest.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Sentence Transformers model name.")

    query = subparsers.add_parser("query", help="Run hybrid search against an existing collection.")
    query.add_argument("query", help="Search query text.")
    add_retrieval_args(query)

    ask = subparsers.add_parser("ask", help="Retrieve context and generate a grounded answer with Gemini.")
    ask.add_argument("query", help="Question to answer.")
    ask.add_argument("--gemini-model", default=None, help="Gemini model name (defaults to GEMINI_MODEL in .env).")
    add_retrieval_args(ask)

    args = parser.parse_args()
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "ask":
        run_ask(args)


if __name__ == "__main__":
    main()
