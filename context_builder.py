from typing import Dict, List, Tuple


def format_source_label(metadata: Dict[str, object]) -> str:
    parts = []
    if metadata.get("document"):
        parts.append(str(metadata["document"]))
    if metadata.get("company"):
        parts.append(f"Company: {metadata['company']}")
    if metadata.get("quarter"):
        parts.append(f"Quarter: {metadata['quarter']}")
    if metadata.get("page") is not None:
        parts.append(f"Page: {metadata['page']}")
    if metadata.get("chunk_id"):
        parts.append(f"Chunk: {metadata['chunk_id']}")
    return " | ".join(parts)


def format_citation(metadata: Dict[str, object]) -> str:
    document = metadata.get("document", "Unknown document")
    page = metadata.get("page", "?")
    return f"{document} Page {page}"


def _context_sort_key(chunk: Dict[str, object]) -> tuple:
    metadata = chunk.get("metadata", {})
    return (
        str(metadata.get("company", "")),
        str(metadata.get("quarter", "")),
        int(metadata.get("page", 0) or 0),
        -float(chunk.get("score", 0.0)),
    )


def build_context(chunks: List[Dict[str, object]]) -> Tuple[str, List[str]]:
    """Build LLM context from retrieved chunks and collect unique citations."""
    if not chunks:
        return "", []

    ordered_chunks = sorted(chunks, key=_context_sort_key)
    sections = []
    citations: List[str] = []
    seen_citations = set()

    for index, chunk in enumerate(ordered_chunks, start=1):
        metadata = chunk["metadata"]
        label = format_source_label(metadata)
        sections.append(f"[Source {index}: {label}]\n{chunk['document']}")

        citation = format_citation(metadata)
        if citation not in seen_citations:
            seen_citations.add(citation)
            citations.append(citation)

    return "\n\n".join(sections), citations
