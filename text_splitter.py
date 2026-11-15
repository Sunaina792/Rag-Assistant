"""
text_splitter.py – Pure-Python recursive text splitter. No external dependencies.
"""

from typing import List, Dict, Any

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 200
SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, sep: str) -> List[str]:
    return text.split(sep) if sep else list(text)


def _build_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Sliding-window chunker that tries paragraph → sentence → word boundaries.
    """
    # Choose the coarsest separator that actually appears in the text
    chosen_sep = ""
    for sep in SEPARATORS:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    parts = _split_on_separator(text, chosen_sep)

    chunks: List[str] = []
    window: List[str] = []
    window_len: int = 0

    for part in parts:
        part_len = len(part) + len(chosen_sep)

        if window and window_len + part_len > chunk_size:
            # Flush current window as a chunk
            chunks.append(chosen_sep.join(window).strip())

            # Build overlap: walk backwards until we fill chunk_overlap
            overlap: List[str] = []
            overlap_len = 0
            for p in reversed(window):
                if overlap_len + len(p) + len(chosen_sep) > chunk_overlap:
                    break
                overlap.insert(0, p)
                overlap_len += len(p) + len(chosen_sep)

            window = overlap
            window_len = overlap_len

        window.append(part)
        window_len += part_len

    if window:
        chunks.append(chosen_sep.join(window).strip())

    return [c for c in chunks if c]


def split_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split a list of page dicts into overlapping text chunks.

    Args:
        pages: Output from document_loader.load_document().

    Returns:
        List of chunk dicts with 'text', 'source', 'page' keys.
    """
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        raw = page.get("text", "").strip()
        if not raw:
            continue
        for chunk_text in _build_chunks(raw, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(
                {
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return chunks
