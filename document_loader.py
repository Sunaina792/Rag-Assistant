"""
document_loader.py – Handles loading and text extraction from PDF and TXT files.
"""

import io
from typing import List, Dict, Any

import pypdf


def load_pdf(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file.

    Args:
        file_bytes: Raw bytes of the PDF file.
        file_name:  Original file name (used as metadata).

    Returns:
        A list of dicts, each with 'text', 'source', and 'page' keys.
    """
    pages = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(
                    {
                        "text": text,
                        "source": file_name,
                        "page": page_num,
                    }
                )
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{file_name}': {e}") from e
    return pages


def load_txt(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a plain-text file.

    Args:
        file_bytes: Raw bytes of the TXT file.
        file_name:  Original file name (used as metadata).

    Returns:
        A list with a single dict containing 'text', 'source', and 'page'.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read TXT '{file_name}': {e}") from e

    if not text:
        return []

    return [{"text": text, "source": file_name, "page": 1}]


def load_document(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Dispatch to the correct loader based on file extension.

    Supported extensions: .pdf, .txt

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name:  Original file name.

    Returns:
        A list of page dicts.

    Raises:
        ValueError: If the file type is unsupported.
    """
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        return load_pdf(file_bytes, file_name)
    elif lower_name.endswith(".txt"):
        return load_txt(file_bytes, file_name)
    else:
        raise ValueError(
            f"Unsupported file type for '{file_name}'. Only PDF and TXT are supported."
        )

