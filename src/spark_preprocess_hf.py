"""
Spark preprocessing for a single 10-K document loaded from Hugging Face.

This module handles loading the AIG 2006 10-K filing from the Hugging Face
edgar-corpus dataset, combining its sections, and creating a Spark DataFrame
of text chunks for retrieval.

Source dataset:
    eloukas/edgar-corpus (config: year_2006)
    https://huggingface.co/datasets/eloukas/edgar-corpus
"""

from typing import Dict, List, Any

import os
import re
import json
import pickle
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset
from pyspark.sql import SparkSession, DataFrame

# Module-level cache for parsed AIG document
_AIG_DOC_CACHE: Dict[str, Any] = {}


def split_into_chunks(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[Dict]:
    """
    Split text into overlapping chunks for retrieval indexing.

    Uses a smart chunking strategy that:
    1. First attempts to split on 10-K section headers (ITEM 1, ITEM 2, etc.)
    2. For oversized sections, splits on sentence/paragraph boundaries
    3. Maintains overlap between chunks for context preservation

    Args:
        text: Full document text to split.
        chunk_size: Target maximum characters per chunk. Default 2000.
        overlap: Characters to overlap between consecutive chunks. Default 200.

    Returns:
        List of chunk dictionaries, each containing:
            - chunk_id (int): Sequential identifier starting at 0
            - text (str): Chunk content
            - char_start (int): Starting position in original document
    """
    chunks = []  # Will hold all chunk dictionaries

    # STEP 1: Try to split by SEC 10-K section headers (e.g., "ITEM 1. BUSINESS")
    # This regex captures lines like "ITEM 1A. RISK FACTORS"
    section_pattern = r'\n(ITEM\s+\d+[A-Z]?\.?\s*[A-Z][^\n]+)'
    sections = re.split(section_pattern, text, flags=re.IGNORECASE)

    current_chunk = ""  # Buffer to accumulate text until it reaches chunk_size
    chunk_id = 0  # Counter for unique chunk identifiers

    # STEP 2: Process each section and build chunks
    for section in sections:
        if not section.strip():
            continue  # Skip empty sections

        # If adding this section exceeds chunk_size, save current chunk and start new one
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'char_start': sum(len(c['text']) for c in chunks),
            })
            chunk_id += 1
            # Keep some overlap for context continuity between chunks
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""

        current_chunk += section  # Add section to current chunk buffer

        # STEP 3: If current chunk is too large, split it at natural break points
        while len(current_chunk) > chunk_size:
            # Find the best break point (prefer paragraph > line > sentence > comma)
            break_point = chunk_size
            for sep in ['\n\n', '\n', '. ', ', ']:  # Priority order of separators
                pos = current_chunk.rfind(sep, 0, chunk_size)  # Search backwards
                if pos > chunk_size // 2:  # Only use if at least halfway through
                    break_point = pos + len(sep)
                    break

            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk[:break_point].strip(),
                'char_start': sum(len(c['text']) for c in chunks),
            })
            chunk_id += 1
            # Keep overlap from the end of saved chunk for context continuity
            current_chunk = current_chunk[break_point - overlap:]

    # STEP 4: Save any remaining text as the final chunk
    if current_chunk.strip():
        chunks.append({
            'chunk_id': chunk_id,
            'text': current_chunk.strip(),
            'char_start': sum(len(c['text']) for c in chunks),
        })

    return chunks


def _normalize_cik(value: Any) -> str:
    """
    Normalize a CIK (Central Index Key) by stripping leading zeros.

    SEC CIKs are 10-digit identifiers that may have leading zeros. This
    function normalizes them for consistent comparison.

    Args:
        value: CIK value (string or integer), may be None.

    Returns:
        Normalized CIK string without leading zeros, or empty string if None.

    Example:
        >>> _normalize_cik("0000005272")
        '5272'
    """
    if value is None:
        return ""
    return str(value).lstrip("0")


def _combine_sections(row: Dict) -> str:
    """
    Combine all 10-K section fields into a single text document.

    SEC 10-K filings have standardized sections (Item 1-15). This function
    concatenates all available sections with headers for downstream processing.

    Args:
        row: Dictionary from the HuggingFace dataset containing section_1
             through section_15 fields.

    Returns:
        Single string with all sections concatenated, each prefixed with
        its section name (e.g., "SECTION_1\\n<content>").
    """
    section_keys = [
        "section_1", "section_1A", "section_1B",
        "section_2", "section_3", "section_4",
        "section_5", "section_6", "section_7",
        "section_7A", "section_8", "section_9",
        "section_9A", "section_9B",
        "section_10", "section_11", "section_12",
        "section_13", "section_14", "section_15",
    ]

    parts: List[str] = []
    for key in section_keys:
        value = row.get(key)
        if value:
            parts.append(f"{key.upper()}\n{value}")
    return "\n\n".join(parts).strip()


def _cached_file_for_url(cache_dir: str, url: str) -> str:
    """
    Find the local cached file path for a HuggingFace dataset URL.

    HuggingFace datasets library caches downloaded files with metadata.
    This function looks up the local path for a given remote URL.

    Args:
        cache_dir: Path to the HuggingFace cache directory.
        url: Remote URL of the dataset file to find.

    Returns:
        Local file path if found, empty string otherwise.
    """
    downloads_dir = Path(cache_dir) / "downloads"
    if not downloads_dir.exists():
        return ""

    for meta in downloads_dir.glob("*.json"):
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        if data.get("url") == url:
            bin_path = meta.with_suffix("")
            return str(bin_path) if bin_path.exists() else ""
    return ""


def _load_hf_year_2006_jsonl(cache_dir: str) -> Dataset:
    """
    Load HuggingFace JSONL files for year 2006 from the local cache.

    Loads train, validation, and test splits from cached JSONL files and
    concatenates them into a single dataset.

    Args:
        cache_dir: Path to the directory containing cached HF downloads.

    Returns:
        Concatenated HuggingFace Dataset containing all 2006 filings.

    Raises:
        FileNotFoundError: If any required split file is not in the cache.
    """
    urls = {
        "train": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/train.jsonl",
        "validation": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/validate.jsonl",
        "test": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/test.jsonl",
    }

    data_files = {}
    for split, url in urls.items():
        path = _cached_file_for_url(cache_dir, url)
        if not path:
            raise FileNotFoundError(
                f"Missing cached file for {split}. Expected HF download in {cache_dir}/downloads."
            )
        data_files[split] = path

    ds_dict = load_dataset("json", data_files=data_files)
    return concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])


def load_aig_10k_2006(target_cik: str = "5272") -> Dict[str, Any]:
    """
    Load the AIG 2006 10-K filing from HuggingFace and combine sections.

    Uses a two-level cache: in-memory for repeated calls within the same process,
    and a pickle file on disk to skip JSONL parsing on subsequent runs.

    Args:
        target_cik: SEC Central Index Key for the company. Default "5272" is AIG.

    Returns:
        Dictionary containing:
            - year (int): Filing year (2006)
            - company (str): Company name
            - cik (str): Central Index Key
            - file_name (str): Original SEC filing filename
            - text (str): Combined text from all sections

    Raises:
        ValueError: If no matching 10-K is found or sections are empty.
    """
    global _AIG_DOC_CACHE

    # CACHE LEVEL 1: Check in-memory cache (fastest, within same process)
    if target_cik in _AIG_DOC_CACHE:
        return _AIG_DOC_CACHE[target_cik]

    cache_dir = os.path.join("data", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # CACHE LEVEL 2: Check disk cache (persists across runs, avoids re-parsing)
    pickle_path = os.path.join(cache_dir, f"aig_10k_2006_{target_cik}.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            doc = pickle.load(f)
        _AIG_DOC_CACHE[target_cik] = doc  # Also store in memory for future calls
        return doc

    # CACHE MISS: Download JSONL files from HuggingFace and parse
    # Downloads train/validation/test splits (~1.5GB total) on first run
    print("  Downloading AIG 10-K from HuggingFace (first run only)...")
    from huggingface_hub import hf_hub_download

    downloads_dir = os.path.join(cache_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Download the three JSONL splits
    splits = {
        "train": "2006/train.jsonl",
        "validation": "2006/validate.jsonl",
        "test": "2006/test.jsonl",
    }
    data_files = {}
    for split_name, filename in splits.items():
        local_path = hf_hub_download(
            repo_id="eloukas/edgar-corpus",
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        data_files[split_name] = local_path

    # Load and merge all splits
    ds_dict = load_dataset("json", data_files=data_files)
    merged = concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])

    target_cik_norm = _normalize_cik(target_cik)

    def _is_aig_10k(row):
        cik = _normalize_cik(row.get("cik"))
        return cik == target_cik_norm

    matches = merged.filter(_is_aig_10k)
    if len(matches) == 0:
        raise ValueError("No matching AIG 10-K found in year_2006 dataset.")

    row = matches[0]
    text = _combine_sections(row)
    if not text:
        raise ValueError("Found AIG 10-K but sections are empty.")

    doc = {
        "year": int(row.get("year", 2006)),
        "company": row.get("company"),
        "cik": row.get("cik"),
        "file_name": row.get("file_name"),
        "text": text,
    }

    # Save to both caches for future runs
    with open(pickle_path, "wb") as f:
        pickle.dump(doc, f)  # Disk cache (persists across runs)
    _AIG_DOC_CACHE[target_cik] = doc  # Memory cache (fast access within process)

    return doc


def build_chunks_df_from_hf(
    spark: SparkSession,
    chunk_size: int = 2000,
    overlap: int = 200
) -> DataFrame:
    """
    Load AIG 10-K from HuggingFace and return a Spark DataFrame of chunks.

    Combines document loading, text chunking, and Spark DataFrame creation
    into a single convenient function for the pipeline.

    Args:
        spark: Active SparkSession instance.
        chunk_size: Maximum characters per chunk. Default 2000.
        overlap: Character overlap between consecutive chunks for context
                 preservation. Default 200.

    Returns:
        Spark DataFrame with columns:
            - year (int): Document year
            - chunk_id (int): Sequential chunk identifier
            - text (str): Chunk text content
            - char_start (int): Starting character position in original doc

    Example:
        >>> spark = SparkSession.builder.appName("RAG").getOrCreate()
        >>> chunks_df = build_chunks_df_from_hf(spark)
        >>> chunks_df.count()  # e.g., 450 chunks
    """
    doc = load_aig_10k_2006()
    chunks = split_into_chunks(doc["text"], chunk_size=chunk_size, overlap=overlap)

    rows = []
    for c in chunks:
        rows.append(
            {
                "year": int(doc["year"]),
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "char_start": c.get("char_start", 0),
            }
        )

    return spark.createDataFrame(rows)
