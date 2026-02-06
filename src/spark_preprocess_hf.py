# Loads AIG 2006 10-K from HuggingFace, splits it into chunks,
# and returns a Spark DataFrame ready for retrieval

import os
import re
import json
import pickle
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from pyspark.sql import SparkSession

# In-memory cache so we don't reload the document every time
_AIG_DOC_CACHE = {}


def split_into_chunks(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks, breaking on section headers first."""
    chunks = []

    # Split on 10-K section headers like "ITEM 1. BUSINESS"
    section_pattern = r'\n(ITEM\s+\d+[A-Z]?\.?\s*[A-Z][^\n]+)'
    sections = re.split(section_pattern, text, flags=re.IGNORECASE)

    current_chunk = ""
    chunk_id = 0

    for section in sections:
        if not section.strip():
            continue

        # If adding this section makes the chunk too big, save it and start fresh
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'char_start': sum(len(c['text']) for c in chunks),
            })
            chunk_id += 1
            # Keep some overlap for context between chunks
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""

        current_chunk += section

        # If the chunk is still too big, break it at natural points
        while len(current_chunk) > chunk_size:
            break_point = chunk_size
            for sep in ['\n\n', '\n', '. ', ', ']:
                pos = current_chunk.rfind(sep, 0, chunk_size)
                if pos > chunk_size // 2:
                    break_point = pos + len(sep)
                    break

            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk[:break_point].strip(),
                'char_start': sum(len(c['text']) for c in chunks),
            })
            chunk_id += 1
            current_chunk = current_chunk[break_point - overlap:]

    # Save whatever's left as the last chunk
    if current_chunk.strip():
        chunks.append({
            'chunk_id': chunk_id,
            'text': current_chunk.strip(),
            'char_start': sum(len(c['text']) for c in chunks),
        })

    return chunks


def _normalize_cik(value):
    """Strip leading zeros from a CIK identifier (e.g. '0000005272' -> '5272')."""
    if value is None:
        return ""
    return str(value).lstrip("0")


def _combine_sections(row):
    """Join all 10-K section fields into one big text string."""
    section_keys = [
        "section_1", "section_1A", "section_1B",
        "section_2", "section_3", "section_4",
        "section_5", "section_6", "section_7",
        "section_7A", "section_8", "section_9",
        "section_9A", "section_9B",
        "section_10", "section_11", "section_12",
        "section_13", "section_14", "section_15",
    ]

    parts = []
    for key in section_keys:
        value = row.get(key)
        if value:
            parts.append(f"{key.upper()}\n{value}")
    return "\n\n".join(parts).strip()


def _cached_file_for_url(cache_dir, url):
    """Look up the local cached file for a HuggingFace download URL."""
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


def _load_hf_year_2006_jsonl(cache_dir):
    """Load 2006 JSONL splits from the local HuggingFace cache."""
    urls = {
        "train": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/train.jsonl",
        "validation": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/validate.jsonl",
        "test": "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/2006/test.jsonl",
    }

    data_files = {}
    for split, url in urls.items():
        path = _cached_file_for_url(cache_dir, url)
        if not path:
            raise FileNotFoundError(f"Missing cached file for {split}. Expected in {cache_dir}/downloads.")
        data_files[split] = path

    ds_dict = load_dataset("json", data_files=data_files)
    return concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])


def load_aig_10k_2006(target_cik="5272"):
    """Load the AIG 2006 10-K from HuggingFace. Uses memory + disk cache."""
    global _AIG_DOC_CACHE

    # Check memory cache first
    if target_cik in _AIG_DOC_CACHE:
        return _AIG_DOC_CACHE[target_cik]

    cache_dir = os.path.join("data", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Check disk cache (pickle file from a previous run)
    pickle_path = os.path.join(cache_dir, f"aig_10k_2006_{target_cik}.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            doc = pickle.load(f)
        _AIG_DOC_CACHE[target_cik] = doc
        return doc

    # Cache miss — download from HuggingFace
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
        data_files[split_name] = hf_hub_download(
            repo_id="eloukas/edgar-corpus",
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    # Merge all splits and find the AIG filing
    ds_dict = load_dataset("json", data_files=data_files)
    merged = concatenate_datasets([ds_dict["train"], ds_dict["validation"], ds_dict["test"]])

    target_cik_norm = _normalize_cik(target_cik)
    matches = merged.filter(lambda row: _normalize_cik(row.get("cik")) == target_cik_norm)

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

    # Save to disk cache and memory cache
    with open(pickle_path, "wb") as f:
        pickle.dump(doc, f)
    _AIG_DOC_CACHE[target_cik] = doc

    return doc


def build_chunks_df_from_hf(spark, chunk_size=2000, overlap=200):
    """Load AIG 10-K from HuggingFace, chunk it, and return a Spark DataFrame."""
    doc = load_aig_10k_2006()
    chunks = split_into_chunks(doc["text"], chunk_size=chunk_size, overlap=overlap)

    rows = [
        {
            "year": int(doc["year"]),
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "char_start": c.get("char_start", 0),
        }
        for c in chunks
    ]

    return spark.createDataFrame(rows)
