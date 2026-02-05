# Tools & Libraries Reference

This document lists all tools and libraries used in the RAG Ground Truth Pipeline, organized by purpose and the files where they are used.

---

## 1. Data Processing & Manipulation

### pandas
- **Purpose**: DataFrame operations for CSV output
- **Used in**: `src/main_pipeline_spark.py`
- **What it does**: Converts extraction results to a CSV file for easy spreadsheet analysis

### numpy
- **Purpose**: Numerical operations (dependency for pandas/pyspark)
- **Used in**: Indirect dependency
- **What it does**: Provides efficient array operations under the hood

---

## 2. Big Data Processing (PySpark)

### pyspark
- **Purpose**: Distributed data processing using Apache Spark
- **Used in**: `src/spark_retriever.py`, `src/spark_preprocess_hf.py`, `src/main_pipeline_spark.py`
- **Components used**:
  - `SparkSession` - Creates and manages the Spark application
  - `DataFrame` - Distributed dataset for chunk storage
  - `functions (F)` - Column operations (filter, select, orderBy)
  - `Pipeline` - Chains ML transformers together
  - `RegexTokenizer` - Splits text into tokens using regex
  - `HashingTF` - Converts tokens to term frequency vectors
  - `IDF` - Applies Inverse Document Frequency weighting
  - `UDF (User Defined Function)` - Custom Python functions in Spark

**Why PySpark?**: The assignment required implementing retrieval using PySpark ML for TF-IDF vectorization.

---

## 3. Machine Learning (Spark ML)

### pyspark.ml.feature
- **Purpose**: Text vectorization for TF-IDF retrieval
- **Used in**: `src/spark_retriever.py`
- **Components**:

| Component | Purpose |
|-----------|---------|
| `RegexTokenizer` | Splits text into words using `\W+` pattern |
| `HashingTF` | Hashes tokens to fixed-size frequency vectors (262,144 features) |
| `IDF` | Weights terms by inverse document frequency (rare words score higher) |

---

## 4. LLM Integration

### groq
- **Purpose**: API client for Groq cloud (hosts Llama models)
- **Used in**: `src/llm_extractor.py`
- **What it does**:
  - Sends prompts to Llama 3.1-8b-instant model
  - Returns structured JSON extractions
  - Handles rate limiting and retries

### python-dotenv
- **Purpose**: Load environment variables from `.env` file
- **Used in**: `src/llm_extractor.py`, `src/main_pipeline_spark.py`
- **What it does**: Securely loads `GROQ_API_KEY` without hardcoding secrets

---

## 5. Data Sources

### datasets (HuggingFace)
- **Purpose**: Load SEC filings from HuggingFace Hub
- **Used in**: `src/spark_preprocess_hf.py`
- **Functions used**:
  - `load_dataset()` - Loads JSONL files from cache
  - `concatenate_datasets()` - Merges train/validation/test splits
  - `Dataset.filter()` - Finds AIG 10-K by CIK number

### huggingface-hub
- **Purpose**: Dependency for datasets library
- **Used in**: Indirect dependency
- **What it does**: Handles authentication and caching for HuggingFace datasets

---

## 6. Configuration & Serialization

### pyyaml
- **Purpose**: Parse YAML configuration files
- **Used in**: `src/config_loader.py`
- **What it does**: Loads `config/pipeline_config.yaml` into Python dictionaries

### json (stdlib)
- **Purpose**: JSON parsing and serialization
- **Used in**: `src/llm_extractor.py`, `src/main_pipeline_spark.py`, `src/spark_preprocess_hf.py`
- **What it does**:
  - Parses LLM JSON responses
  - Saves extraction results to JSON
  - Loads ground truth data

### pickle (stdlib)
- **Purpose**: Binary serialization for caching
- **Used in**: `src/spark_preprocess_hf.py`
- **What it does**: Caches parsed AIG document to disk to avoid re-parsing on subsequent runs

---

## 7. Standard Library Utilities

### re (regex)
- **Purpose**: Pattern matching for text chunking
- **Used in**: `src/spark_preprocess_hf.py`
- **What it does**: Splits document by SEC 10-K section headers (e.g., "ITEM 1. BUSINESS")

### os
- **Purpose**: File system operations and environment variables
- **Used in**: All source files
- **What it does**: Path handling, directory creation, environment variable access

### time
- **Purpose**: Rate limiting and delays
- **Used in**: `src/llm_extractor.py`
- **What it does**: Implements delays between API calls and retry backoff

### argparse
- **Purpose**: Command-line argument parsing
- **Used in**: `run.py`
- **What it does**: Handles `--config`, `--years`, `--verbose`, `--dry-run` flags

### logging
- **Purpose**: Structured logging output
- **Used in**: `src/main_pipeline_spark.py`, `run.py`
- **What it does**: Provides timestamped log messages with severity levels

### dataclasses
- **Purpose**: Structured data containers
- **Used in**: `src/main_pipeline_spark.py`
- **What it does**: Defines `ComparisonResult` class for storing extraction comparisons

### datetime
- **Purpose**: Timestamp generation
- **Used in**: `src/main_pipeline_spark.py`
- **What it does**: Records pipeline run timestamps in results

### pathlib
- **Purpose**: Object-oriented path handling
- **Used in**: `src/spark_preprocess_hf.py`
- **What it does**: Navigates HuggingFace cache directory structure

---

## 8. File-by-File Summary

| File | Key Libraries |
|------|---------------|
| `run.py` | argparse, logging, sys |
| `src/config_loader.py` | pyyaml, os |
| `src/spark_preprocess_hf.py` | datasets, pyspark, pickle, re |
| `src/spark_retriever.py` | pyspark.ml (RegexTokenizer, HashingTF, IDF, Pipeline) |
| `src/llm_extractor.py` | groq, python-dotenv, json, time |
| `src/main_pipeline_spark.py` | pyspark, pandas, logging, dataclasses |

---

## 9. Architecture Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HuggingFace    │     │    PySpark ML    │     │   Groq API      │
│  datasets       │────▶│  TF-IDF Pipeline │────▶│  (Llama 3.1)    │
│  (data source)  │     │  (retrieval)     │     │  (extraction)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  pickle         │     │  Spark DataFrame │     │  JSON parsing   │
│  (caching)      │     │  (chunk storage) │     │  (results)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 10. Why These Choices?

| Tool | Why Chosen |
|------|------------|
| **PySpark** | Assignment requirement for distributed processing |
| **Groq/Llama** | Free API, fast inference, open-source model |
| **HuggingFace** | Official source for edgar-corpus dataset |
| **TF-IDF** | Simple, effective lexical retrieval for exact phrase matching |
| **YAML config** | Separates settings from code, easy to modify |
| **Pickle caching** | Avoids slow HuggingFace parsing on repeated runs |
