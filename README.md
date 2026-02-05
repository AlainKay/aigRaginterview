# RAG Ground Truth Pipeline (Single Document)

Single-document RAG pipeline for the AIG 2006 10-K using **PySpark** and the Hugging Face dataset `eloukas/edgar-corpus`. This project satisfies the technical exercise requirements with 2 numeric variables, 1 categorical variable, and 5 observations each.

## Key Facts

| Item | Value |
|------|-------|
| Company | AIG |
| Filing type | 10-K |
| Dataset | [eloukas/edgar-corpus](https://huggingface.co/datasets/eloukas/edgar-corpus) |
| Implementation | PySpark |
| Document | AIG 2006 10-K (single document) |
| Accuracy | **100% (15/15)** |

## Variables Extracted

| Variable | Type | Description |
|----------|------|-------------|
| `fas133_fas52_amounts_included` | Numeric | FAS 133/FAS 52 amounts (millions USD) |
| `capital_markets_hedging_effect` | Numeric | Capital Markets hedging effect (millions USD) |
| `business_segments` | Categorical | Reportable business segments |

## Project Structure

```
ragGroundTruth/
├── config/
│   └── pipeline_config.yaml    # All configurable settings
├── data/
│   ├── ground_truth_2006_doc.json  # Manual ground truth (15 observations)
│   └── hf_cache/               # HuggingFace + pickle cache
├── results/
│   ├── extraction_results.json # Full results with metadata
│   └── extraction_results.csv  # Extracted vs ground truth comparison
├── src/
│   ├── config_loader.py        # YAML config loading
│   ├── llm_extractor.py        # Groq/Llama extraction
│   ├── main_pipeline_spark.py  # Pipeline orchestration
│   ├── spark_preprocess_hf.py  # HF loading + chunking
│   └── spark_retriever.py      # TF-IDF retrieval (Spark ML)
├── DESIGN.md                   # Architecture documentation
├── run.py                      # CLI entry point
└── requirements.txt            # Pinned dependencies
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your-key-here
```

## Run

```bash
# Full run (all 5 years: 2002-2006)
python run.py

# Quick test with single year
python run.py --years 2006

# Show config without running
python run.py --dry-run

# Verbose logging
python run.py --verbose
```

## Results

The pipeline extracts 3 variables × 5 years = 15 observations and compares against ground truth.

**Latest run: 100% accuracy (15/15 matches)**

| Year | FAS133/FAS52 | Capital Markets Hedging | Business Segments |
|------|--------------|-------------------------|-------------------|
| 2002 | -91 ✓ | 220 ✓ | 4 segments ✓ |
| 2003 | 78 ✓ | -1010 ✓ | 4 segments ✓ |
| 2004 | -140 ✓ | -122 ✓ | 4 segments ✓ |
| 2005 | -495 ✓ | 2010 ✓ | 4 segments ✓ |
| 2006 | 355 ✓ | -1820 ✓ | 4 segments ✓ |

## Pipeline Flow

```
HuggingFace Dataset → Spark Chunking → TF-IDF Index → Query Retrieval → LLM Extraction → Comparison
```

1. **Load**: AIG 2006 10-K from HuggingFace `eloukas/edgar-corpus`
2. **Chunk**: Split into overlapping chunks (2000 chars, 200 overlap)
3. **Index**: Build TF-IDF vectors with Spark ML
4. **Retrieve**: Find top-K relevant chunks per variable query
5. **Extract**: Use Groq/Llama to extract values from chunks
6. **Compare**: Validate against ground truth (1% tolerance for numeric, Jaccard for categorical)

## Design

See [DESIGN.md](DESIGN.md) for detailed architecture and methodology.
