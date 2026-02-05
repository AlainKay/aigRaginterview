# Design: Single-Document RAG Pipeline (AIG 2006 10-K)

## Goal

Build a retrieval‑augmented pipeline that extracts **2 numeric** and **1 categorical** variable from a **single HF document** and compares the results to ground truth. The example below focuses on `capital_markets_hedging_effect`.

## Data Source

- Dataset: Hugging Face `eloukas/edgar-corpus`
- Config: `year_2006`
- Document: AIG 2006 10‑K (CIK 5272)

The document is loaded from cached HF JSONL files and assembled by concatenating `section_1`–`section_15` fields into one text body.

## Variables

Numeric:
- `fas133_fas52_amounts_included`
- `capital_markets_hedging_effect`

Categorical:
- `business_segments`

Each numeric variable has **5 observations** (years 2002–2006). The categorical variable is repeated for each observation year to meet the 15‑row requirement.

## High‑Level Pipeline

1. Load the HF document (single 2006 10‑K).
2. Chunk the document into overlapping spans.
3. Build TF‑IDF index in Spark.
4. Retrieve top chunks per variable query.
5. Construct prompt with retrieved context.
6. Call LLM to extract value.
7. Compare with ground truth.
8. Write JSON + CSV results.

## Retrieval Design

- Retriever: Spark TF‑IDF (HashingTF + IDF)
- Similarity: cosine similarity computed in Spark UDF
- Query strategy: variable‑specific lexical queries to locate the exact sentence containing the 5‑year series

This is intentionally lexical because the target sentence is explicit and stable across the document.

## Extraction Design (Example: `capital_markets_hedging_effect`)

**Source sentence in the HF document**:

“For 2006, 2005, 2004, 2003 and 2002, respectively, the effect was $(1.82) billion, $2.01 billion, $(122) million, $(1.01) billion and $220 million in both revenues and operating income for Capital Markets.”

**Prompt behavior**:
- Anchor to the exact sentence above
- Select the **value for the target year**
- Convert billions to millions
- Preserve sign when value is in parentheses
- Output JSON only

**Why this works**:
- The sentence contains a **single 5‑year series** with one value per year.
- Retrieval targets “Capital Markets” and the “effect was … in both revenues and operating income” clause.
- The model maps year → position deterministically.

## Ground Truth Construction

Ground truth values are captured directly from the same sentence in the HF document and stored in:

- `data/ground_truth_2006_doc.json`

Each year has the numeric value in **millions** and a categorical list for `business_segments`.

## Evaluation

- Numeric match: relative error ≤ 1% (`numeric_tolerance` in config)
- Categorical match: Jaccard similarity ≥ 0.5
- Outputs:
  - `results/extraction_results.json`
  - `results/extraction_results.csv`

## Files That Implement This

- `src/main_pipeline_spark.py`: orchestration
- `src/spark_preprocess_hf.py`: HF loading + chunking
- `src/spark_retriever.py`: TF‑IDF retrieval
- `src/llm_extractor.py`: prompts + LLM calls
- `config/pipeline_config.yaml`: queries, model, thresholds
- `data/ground_truth_2006_doc.json`: ground truth

## Why Single‑Document

The assignment requires retrieval and ground truth from one document. Using the HF 2006 AIG 10‑K ensures both extraction and ground truth are sourced from the same text, with minimal ambiguity.
