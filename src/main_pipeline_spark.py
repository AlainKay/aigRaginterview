
# loads AIG 10-K from HuggingFace,
# retrieves chunks,
# extracts data with LLM,
# compares to ground truth

import os
import json
import logging
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from src.config_loader import load_config
from src.llm_extractor import run_extraction
from src.spark_retriever import SparkRetriever
from src.spark_preprocess_hf import build_chunks_df_from_hf

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
config = load_config()

TARGET_YEARS = config.get("target_years", [2002, 2003, 2004, 2005, 2006])
VARIABLE_QUERIES = config.get("variable_queries", {})
PATHS = config.get("paths", {})
COMPARISON = config.get("comparison", {})

# (variable key, comparison type, display name)
VARIABLES = [
    ("fas133_fas52_amounts_included",  "numeric",      "FAS133/FAS52 Amounts Included"),
    ("capital_markets_hedging_effect",  "numeric",      "Capital Markets Hedging Effect"),
    ("business_segments",               "categorical",  "Business Segments"),
]


# --- Comparison ---

def compare(extracted, ground_truth, var_type):
    """Compare an extracted value to ground truth. Returns a result dict."""

    if extracted is None:
        return {"match": False, "extracted": None, "ground_truth": ground_truth, "error": "extraction_failed"}

    if var_type == "numeric":
        tolerance = COMPARISON.get("numeric_tolerance", 0.01)
        try:
            ext_val = float(extracted)
            gt_val = float(ground_truth)
            if gt_val == 0:
                return {"match": ext_val == 0, "extracted": ext_val, "ground_truth": gt_val}
            rel_error = abs(ext_val - gt_val) / abs(gt_val)
            return {"match": rel_error <= tolerance, "extracted": ext_val, "ground_truth": gt_val, "relative_error": rel_error}
        except (TypeError, ValueError) as e:
            return {"match": False, "extracted": extracted, "ground_truth": ground_truth, "error": str(e)}

    if var_type == "categorical":
        threshold = COMPARISON.get("categorical_threshold", 0.5)
        try:
            ext_set = {s.lower().strip() for s in extracted}
            gt_set = {s.lower().strip() for s in ground_truth}
            shared = ext_set & gt_set
            all_items = ext_set | gt_set
            jaccard = len(shared) / len(all_items) if all_items else 0
            return {"match": jaccard >= threshold, "extracted": extracted, "ground_truth": ground_truth,
                    "jaccard": jaccard, "common": list(shared)}
        except Exception as e:
            return {"match": False, "extracted": extracted, "ground_truth": ground_truth, "error": str(e)}


# --- Helpers ---

def load_ground_truth():
    """Load expected values from the ground truth JSON file"""
    filepath = PATHS.get("ground_truth", "data/ground_truth_2006_doc.json")
    with open(filepath, "r") as f:
        return json.load(f)


def process_year(year, retriever, gt_lookup, retrieval_config):
    """Retrieve, extract, and compare for one year"""
    logger.info(f"Processing year {year}")

    # Retrieve relevant chunks
    retrieval_context = {}
    for variable, queries in VARIABLE_QUERIES.items():
        retrieval_context[variable] = retriever.retrieve_for_variable(
            variable, None, queries, top_k=retrieval_config.get("top_k", 5),
        )
    for name, chunks in retrieval_context.items():
        score = chunks[0]["score"] if chunks else 0
        logger.info(f"  {name}: {len(chunks)} chunks (top: {score:.4f})")

    # Extract with LLM
    logger.info("  Extracting with LLM...")
    extractions = run_extraction(retrieval_context, year)

    # Compare each variable
    gt_year = gt_lookup.get(year, {})
    comparisons = {}

    for var_name, var_type, display_name in VARIABLES:
        if var_name not in extractions:
            continue
        ext_val = extractions[var_name].get("value")
        gt_val = gt_year.get(var_name, {}).get("value")
        result = compare(ext_val, gt_val, var_type)
        comparisons[var_name] = result

        status = "PASS" if result["match"] else "FAIL"
        logger.info(f"  {display_name}: {ext_val} vs {gt_val} [{status}]")

    return {"year": year, "extractions": extractions, "comparisons": comparisons}


def calculate_accuracy(results):
    """Count matches and return (total, matches, accuracy)"""
    every_comparison = []
    for year_result in results:
        for comp in year_result["comparisons"].values():
            every_comparison.append(comp)
    total = len(every_comparison)
    matches = sum(1 for c in every_comparison if c.get("match"))
    accuracy = matches / total if total > 0 else 0
    return total, matches, accuracy


def save_results(summary, results):
    """Write results to JSON and CSV"""
    json_path = PATHS.get("results_json", "results/extraction_results.json")
    csv_path = PATHS.get("results_csv", "results/extraction_results.csv")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"JSON saved: {json_path}")

    rows = []
    for year_result in results:
        for var_name, _, _ in VARIABLES:
            if var_name not in year_result["comparisons"]:
                continue
            comp = year_result["comparisons"][var_name]
            ext = year_result["extractions"].get(var_name, {})
            rows.append({
                "year": year_result["year"],
                "variable": var_name,
                "extracted": ext.get("value"),
                "expected": comp.get("ground_truth"),
                "match": comp.get("match", False),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")


# --- Main pipeline ---

def run_pipeline():
    """Load -> chunk -> index -> retrieve -> extract -> compare"""
    logger.info("=" * 60)
    logger.info("AIG 10-K RAG Extraction Pipeline")
    logger.info("=" * 60)

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set — add it to your .env file")
        return None

    retrieval_config = config.get("retrieval", {})

    # Step 1: Start Spark
    logger.info("[1/5] Starting Spark...")
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if os.path.exists(venv_python):
        os.environ["PYSPARK_PYTHON"] = venv_python
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    spark = (
        SparkSession.builder.appName("AIG-RAG-Pipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logger.org.apache.spark=WARN")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Step 2: Load and chunk
    logger.info("[2/5] Loading + chunking AIG 2006 10-K...")
    chunks_df = build_chunks_df_from_hf(spark)

    # Step 3: Build search index
    logger.info("[3/5] Building TF-IDF index...")
    retriever = SparkRetriever(spark)
    retriever.build_index(chunks_df)

    # Step 4: Load ground truth
    logger.info("[4/5] Loading ground truth...")
    ground_truth = load_ground_truth()
    gt_lookup = {item["year"]: item["variables"] for item in ground_truth["ground_truth"]}

    # Step 5: Extract and compare each year
    logger.info("[5/5] Extracting for each year...")
    results = [process_year(year, retriever, gt_lookup, retrieval_config) for year in TARGET_YEARS]

    total, matches, accuracy = calculate_accuracy(results)

    summary = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "years": TARGET_YEARS,
            "total": total,
            "matches": matches,
            "accuracy": accuracy,
        },
        "results_by_year": results,
    }

    save_results(summary, results)

    logger.info("=" * 60)
    logger.info(f"DONE — {matches}/{total} matched ({accuracy:.0%} accuracy)")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_pipeline()
