"""
Spark-based RAG Pipeline (single-document) using Hugging Face edgar-corpus.

This module orchestrates the complete RAG pipeline for extracting financial
data from AIG's 2006 10-K SEC filing. It coordinates document loading,
chunking, TF-IDF indexing, retrieval, LLM extraction, and ground truth
comparison.

Pipeline Steps:
    1. Load AIG 2006 10-K from Hugging Face edgar-corpus dataset
    2. Split document into overlapping chunks using Spark
    3. Build TF-IDF index with Spark ML pipeline
    4. Retrieve relevant chunks for each variable query
    5. Extract values using LLM (Groq API)
    6. Compare extractions against ground truth
    7. Generate JSON and CSV results report

Usage:
    This module is typically invoked via run.py, but can also be run directly:

    >>> from main_pipeline_spark import run_pipeline
    >>> results = run_pipeline()
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

import logging

from src.config_loader import load_config
from src.llm_extractor import LLMExtractor
from src.spark_retriever import SparkRetriever
from src.spark_preprocess_hf import build_chunks_df_from_hf

from pyspark.sql import SparkSession

# Initialize environment and configuration
load_dotenv()  # Load API keys from .env file
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
CONFIG = load_config("config/pipeline_config.yaml")  # Load settings from YAML config

# Pipeline configuration from YAML (can be overridden by command line args)
TARGET_YEARS: List[int] = CONFIG.get("target_years", [2002, 2003, 2004, 2005, 2006])  # Years to extract data for
VARIABLE_QUERIES: Dict = CONFIG.get("variable_queries", {})  # Search queries for each variable
PATHS: Dict = CONFIG.get("paths", {})  # File paths for inputs/outputs
COMPARISON: Dict = CONFIG.get("comparison", {})  # Tolerance settings for matching


@dataclass
class ComparisonResult:
    """
    Container for extraction-to-ground-truth comparison results.

    Stores the comparison outcome along with detailed metrics for either
    numeric (relative error) or categorical (Jaccard similarity) variables.

    Attributes:
        match: Whether the extraction matches ground truth within tolerance.
        extracted: The value extracted by the LLM.
        ground_truth: The expected value from ground truth dataset.
        extracted_source_chunks: Chunk IDs used for extraction.
        gt_source_chunks: Chunk IDs referenced in ground truth (if any).
        error_type: Error description if extraction failed (optional).
        relative_error: Relative error for numeric comparisons (optional).
        jaccard_similarity: Jaccard index for categorical comparisons (optional).
        common_segments: Overlapping segments for categorical data (optional).
    """
    match: bool
    extracted: Any
    ground_truth: Any
    extracted_source_chunks: List[int]
    gt_source_chunks: List[int]
    error_type: str = None
    relative_error: float = None
    jaccard_similarity: float = None
    common_segments: List[str] = None

    def to_dict(self) -> Dict:
        result = {
            "match": self.match,
            "extracted": self.extracted,
            "ground_truth": self.ground_truth,
            "extracted_source_chunks": self.extracted_source_chunks,
            "gt_source_chunks": self.gt_source_chunks,
        }
        if self.error_type:
            result["error_type"] = self.error_type
        if self.relative_error is not None:
            result["relative_error"] = self.relative_error
        if self.jaccard_similarity is not None:
            result["jaccard_similarity"] = self.jaccard_similarity
        if self.common_segments is not None:
            result["common_segments"] = self.common_segments
        return result


def compare_numeric(
    extracted: Any,
    ground_truth: Any,
    ext_chunks: List[int],
    gt_chunks: List[int]
) -> ComparisonResult:
    """
    Compare extracted numeric value against ground truth with tolerance.

    Uses relative error to determine if values match, allowing for small
    differences due to rounding or unit conversion (e.g., $2.01B vs $2010M).

    Args:
        extracted: Value extracted by the LLM (may be None on failure).
        ground_truth: Expected value from ground truth dataset.
        ext_chunks: List of chunk IDs used during extraction.
        gt_chunks: List of chunk IDs referenced in ground truth.

    Returns:
        ComparisonResult with match=True if relative error <= tolerance
        (default 1%), or match=False otherwise.
    """
    tolerance = COMPARISON.get("numeric_tolerance", 0.01)

    if extracted is None:
        return ComparisonResult(
            match=False,
            extracted=None,
            ground_truth=ground_truth,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            error_type="extraction_failed",
        )

    try:
        ext_val = float(extracted)
        gt_val = float(ground_truth)

        # Handle edge case: if ground truth is 0, exact match required
        if gt_val == 0:
            match = ext_val == 0
            rel_error = None
        else:
            # Relative error = |extracted - expected| / |expected|
            # Example: extracted=1820, expected=1800 → error = 20/1800 = 1.1%
            rel_error = abs(ext_val - gt_val) / abs(gt_val)
            match = rel_error <= tolerance  # Match if error within tolerance (default 1%)

        return ComparisonResult(
            match=match,
            extracted=ext_val,
            ground_truth=gt_val,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            relative_error=rel_error,
        )

    except (TypeError, ValueError) as e:
        return ComparisonResult(
            match=False,
            extracted=extracted,
            ground_truth=ground_truth,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            error_type=str(e),
        )


def compare_categorical(
    extracted: Any,
    ground_truth: Any,
    ext_chunks: List[int],
    gt_chunks: List[int]
) -> ComparisonResult:
    """
    Compare extracted categorical values against ground truth using Jaccard similarity.

    Computes the Jaccard index (intersection over union) between the extracted
    and ground truth sets, treating values as case-insensitive strings.

    Args:
        extracted: List of values extracted by the LLM (e.g., segment names).
        ground_truth: Expected list of values from ground truth dataset.
        ext_chunks: List of chunk IDs used during extraction.
        gt_chunks: List of chunk IDs referenced in ground truth.

    Returns:
        ComparisonResult with match=True if Jaccard similarity >= threshold
        (default 0.5), along with the computed similarity and common elements.
    """
    threshold = COMPARISON.get("categorical_threshold", 0.5)

    if extracted is None:
        return ComparisonResult(
            match=False,
            extracted=None,
            ground_truth=ground_truth,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            error_type="extraction_failed",
        )

    try:
        # Convert to lowercase sets for case-insensitive comparison
        ext_set = {s.lower().strip() for s in extracted}
        gt_set = {s.lower().strip() for s in ground_truth}

        # Jaccard similarity = |intersection| / |union|
        # Example: extracted={"A","B","C"}, expected={"A","B","D"} → Jaccard = 2/4 = 0.5
        intersection = ext_set & gt_set  # Items in both sets
        union = ext_set | gt_set  # All unique items
        jaccard = len(intersection) / len(union) if union else 0

        return ComparisonResult(
            match=jaccard >= threshold,
            extracted=extracted,
            ground_truth=ground_truth,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            jaccard_similarity=jaccard,
            common_segments=list(intersection),
        )

    except Exception as e:
        return ComparisonResult(
            match=False,
            extracted=extracted,
            ground_truth=ground_truth,
            extracted_source_chunks=ext_chunks,
            gt_source_chunks=gt_chunks,
            error_type=str(e),
        )


# Registry of variables to extract with their comparison functions
# - Numeric variables: use relative error comparison (tolerance-based)
# - Categorical variables: use Jaccard similarity comparison (set overlap)
VARIABLES = {
    "fas133_fas52_amounts_included": {
        "compare_fn": compare_numeric,  # Numeric: amounts in millions USD
        "display_name": "FAS133/FAS52 Amounts Included",
    },
    "capital_markets_hedging_effect": {
        "compare_fn": compare_numeric,  # Numeric: effect in millions USD
        "display_name": "Capital Markets Hedging Effect",
    },
    "business_segments": {
        "compare_fn": compare_categorical,  # Categorical: list of segment names
        "display_name": "Business Segments",
    },
}


def load_ground_truth() -> Dict:
    """
    Load the ground truth dataset from JSON file.

    Reads the hand-curated ground truth values that were extracted manually
    from the source document for validation purposes.

    Returns:
        Dictionary containing:
            - metadata: Source information (company, filing type, etc.)
            - ground_truth: List of year records, each with variables dict

    Raises:
        FileNotFoundError: If the ground truth file path is invalid.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    filepath = PATHS.get("ground_truth", "data/ground_truth_2006_doc.json")
    with open(filepath, "r") as f:
        return json.load(f)


def process_year(
    year: int,
    retriever: SparkRetriever,
    extractor: LLMExtractor,
    gt_lookup: Dict[int, Dict],
    retrieval_config: Dict
) -> Dict:
    """
    Process a single observation year: retrieve, extract, and compare.

    Orchestrates the retrieval of relevant chunks for each variable,
    LLM-based extraction of values, and comparison against ground truth.

    Args:
        year: The observation year to process (e.g., 2006).
        retriever: Initialized SparkRetriever with built index.
        extractor: Initialized LLMExtractor for API calls.
        gt_lookup: Dictionary mapping years to their ground truth variables.
        retrieval_config: Configuration dict with top_k and other settings.

    Returns:
        Dictionary containing:
            - year: The processed year
            - extractions: Dict of variable name -> extraction result
            - comparisons: Dict of variable name -> comparison result
    """
    logger.info(f"Processing Observation Year {year}")

    retrieval_context = {}
    for variable, queries in VARIABLE_QUERIES.items():
        retrieval_context[variable] = retriever.retrieve_for_variable(
            variable,
            None,  # do not filter by year; all data is from 2006 doc
            queries,
            top_k=retrieval_config.get("top_k", 5),
        )

    for var_name, chunks in retrieval_context.items():
        score = chunks[0]["score"] if chunks else 0
        logger.info(f"  {var_name}: {len(chunks)} chunks (top score: {score:.4f})")

    logger.info("  Extracting with LLM...")
    extractions = extractor.extract_all_variables(retrieval_context, year)

    gt_year = gt_lookup.get(year, {})

    comparisons = {}
    for var_name, var_config in VARIABLES.items():
        if var_name not in extractions:
            continue

        ext_data = extractions[var_name]
        gt_data = gt_year.get(var_name, {})

        ext_val = ext_data.get("value")
        ext_chunks = ext_data.get("source_chunk_ids", [])
        gt_val = gt_data.get("value")
        gt_chunks = gt_data.get("source_chunk_ids", [])

        result = var_config["compare_fn"](ext_val, gt_val, ext_chunks, gt_chunks)
        comparisons[var_name] = result.to_dict()

        status = "✓" if result.match else "✗"
        if result.jaccard_similarity is not None:
            logger.info(
                f"  {var_config['display_name']}: Jaccard={result.jaccard_similarity:.2f} [{status}]"
            )
        else:
            logger.info(f"  {var_config['display_name']}: {ext_val} vs {gt_val} [{status}]")

    return {"year": year, "extractions": extractions, "comparisons": comparisons}


def calculate_accuracy(results: List[Dict]) -> Tuple[int, int, float]:
    """
    Calculate overall extraction accuracy across all years and variables.

    Counts the total number of comparisons and successful matches to
    compute an accuracy percentage.

    Args:
        results: List of year result dictionaries from process_year().

    Returns:
        Tuple of (total_comparisons, successful_matches, accuracy_ratio).
        Accuracy is 0.0 if no comparisons were made.
    """
    total = 0
    matches = 0
    for year_result in results:
        for comparison in year_result["comparisons"].values():
            total += 1
            if comparison.get("match"):
                matches += 1
    accuracy = matches / total if total > 0 else 0
    return total, matches, accuracy


def save_results(summary: Dict, results: List[Dict]) -> None:
    """
    Save extraction results to JSON and CSV files.

    Creates both a detailed JSON file with full extraction metadata and
    a flattened CSV file suitable for spreadsheet analysis.

    Args:
        summary: Complete pipeline summary including accuracy metrics.
        results: List of year result dictionaries with extractions/comparisons.

    Returns:
        None. Files are written to paths specified in config.

    Side Effects:
        - Creates results directory if it doesn't exist
        - Writes extraction_results.json with full details
        - Writes extraction_results.csv with flattened data
    """
    json_path = PATHS.get("results_json", "results/extraction_results.json")
    csv_path = PATHS.get("results_csv", "results/extraction_results.csv")

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to: {json_path}")

    rows = []
    for year_result in results:
        year = year_result["year"]
        for var_name in VARIABLES.keys():
            if var_name not in year_result["comparisons"]:
                continue

            comp = year_result["comparisons"][var_name]
            ext = year_result["extractions"].get(var_name, {})

            rows.append(
                {
                    "year": year,
                    "variable": var_name,
                    "extracted_value": ext.get("value"),
                    "ground_truth_value": comp.get("ground_truth"),
                    "match": comp.get("match"),
                    "confidence": ext.get("confidence"),
                    "extracted_source_chunks": str(comp.get("extracted_source_chunks", [])),
                    "gt_source_chunks": str(comp.get("gt_source_chunks", [])),
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Results CSV saved to: {csv_path}")


def run_pipeline() -> Dict:
    """
    Execute the complete RAG extraction pipeline.

    This is the main entry point that orchestrates all pipeline stages:
    Spark initialization, document loading, indexing, extraction, and
    result generation.

    Returns:
        Dictionary containing:
            - pipeline_run: Metadata (timestamp, years, accuracy)
            - results_by_year: List of per-year extraction results

        Returns None if GROQ_API_KEY is not set.

    Raises:
        Exception: Various exceptions may propagate from Spark, Groq API,
                   or file I/O operations.

    Example:
        >>> result = run_pipeline()
        >>> print(f"Accuracy: {result['pipeline_run']['accuracy']:.1%}")
    """
    logger.info("=" * 60)
    logger.info("AIG 10-K RAG Extraction Pipeline (Spark + HF, single-doc)")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set")
        logger.error("Set it in .env file: GROQ_API_KEY=your-key-here")
        return None

    llm_config = CONFIG.get("llm", {})
    retrieval_config = CONFIG.get("retrieval", {})

    # PIPELINE STEP 1: Initialize Spark
    logger.info("[1/5] Initializing Spark session...")
    # Tell PySpark to use the same Python as our virtual environment
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if os.path.exists(venv_python):
        os.environ["PYSPARK_PYTHON"] = venv_python
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python
    # Create Spark session with reduced logging for cleaner output
    spark = (
        SparkSession.builder.appName("AIG-RAG-Pipeline-HF")
        .config("spark.sql.shuffle.partitions", "8")  # Keep partitions small for single-doc
        .config("spark.ui.showConsoleProgress", "false")  # Hide progress bars
        .config("spark.driver.extraJavaOptions", "-Dlog4j.logger.org.apache.spark=WARN")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # Suppress INFO spam from Spark

    # PIPELINE STEP 2: Load document and create chunks
    logger.info("[2/5] Loading + chunking AIG 2006 10-K from HF...")
    chunks_df = build_chunks_df_from_hf(spark)  # Returns Spark DataFrame of text chunks

    # PIPELINE STEP 3: Build TF-IDF search index
    logger.info("[3/5] Building TF-IDF index in Spark...")
    retriever = SparkRetriever(spark)
    retriever.build_index(chunks_df)  # Fits TF-IDF model and indexes all chunks

    # PIPELINE STEP 4: Load expected values for comparison
    logger.info("[4/5] Loading ground truth data...")
    ground_truth = load_ground_truth()  # Load hand-curated expected values
    gt_lookup = {item["year"]: item["variables"] for item in ground_truth["ground_truth"]}

    # PIPELINE STEP 5: Extract values for each year using retrieval + LLM
    logger.info("[5/5] Extracting data for each observation year...")
    extractor = LLMExtractor(model=llm_config.get("model"))
    results = [
        process_year(year, retriever, extractor, gt_lookup, retrieval_config)
        for year in TARGET_YEARS  # Process each year (2002-2006)
    ]

    total, matches, accuracy = calculate_accuracy(results)

    summary = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "years_processed": TARGET_YEARS,
            "total_comparisons": total,
            "total_matches": matches,
            "accuracy": accuracy,
        },
        "results_by_year": results,
    }

    save_results(summary, results)

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Observation years processed: {TARGET_YEARS}")
    logger.info(f"Total extractions: {total}")
    logger.info(f"Successful matches: {matches}")
    logger.info(f"Overall accuracy: {accuracy:.1%}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_pipeline()
