#!/usr/bin/env python3
"""
RAG Ground Truth Pipeline - Spark + HF (single document).

This is the main entry point for the RAG extraction pipeline. It handles
command-line argument parsing, configuration loading, and pipeline execution.

Usage:
    python run.py                    # Run full Spark pipeline
    python run.py --config custom.yaml  # Use custom config file
    python run.py --dry-run          # Show config without running
    python run.py --verbose          # Enable debug logging

Exit Codes:
    0: Success
    1: Configuration error or pipeline failure
"""

import argparse
import sys
import os
import logging
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config_loader import load_config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace object with parsed arguments:
            - config (str): Path to YAML configuration file
            - verbose (bool): Whether to enable debug logging
            - dry_run (bool): Whether to show config and exit
    """
    parser = argparse.ArgumentParser(
        description="AIG 10-K RAG Extraction Pipeline (Spark + HF, single-doc)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                      Run with default config
  python run.py --verbose            Enable debug logging
  python run.py --dry-run            Show configuration without running
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config/pipeline_config.yaml",
        help="Path to configuration file (default: config/pipeline_config.yaml)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show configuration and exit without running",
    )

    parser.add_argument(
        "--years",
        "-y",
        type=str,
        default=None,
        help="Comma-separated years to process (e.g., '2006' or '2005,2006'). Overrides config.",
    )

    return parser.parse_args()


def show_config(config: Dict[str, Any]) -> None:
    """
    Display key configuration settings to the console.

    Prints a formatted summary of the pipeline configuration for
    verification before running.

    Args:
        config: Configuration dictionary loaded from YAML file.
    """
    print("\n" + "=" * 50)
    print("CONFIGURATION (SPARK + HF, SINGLE DOC)")
    print("=" * 50)

    years_to_use = config.get("target_years", [])
    print(f"Observation years: {years_to_use}")

    llm = config.get("llm", {})
    print(f"LLM Provider: {llm.get('provider', 'groq')}")
    print(f"LLM Model: {llm.get('model', 'llama-3.1-8b-instant')}")

    retrieval = config.get("retrieval", {})
    print(f"Top-K Retrieval: {retrieval.get('top_k', 5)}")
    print(f"Chunks per Extraction: {retrieval.get('chunks_per_extraction', 2)}")

    paths = config.get("paths", {})
    print(f"Ground Truth: {paths.get('ground_truth')}")
    print(f"Results JSON: {paths.get('results_json')}")
    print(f"Results CSV: {paths.get('results_csv')}")

    print("=" * 50 + "\n")


def main() -> None:
    """
    Main CLI entry point for the RAG extraction pipeline.

    Parses arguments, loads configuration, and either displays config
    (dry-run mode) or executes the full Spark-based extraction pipeline.

    Exits with code 1 on configuration errors or pipeline failures.
    """
    args = parse_args()

    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if os.path.exists(venv_python):
        os.environ["PYSPARK_PYTHON"] = venv_python
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if args.dry_run:
        show_config(config)
        print("Dry run complete. Use without --dry-run to execute.")
        return

    from main_pipeline_spark import run_pipeline
    import main_pipeline_spark as pipeline_module

    pipeline_module.CONFIG = config

    # Override years if --years flag provided (allows quick testing with subset)
    # Only years 2002-2006 are valid because we're using the 2006 10-K filing
    # which contains 5-year historical data for those years
    VALID_YEARS = {2002, 2003, 2004, 2005, 2006}
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]  # Parse comma-separated years
        invalid = set(years) - VALID_YEARS
        if invalid:
            logger.error(f"Invalid years: {invalid}. Only 2002-2006 supported (data from 2006 10-K).")
            sys.exit(1)
        logger.info(f"Overriding target years: {years}")
        pipeline_module.TARGET_YEARS = years
    else:
        pipeline_module.TARGET_YEARS = config.get("target_years", [])  # Use config default

    result = run_pipeline()

    if result is None:
        logger.error("Pipeline failed")
        sys.exit(1)

    accuracy = result["pipeline_run"]["accuracy"]
    logger.info(f"Pipeline completed with {accuracy:.1%} accuracy")


if __name__ == "__main__":
    main()
