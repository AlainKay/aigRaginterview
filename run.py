# Entry point for the RAG extraction pipeline

import sys
import os
import logging
import argparse

from src.config_loader import load_config
from src.main_pipeline_spark import run_pipeline
import src.main_pipeline_spark as pipeline_module


def parse_args():
    parser = argparse.ArgumentParser(description="AIG 10-K RAG Extraction Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show config and exit")
    parser.add_argument("--years", "-y", type=str, default=None,
                        help="Comma-separated years (e.g. '2005,2006'). Overrides config.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Point PySpark at the venv Python
    venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
    if os.path.exists(venv_python):
        os.environ["PYSPARK_PYTHON"] = venv_python
        os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    config = load_config()
    logger.info("Config loaded")

    if args.dry_run:
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"Years: {config.get('target_years', [])}")
        print(f"LLM: {config.get('llm', {}).get('model', 'llama-3.1-8b-instant')}")
        print(f"Top-K: {config.get('retrieval', {}).get('top_k', 5)}")
        print(f"Ground Truth: {config.get('paths', {}).get('ground_truth')}")
        print("=" * 50)
        return

    # Override years if --years flag provided
    VALID_YEARS = {2002, 2003, 2004, 2005, 2006}
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
        invalid = set(years) - VALID_YEARS
        if invalid:
            logger.error(f"Invalid years: {invalid}. Only 2002-2006 supported.")
            sys.exit(1)
        logger.info(f"Overriding target years: {years}")
        pipeline_module.TARGET_YEARS = years

    result = run_pipeline()

    if result is None:
        logger.error("Pipeline failed")
        sys.exit(1)

    accuracy = result["pipeline_run"]["accuracy"]
    logger.info(f"Pipeline completed with {accuracy:.1%} accuracy")


if __name__ == "__main__":
    main()
