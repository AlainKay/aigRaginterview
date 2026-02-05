"""
LLM-based extraction module for SEC filing data.

This module provides the LLMExtractor class that uses the Groq API (with Llama models)
to extract structured financial data from retrieved document chunks. It handles
prompt construction, API calls with retry logic, and response parsing.

Supported variables:
    - fas133_fas52_amounts_included: Numeric (millions USD)
    - capital_markets_hedging_effect: Numeric (millions USD)
    - business_segments: Categorical (list of segment names)
"""

import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from groq import Groq
from src.config_loader import load_config

# Load environment variables
load_dotenv()

# Load configuration
CONFIG = load_config("config/pipeline_config.yaml")
RATE_LIMITS = CONFIG.get('rate_limits', {})
RETRIEVAL_CONFIG = CONFIG.get('retrieval', {})
LLM_CONFIG = CONFIG.get('llm', {})

# Rate limit handling constants (loaded from config file)
MAX_RETRIES = RATE_LIMITS.get('max_retries', 3)          # How many times to retry failed API calls
RETRY_DELAY = RATE_LIMITS.get('retry_delay_seconds', 5)  # Base delay between retries (multiplied by attempt #)
CALL_DELAY = RATE_LIMITS.get('call_delay_seconds', 10)   # Delay between successful API calls
RATE_LIMIT_WAIT = RATE_LIMITS.get('rate_limit_wait_seconds', 60)  # Wait time when rate limited (HTTP 429)
CHUNKS_PER_EXTRACTION = RETRIEVAL_CONFIG.get('chunks_per_extraction', 2)  # How many chunks to send to LLM


class LLMExtractor:
    """
    Extract structured data from SEC filing chunks using an LLM.

    This class constructs targeted prompts for each variable type and handles
    the Groq API interaction with automatic retry logic for rate limits and
    JSON parsing errors.

    Attributes:
        client: Groq API client instance.
        model: Model identifier (e.g., 'llama-3.1-8b-instant').

    Example:
        >>> extractor = LLMExtractor()
        >>> result = extractor.extract_capital_markets_hedging_effect(chunks, 2006)
        >>> print(result['value'])  # e.g., -1820
    """

    def __init__(self, model: str = None):
        """
        Initialize the LLM extractor with Groq API credentials.

        Args:
            model: Groq model identifier to use for extraction. If None,
                   defaults to the value in pipeline_config.yaml (typically
                   'llama-3.1-8b-instant').

        Raises:
            ValueError: If GROQ_API_KEY environment variable is not set.
        """
        # Read API key from environment so we don't hardcode secrets.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        # Create the Groq client once and reuse it.
        self.client = Groq(api_key=api_key)
        self.model = model or LLM_CONFIG.get('model', 'llama-3.1-8b-instant')

    def extract_business_segments(
        self,
        chunks: List[Dict],
        year: int
    ) -> Dict[str, Any]:
        """
        Extract the list of reportable business segments for a given year.

        Constructs a prompt that asks the LLM to identify the major reportable
        segments from the retrieved chunks, typically found in Item 1 of the 10-K.

        Args:
            chunks: List of retrieved chunk dictionaries, each containing:
                - chunk_id (int): Unique identifier
                - text (str): Chunk content
                - score (float): Retrieval relevance score
            year: Fiscal year for which to extract segments (e.g., 2006).

        Returns:
            Dictionary containing:
                - value (List[str]): List of segment names
                - confidence (str): 'high', 'medium', 'low', or 'error'
                - source_text (str): Text excerpt supporting the extraction
                - source_chunk_ids (List[int]): IDs of chunks used
                - variable (str): 'Business Segments'
                - year (int): The target year
        """
        # Use only the top chunks so the prompt stays short and focused.
        used_chunks = chunks[:CHUNKS_PER_EXTRACTION]
        source_chunk_ids = [c['chunk_id'] for c in used_chunks]
        # Build the context text the LLM will read.
        context = "\n\n---\n\n".join([
            f"[Chunk {c['chunk_id']}]\n{c['text']}"
            for c in used_chunks
        ])

        # Different segment names by year
        segment_hint = ""
        if year <= 2008:
            segment_hint = "For 2006-2008, AIG typically had: General Insurance, Life Insurance & Retirement Services, Financial Services, Asset Management"
        else:
            segment_hint = "For 2009-2010, AIG reorganized into: Chartis (insurance), SunAmerica (life/retirement), Financial Services"

        prompt = f"""You are a financial data extraction expert analyzing AIG's SEC 10-K filing.

TASK: Extract the REPORTABLE BUSINESS SEGMENTS for fiscal year {year}.

CONTEXT FROM 10-K FILING:
{context}

CRITICAL INSTRUCTIONS:
1. Look for "reportable segments" or "principal business units" in Item 1. Business
2. List ONLY the major reportable segments (usually 3-4 segments)
3. {segment_hint}
4. Use the exact segment names as they appear in the filing
5. Do NOT include sub-segments or "Other" categories

Respond with ONLY this JSON (no other text):
{{
    "value": ["Segment 1", "Segment 2", "Segment 3", "Segment 4"],
    "confidence": "high",
    "source_text": "<text listing the segments>"
}}"""

        # Call the LLM and attach chunk IDs for traceability.
        result = self._call_llm(prompt, "Business Segments", year)
        result['source_chunk_ids'] = source_chunk_ids
        return result

    def extract_fas133_fas52_amounts_included(
        self,
        chunks: List[Dict],
        year: int
    ) -> Dict[str, Any]:
        """
        Extract the FAS 133/FAS 52 amounts included for a specific year.

        Looks for the 5-year series sentence in the 10-K and extracts the
        value corresponding to the target year. Handles conversion from
        billions to millions and interprets parentheses as negative values.

        Args:
            chunks: List of retrieved chunk dictionaries, each containing:
                - chunk_id (int): Unique identifier
                - text (str): Chunk content
                - score (float): Retrieval relevance score
            year: Fiscal year for which to extract the amount (2002-2006).

        Returns:
            Dictionary containing:
                - value (int): Amount in millions USD (negative if in parentheses)
                - confidence (str): 'high', 'medium', 'low', or 'error'
                - source_text (str): Exact text showing the extracted value
                - source_chunk_ids (List[int]): IDs of chunks used
                - variable (str): 'FAS133/FAS52 Amounts Included'
                - year (int): The target year
                - unit (str): 'millions USD'
        """
        # Use only the top chunks so the prompt stays short and focused.
        used_chunks = chunks[:CHUNKS_PER_EXTRACTION]
        source_chunk_ids = [c['chunk_id'] for c in used_chunks]
        # Build the context text the LLM will read.
        context = "\n\n---\n\n".join([
            f"[Chunk {c['chunk_id']}]\n{c['text']}"
            for c in used_chunks
        ])

        prompt = f"""You are extracting a 5-year series from AIG's 2006 10-K.

TASK: Find the amounts included related to FAS 133 and FAS 52 for year {year}.

CONTEXT:
{context}

INSTRUCTIONS:
1. Look for the sentence: "For 2006, 2005, 2004, 2003 and 2002, respectively, the amounts included are ..."
2. Extract the value for {year} only.
3. Convert to millions (e.g., 1.86 billion -> 1860).
4. Parentheses indicate negative (e.g., $(1.86) billion -> -1860).
5. Return an integer in millions.

Respond with ONLY this JSON:
{{"value": <integer in millions>, "confidence": "high", "source_text": "<exact text showing the value>"}}"""

        # Call the LLM and attach chunk IDs for traceability.
        result = self._call_llm(prompt, "FAS133/FAS52 Amounts Included", year, "millions USD")
        result['source_chunk_ids'] = source_chunk_ids
        return result

    def extract_capital_markets_hedging_effect(
        self,
        chunks: List[Dict],
        year: int
    ) -> Dict[str, Any]:
        """
        Extract the Capital Markets hedging effect for a specific year.

        Searches for the sentence describing the hedging effect on revenues
        and operating income for Capital Markets, extracting the value for
        the target year from the 5-year series.

        Args:
            chunks: List of retrieved chunk dictionaries, each containing:
                - chunk_id (int): Unique identifier
                - text (str): Chunk content
                - score (float): Retrieval relevance score
            year: Fiscal year for which to extract the effect (2002-2006).

        Returns:
            Dictionary containing:
                - value (int): Effect in millions USD (negative if in parentheses)
                - confidence (str): 'high', 'medium', 'low', or 'error'
                - source_text (str): Exact text showing the extracted value
                - source_chunk_ids (List[int]): IDs of chunks used
                - variable (str): 'Capital Markets Hedging Effect'
                - year (int): The target year
                - unit (str): 'millions USD'
        """
        # Use only the top chunks so the prompt stays short and focused.
        used_chunks = chunks[:CHUNKS_PER_EXTRACTION]
        source_chunk_ids = [c['chunk_id'] for c in used_chunks]
        # Build the context text the LLM will read.
        context = "\n\n---\n\n".join([
            f"[Chunk {c['chunk_id']}]\n{c['text']}"
            for c in used_chunks
        ])

        prompt = f"""You are extracting a 5-year series from AIG's 2006 10-K.

TASK: Find the Capital Markets hedging effect (in both revenues and operating income) for year {year}.

CONTEXT:
{context}

INSTRUCTIONS:
1. Look for the sentence: "For 2006, 2005, 2004, 2003 and 2002, respectively, the effect was ... in both revenues and operating income for Capital Markets."
2. Extract the value for {year} only (revenues and operating income are the same in this sentence).
3. Convert to millions (e.g., 1.82 billion -> 1820).
4. Parentheses indicate negative (e.g., $(1.82) billion -> -1820).
5. Return an integer in millions.

Respond with ONLY this JSON:
{{"value": <integer in millions>, "confidence": "high", "source_text": "<exact text showing the value>"}}"""

        # Call the LLM and attach chunk IDs for traceability.
        result = self._call_llm(prompt, "Capital Markets Hedging Effect", year, "millions USD")
        result['source_chunk_ids'] = source_chunk_ids
        return result

    def _call_llm(
        self,
        prompt: str,
        variable_name: str,
        year: int,
        unit: str = None
    ) -> Dict[str, Any]:
        """
        Call the LLM API and parse the JSON response.

        Handles retry logic for rate limits (HTTP 429) and JSON parsing errors.
        Implements exponential backoff between retries.

        Args:
            prompt: Complete prompt string including context and instructions.
            variable_name: Human-readable name for the variable being extracted.
            year: Target year for the extraction.
            unit: Unit of measurement (e.g., 'millions USD'). None for categorical.

        Returns:
            Dictionary containing:
                - value: Extracted value (int for numeric, List[str] for categorical)
                - confidence (str): 'high', 'medium', 'low', or 'error'
                - source_text (str): Supporting text from the document
                - variable (str): The variable_name parameter
                - year (int): The year parameter
                - unit (str, optional): The unit parameter if provided
                - error (str, optional): Error message if extraction failed
        """
        last_error = None  # Track the most recent error for reporting

        # Retry loop with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                # Wait longer on each retry attempt (exponential backoff)
                if attempt > 0:
                    time.sleep(RETRY_DELAY * (attempt + 1))

                # Make the actual API call to Groq
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise financial data extraction assistant. Extract exact values from SEC filings. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=300
                )

                # Extract the text response from the API result
                response_text = response.choices[0].message.content.strip()

                # Clean up response: LLM sometimes wraps JSON in markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                # Parse the JSON response from the LLM
                result = json.loads(response_text.strip())

                # For numeric variables, ensure the value is an integer (not string)
                if unit and result.get('value') is not None:
                    try:
                        result['value'] = int(float(result['value']))
                    except (TypeError, ValueError):
                        pass

                # Add metadata to the result for traceability
                result['variable'] = variable_name
                result['year'] = year
                if unit:
                    result['unit'] = unit

                time.sleep(CALL_DELAY)  # Polite delay between API calls
                return result

            except json.JSONDecodeError as e:
                # LLM returned invalid JSON - will retry
                last_error = f'JSON parse error: {str(e)}'
            except Exception as e:
                last_error = str(e)
                # Check if we hit API rate limit (HTTP 429)
                if "429" in str(e) or "rate" in str(e).lower():
                    print(f"  Rate limited, waiting {RATE_LIMIT_WAIT}s...")
                    time.sleep(RATE_LIMIT_WAIT)

        # All retries failed - return error result
        return {
            'variable': variable_name,
            'year': year,
            'value': None,
            'confidence': 'error',
            'error': last_error
        }

    def extract_all_variables(
        self,
        retrieval_context: Dict[str, List[Dict]],
        year: int
    ) -> Dict[str, Dict]:
        """
        Extract all configured variables for a single observation year.

        Orchestrates extraction of business_segments, fas133_fas52_amounts_included,
        and capital_markets_hedging_effect using their respective methods.

        Args:
            retrieval_context: Dictionary mapping variable names to their
                              retrieved chunks. Keys should match the variable
                              names in VARIABLE_QUERIES config.
            year: Fiscal year for which to extract all variables.

        Returns:
            Dictionary mapping variable names to their extraction results.
            Each result contains value, confidence, source_text, and metadata.

        Example:
            >>> context = {
            ...     'business_segments': [...chunks...],
            ...     'capital_markets_hedging_effect': [...chunks...]
            ... }
            >>> results = extractor.extract_all_variables(context, 2006)
            >>> print(results['business_segments']['value'])
        """
        # Gather all variable extractions for this year.
        results = {}

        if 'business_segments' in retrieval_context:
            results['business_segments'] = self.extract_business_segments(
                retrieval_context['business_segments'], year
            )

        if 'fas133_fas52_amounts_included' in retrieval_context:
            results['fas133_fas52_amounts_included'] = self.extract_fas133_fas52_amounts_included(
                retrieval_context['fas133_fas52_amounts_included'], year
            )

        if 'capital_markets_hedging_effect' in retrieval_context:
            results['capital_markets_hedging_effect'] = self.extract_capital_markets_hedging_effect(
                retrieval_context['capital_markets_hedging_effect'], year
            )

        return results
