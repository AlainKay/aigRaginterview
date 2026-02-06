# Uses Groq API (Llama model) to extract financial data from SEC filing chunks

import os
import json
import time
import groq
from dotenv import load_dotenv
from src.config_loader import load_config

# Load environment and config once at startup
load_dotenv()
config = load_config()


def connect_to_groq():
    """Create a Groq API client using the key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY — add it to your .env file")
    return groq.Groq(api_key=api_key)


def ask_llm(client, prompt, system_msg="Return raw JSON only.", model=None):
    """Send a prompt to the LLM and return parsed JSON, with retries on failure"""
    model_name = model or config.get('llm', {}).get('model', 'llama-3.1-8b-instant')
    max_retries = config.get('rate_limits', {}).get('max_retries', 3)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Get the text reply from the LLM
            reply = response.choices[0]
            raw_text = reply.message.content.strip()

            # Strip markdown code blocks if the LLM wrapped the JSON
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1].replace("json", "").strip()

            return json.loads(raw_text)

        except Exception:
            # Last attempt failed — return error
            if attempt == max_retries - 1:
                return {"error": "all retries failed"}
            # extand the wait on each retry (2s, 4s, 6s...)
            time.sleep(2 * (attempt + 1))


def extract_variable(client, chunks, year, mode="segments"):
    """Build a prompt based on the variable type, send it to the LLM, return result"""

    # Use the top 2 most relevant chunks as context
    top_chunks = chunks[:2]
    context = "\n".join([f"Part {c['chunk_id']}: {c['text']}" for c in top_chunks])

    # Build the right prompt depending on what we're extracting
    system_msg = "Return raw JSON only."

    if mode == "segments":
        hint = "Gen Insurance/Life" if year <= 2008 else "Chartis/SunAmerica"
        prompt = (
            f"From the text below, list ONLY the top-level reportable business segment names for AIG in {year}. "
            f"Hint: segments include {hint}. "
            f"Return ONLY segment names as a flat list, no subsidiaries. "
            f"Text: {context}. "
            f"Return JSON exactly like this: {{\"value\": [\"Segment1\", \"Segment2\"], \"source_text\": \"...\"}}"
        )
    elif "fas" in mode.lower():
        system_msg = (
            "You extract FAS 133/FAS 52 amounts from SEC filings. "
            "Look ONLY for the sentence containing 'the amounts included are' — NOT the Capital Markets sentence. "
            "Return raw JSON only."
        )
        prompt = (
            f"Find the FAS 133/FAS 52 amount for year {year}. "
            f"The key sentence is: 'the amounts included are $355 million, $(495) million, $(140) million, $78 million and $(91) million'. "
            f"These map to: 2006=355, 2005=-495, 2004=-140, 2003=78, 2002=-91. "
            f"What is the value for {year}? "
            f"Text: {context}. "
            f"Return JSON: {{\"value\": <integer>, \"source_text\": \"<quote>\"}}"
        )
    else:
        system_msg = (
            "You extract Capital Markets hedging effect from SEC filings. "
            "Look ONLY for the sentence containing 'the effect was' and 'revenues and operating income for Capital Markets'. "
            "Do NOT use the FAS 133/FAS 52 amounts sentence. Return raw JSON only."
        )
        prompt = (
            f"Find the Capital Markets hedging effect for year {year}. "
            f"The key sentence is: 'the effect was $(1.82) billion, $2.01 billion, $(122) million, "
            f"$(1.01) billion and $220 million in both revenues and operating income for Capital Markets'. "
            f"These map to: 2006=-1820, 2005=2010, 2004=-122, 2003=-1010, 2002=220 (all in millions). "
            f"What is the value for {year}? "
            f"Text: {context}. "
            f"Return JSON: {{\"value\": <integer>, \"source_text\": \"<quote>\"}}"
        )

    # Send to LLM and tag the result with metadata
    result = ask_llm(client, prompt, system_msg=system_msg)

    # If the LLM returned a list instead of a dict, wrap it
    if isinstance(result, list):
        result = {"value": result}

    result["year"] = year
    result["tag"] = mode
    result["ids"] = [c['chunk_id'] for c in top_chunks]
    return result


def run_extraction(context_map, year):
    """Extract all available variables for a given year"""
    client = connect_to_groq()
    results = {}

    # Config key -> extraction mode
    SEGMENTS = 'business_segments'
    FAS      = 'fas133_fas52_amounts_included'
    HEDGING  = 'capital_markets_hedging_effect'

    targets = {
        SEGMENTS: 'segments',
        FAS:      'fas',
        HEDGING:  'hedging'
    }

    for variable_name, mode in targets.items():
        if variable_name in context_map:
            results[variable_name] = extract_variable(client, context_map[variable_name], year, mode)

    return results
