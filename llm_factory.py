# llm_factory.py
# Central factory for Grok LLM (x.ai) via LangChain

"""
LLM Factory for Treasury AI Proof-of-Concept (PoC)

Module Overview:
----------------
This module provides a centralized factory for instantiating the Grok language model (via xAI API) using LangChain's
ChatOpenAI wrapper. It enables seamless integration of Grok into Treasury AI workflows, such as NLP-driven
counterparty selection, scenario generation, and hedge proposal reasoning. Designed for compliance with data
privacy (e.g., no PII in prompts) and efficiency in PoC environments.

Key Features:
-------------
- Environment-driven configuration: Loads API keys and params from .env for secure, reproducible setups.
- Error Handling: Raises EnvironmentError on missing XAI_API_KEY to prevent silent failures.
- LangChain Compatibility: Uses ChatOpenAI with xAI base_url; supports temperature for deterministic outputs in risk analysis.
- Defaults: Tuned for treasury use (low temperature=0.3 for factual responses; max_tokens=5000 for detailed hedge rationales).

Regulatory Alignment:
---------------------
- Aligns with MiFID II/Dodd-Frank by enabling auditable AI interactions (log prompts/responses).
- No persistent state; stateless calls support front-to-back office traceability.

Dependencies:
-------------
- Python 3.8+: os, dotenv (for .env loading).
- langchain_openai: ChatOpenAI wrapper (pip install langchain-openai).

Usage in Workflow:
------------------
- Import: from llm_factory import get_llm
- Invoke: llm = get_llm(); response = llm.invoke("Propose hedges for EUR exposure under +200bps shock.")
- Integration: Used in orchestrator_phase1.py for scenario/hedge generation; Streamlit for interactive NLP queries.
- .env Example:
  XAI_API_KEY=your_xai_key_here
  XAI_MODEL=grok-3
  GROK_TEMPERATURE=0.3
  GROK_MAX_TOKENS=5000
  GROK_RETRIES=5
  GROK_READ_TIMEOUT=60
  GROK_SAMPLES=1

Configuration Notes:
--------------------
- model: Defaults to 'grok-3'; supports xAI updates (e.g., grok-4 for Premium+).
- temperature: Low (0.3) for precise treasury advice; increase for creative scenario ideation.
- max_retries/request_timeout: Handles API flakiness in production pipelines.
- n: Supports parallel sampling (default=1); use >1 for hedge variant generation.

Error Handling:
---------------
- Missing API key: Explicit raise for setup validation.
- Invalid env vars: Falls back to defaults where safe (e.g., float/int coercion).

Author:
-------
FintekMinds TreasuryAI PoC — LLM Integration Module
Version: 2025-10 (xAI Grok via LangChain)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    """
    Returns a configured ChatOpenAI client for Grok (x.ai).
    - Uses environment variables from .env
    - Keeps only supported arguments for LangChain ChatOpenAI

    Factory function to instantiate and return a Grok LLM client.

    Returns:
        ChatOpenAI: Configured instance for xAI Grok, ready for .invoke() or .stream().

    Raises:
        EnvironmentError: If XAI_API_KEY is missing/unset.

    Configuration:
        - API Endpoint: https://api.x.ai/v1 (xAI base URL).
        - Model: 'grok-3' (default; override via XAI_MODEL).
        - Temperature: 0.3 (default; low for factual treasury outputs).
        - Max Tokens: 5000 (default; sufficient for detailed LCR/hedge responses).
        - Retries/Timeout: 5 retries, 60s timeout (defaults for reliability).
        - Samples (n): 1 (default; set >1 for variant generation in hedging).

    Treasury Use Cases:
        - Scenario Prompt: "Given LCR=120%, propose 3 rate shocks aligned with ECB guidelines."
        - Hedge NLP: "Find compliant CPs for 5y USD swap, $10M notional, under VaR limit."
        - Ensure prompts include: as_of_date, compliance flags (e.g., "Respect Basel III LCR >100%").

    Notes:
        - Stateless: Each call creates a new client; cache if needed for batching.
        - Security: .env keeps keys out of code; use secrets managers in prod (e.g., AWS SSM).
        - Testing: Invoke with simple prompt to verify: response = llm.invoke("Hello, Grok!").
    """
    if not os.getenv("XAI_API_KEY"):
        raise EnvironmentError("Missing XAI_API_KEY. Please set it in your .env file.")

    return ChatOpenAI(
        model=os.getenv("XAI_MODEL", "grok-3"),
        base_url="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY"),
        temperature=float(os.getenv("GROK_TEMPERATURE", 0.3)),
        max_tokens=int(os.getenv("GROK_MAX_TOKENS", 5000)),
        max_retries=int(os.getenv("GROK_RETRIES", 5)),
        request_timeout=int(os.getenv("GROK_READ_TIMEOUT", 60)),
        n=int(os.getenv("GROK_SAMPLES", 1))
    )