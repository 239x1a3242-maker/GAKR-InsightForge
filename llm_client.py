# backend/llm_client.py
"""
llm_client.py
Adapter for the local Phi-3 mini server defined in run.py.

All other services call: query_llm(prompt: str) -> str or JSON string.
"""

from typing import Any
import os
import requests


# URL of your GAKR AI backend (Phi-3 mini)
# In run.py it runs on port 8080
PHI3_BASE_URL = os.getenv("PHI3_BASE_URL", "http://localhost:8080")
PHI3_ANALYZE_PATH = "/api/analyze"

# API key used in run.py
DEFAULT_API_KEY = os.getenv("PHI3_API_KEY", "gakr-ai-2025-secret")


def query_llm(prompt: str) -> Any:
    """
    Send prompt to Phi-3 mini via POST /api/analyze and return the model text.

    This matches your run.py contract:

    Request (multipart/form-data):
      prompt: str
      api_key: str
      files: optional (we do NOT send files here)

    Response JSON:
      {
        "response": "<model_output_text>",
        "context": {...},
        "status": "success"
      }
    """
    url = PHI3_BASE_URL.rstrip("/") + PHI3_ANALYZE_PATH

    data = {
        "prompt": prompt,
        "api_key": DEFAULT_API_KEY,
    }

    try:
        # No files → general assistant mode
        resp = requests.post(url, data=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        # We care about the model's answer in "response"
        return payload.get("response", "")
    except Exception as e:
        # If something goes wrong, return an empty schema so pipeline still works
        from services.ai_insights_service import NARRATIVE_SCHEMA
        # In production, log `e` to a file or monitoring system
        return NARRATIVE_SCHEMA.copy()
