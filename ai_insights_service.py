"""
AI Insights Service (Structured):
- Takes a single analysis_payload from main.py
- Returns fixed-schema bullet-point narrative
"""

from typing import Dict, Any
from scipy.stats import zscore
from llm_client import query_llm



NARRATIVE_SCHEMA: Dict[str, list] = {
    "descriptive_analytics": [],
    "diagnostic_analytics": [],
    "risk_assessment": [],
    "predictive_analytics": [],
    "prescriptive_analytics": [],
}


def build_prompt(analysis_payload: Dict[str, Any]) -> str:
    prompt = f"""
You are an enterprise data analyst.

Your task is to generate STRUCTURED ANALYTICS NARRATION.

Rules:
- Output must follow EXACTLY this structure:
  1. Descriptive Analytics
  2. Diagnostic Analytics
  3. Risk Assessment
  4. Predictive Analytics
  5. Prescriptive Analytics

Formatting rules:
- Each section must contain 3–5 bullet points
- Each bullet must be a single factual statement
- No paragraphs
- No assumptions
- Use provided metrics only
- Be concise and business-focused

Use this input data (JSON):
{analysis_payload}

Return ONLY valid JSON with this exact schema:
{{
  "descriptive_analytics": ["...", "..."],
  "diagnostic_analytics": ["...", "..."],
  "risk_assessment": ["...", "..."],
  "predictive_analytics": ["...", "..."],
  "prescriptive_analytics": ["...", "..."]
}}
"""
    return prompt.strip()


def validate_narrative(narrative: Dict[str, Any]) -> Dict[str, list]:
    cleaned = {}
    for section in NARRATIVE_SCHEMA.keys():
        bullets = narrative.get(section, [])
        if not isinstance(bullets, list):
            bullets = [str(bullets)]
        bullets = [str(b).strip() for b in bullets if str(b).strip()]
        if len(bullets) > 5:
            bullets = bullets[:5]
        cleaned[section] = bullets
    return cleaned


def generate_structured_narrative(analysis_payload: Dict[str, Any]) -> Dict[str, list]:
    prompt = build_prompt(analysis_payload)
    raw = query_llm(prompt)

    if isinstance(raw, str):
        import json
        try:
            model_json = json.loads(raw)
        except Exception:
            model_json = NARRATIVE_SCHEMA.copy()
    else:
        model_json = raw

    structured = validate_narrative(model_json)
    return structured
