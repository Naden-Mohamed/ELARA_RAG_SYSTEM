import re
from typing import List, Optional
from core.config import get_settings
from core.prompts import REFUSAL_MARKER_EN, REFUSAL_MARKER_AR
from routers.schemas.rag_requests import MockChunkInput

INJECTION_PATTERNS = [
    r"ignore (all|any|the) previous instructions",
    r"ignore your instructions",
    r"disregard (the|your) system prompt",
    r"reveal (the|your) system prompt",
    r"you are now",
    r"jailbreak",
    r"act as if you have no restrictions",
    r"pretend you are not an? ai",
]

PERSONAL_ADVICE_PATTERNS = [
    r"\bmy (grandmother|grandfather|mother|father|baby|infant|child|wife|husband)\b.*\b(dose|dosage|medication|medicine|take|give|stop|start)\b",
    r"\bwhat dose should i (give|take)\b",
    r"\bhow much .* should i give\b",
    r"\bwhat should i take\b",
    r"\bwhat should i give\b",
    r"\bshould i stop\b",
    r"\bis it safe for me to take\b",
    r"\bcan i take\b.*\bfor my\b",
]

CITATION_DOC_PATTERN = re.compile(r"\[(?:Doc|المستند):\s*([^,]+),")


def matches_any(text: str, patterns: List[str]) -> bool:
    """Checks whether any regex pattern matches text (case-insensitive)."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def pre_generation_gate(query: str, chunks: List[MockChunkInput]) -> dict:
    """Deterministic, pre-LLM safety gate. Runs before any Groq call is made.

    Returns:
        {"allow": bool, "reason": str}
    """
    if matches_any(query, INJECTION_PATTERNS):
        return {"allow": False, "reason": "Prompt-injection attempt detected."}

    if matches_any(query, PERSONAL_ADVICE_PATTERNS):
        return {
            "allow": False,
            "reason": "Patient-specific dosing/treatment advice is outside the evidence-only safety boundary.",
        }

    if not chunks:
        return {"allow": False, "reason": "No relevant guideline passages were retrieved."}

    settings = get_settings()
    threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.60)
    top_score = max((c.score for c in chunks), default=0.0)
    if top_score < threshold:
        return {
            "allow": False,
            "reason": f"Retrieval confidence below threshold ({top_score:.3f} < {threshold:.3f}).",
        }

    return {"allow": True, "reason": "Deterministic retrieval gate passed.", "top_score": top_score}


def is_model_refusal(answer: str) -> bool:
    """Detects the model's own instructed refusal phrase (EN or AR)."""
    return REFUSAL_MARKER_EN.lower() in answer.lower() or REFUSAL_MARKER_AR in answer


def validate_grounded_response(answer: str, citations: List[str], chunks: List[MockChunkInput]) -> dict:
    """Post-generation validation: catches missing or fabricated citations.

    A clean model-issued refusal always passes (it correctly has no citations).
    A non-refusal answer must have at least one citation, and every cited
    document name must match a document actually present in the retrieved
    chunks -- catching citations to sources that were never in context.

    Returns:
        {"valid": bool, "is_refusal": bool, "reason": str}
    """
    if is_model_refusal(answer):
        return {"valid": True, "is_refusal": True, "reason": "Model correctly refused due to missing evidence."}

    if not citations:
        return {
            "valid": False,
            "is_refusal": False,
            "reason": "Answer makes claims but includes no citation -- format contract violated.",
        }

    known_doc_names = {c.doc_name.strip().lower() for c in chunks}
    for citation in citations:
        match = CITATION_DOC_PATTERN.search(citation)
        cited_doc = match.group(1).strip().lower() if match else None
        if not cited_doc or cited_doc not in known_doc_names:
            return {
                "valid": False,
                "is_refusal": False,
                "reason": f"Citation references a document not present in retrieved context: {citation!r}",
            }

    return {"valid": True, "is_refusal": False, "reason": "All citations verified against retrieved context."}


def build_safe_fallback_message(language) -> str:
    """Canned safe response used when the gate blocks generation or validation fails post-hoc."""
    from routers.schemas.rag_requests import LanguageEnum
    if language == LanguageEnum.AR:
        return REFUSAL_MARKER_AR
    return REFUSAL_MARKER_EN