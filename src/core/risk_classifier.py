
from enum import Enum
from core.safety_gate import matches_any, INJECTION_PATTERNS, PERSONAL_ADVICE_PATTERNS


class RiskLevel(str, Enum):
    SAFE = "safe"
    AMBIGUOUS = "ambiguous"
    UNSAFE = "unsafe"
    OUT_OF_SCOPE = "out_of_scope"  # decided post-retrieval, once similarity is known


AMBIGUOUS_PATTERNS = [
    r"\bthe (trial|study|report)\b(?!.*\b(who|nice|acog)\b)",   # vague reference, no named source
    r"\bwhat (were|was) the (results|findings)\b",
    r"\bcompare\b.*\band\b",  # multi-entity comparison -- higher hallucination risk
]


def classify_input_risk(query: str) -> dict:
    """Classifies a query's risk level from text alone, before retrieval runs.

    Returns:
        {"risk_level": RiskLevel, "reason": str}
    """
    if matches_any(query, INJECTION_PATTERNS):
        return {"risk_level": RiskLevel.UNSAFE, "reason": "Prompt-injection pattern matched."}

    if matches_any(query, PERSONAL_ADVICE_PATTERNS):
        return {"risk_level": RiskLevel.UNSAFE, "reason": "Personal dosing/treatment advice pattern matched."}

    if matches_any(query, AMBIGUOUS_PATTERNS):
        return {"risk_level": RiskLevel.AMBIGUOUS, "reason": "Query lacks a specific named entity/source."}

    return {"risk_level": RiskLevel.SAFE, "reason": "No risk pattern matched at input stage."}