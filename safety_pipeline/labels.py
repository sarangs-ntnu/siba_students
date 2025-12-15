from __future__ import annotations

import re
from typing import Dict

HEDGING_PATTERNS = [
    r"\bmight\b",
    r"\bcould\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bit may help\b",
]

REFERRAL_PATTERNS = [
    r"talk to a (therapist|counsel(or|or)|professional)",
    r"reach out to (a )?(doctor|gp|primary care)",
    r"contact (a )?(hotline|crisis line|emergency)",
]

NON_DIRECTIVE_PATTERNS = [
    r"if you feel",
    r"would you",
    r"it can help to",
    r"you could",
    r"you might",
]

NON_PRESCRIPTIVE_PATTERNS = [
    r"consider",
    r"could",
    r"might",
    r"option",
    r"explore",
    r"one idea",
]

DIAGNOSTIC_PATTERNS = [
    r"you have [a-z]+",
    r"you are suffering from",
    r"this is definitely",
]


def _has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def derive_safety_labels(response: str) -> Dict[str, bool]:
    """Derive safety-supportive cues from a response.

    The labels are designed to be additive metadata and do not alter content.
    """

    non_diagnostic_language = not _has_pattern(response, DIAGNOSTIC_PATTERNS) and _has_pattern(
        response, HEDGING_PATTERNS
    )
    non_prescriptive_advice = _has_pattern(response, NON_PRESCRIPTIVE_PATTERNS) and not re.search(
        r"\bmust\b|\bhave to\b|\bneed to\b|\bshould\b",
        response,
        flags=re.IGNORECASE,
    )
    professional_referral = _has_pattern(response, REFERRAL_PATTERNS)
    non_directive_phrasing = _has_pattern(response, NON_DIRECTIVE_PATTERNS)

    return {
        "non_diagnostic_language": bool(non_diagnostic_language),
        "non_prescriptive_advice": bool(non_prescriptive_advice),
        "professional_referral_cues": bool(professional_referral),
        "non_directive_phrasing": bool(non_directive_phrasing),
    }
