from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PreprocessedInput:
    subject: str
    issue: str
    company: Optional[str]
    text: str
    multi_intent: bool


_WS_RE = re.compile(r"\s+")
_NOISE_RE = re.compile(r"(?i)\b(sent from my iphone|thanks+|thank you+|pls|plz)\b")


def normalize_company(company: Optional[str]) -> Optional[str]:
    if company is None:
        return None
    c = _WS_RE.sub(" ", str(company).strip()).lower()
    if not c or c in {"none", "null", "nan"}:
        return None
    if "visa" in c:
        return "visa"
    if "hackerrank" in c or "hacker rank" in c:
        return "hackerrank"
    if "claude" in c or "anthropic" in c:
        return "claude"
    return c


def _clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u200b", " ")
    s = _NOISE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def detect_multi_intent(text: str) -> bool:
    t = text.lower()
    # Simple deterministic heuristics: multiple explicit questions, or "also/another issue"
    q = t.count("?")
    if q >= 2:
        return True
    if any(kw in t for kw in [" also ", " another issue", " additionally", " second issue", " separate issue"]):
        return True
    # Common pattern: multiple bullet-like lines
    if "\n-" in text or "\n*" in text:
        return True
    return False


def preprocess(subject: str, issue: str, company: Optional[str]) -> PreprocessedInput:
    subj = _clean_text(subject)
    iss = _clean_text(issue)
    combined = _clean_text(f"{subj}\n{iss}".strip())
    multi = detect_multi_intent(combined)
    return PreprocessedInput(
        subject=subj,
        issue=iss,
        company=normalize_company(company),
        text=combined,
        multi_intent=multi,
    )

