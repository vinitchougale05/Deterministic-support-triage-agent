from __future__ import annotations

import re
from dataclasses import dataclass

from preprocessor import PreprocessedInput


REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}


@dataclass(frozen=True)
class Classification:
    request_type: str
    product_area: str


_FRAUD_RE = re.compile(r"(?i)\b(fraud|scam|chargeback|unauthorized|hacked|stolen|card not present)\b")
_PAYMENT_RE = re.compile(r"(?i)\b(payment|charged|charge|refund|billing|invoice|vat|tax)\b")
_ACCESS_RE = re.compile(r"(?i)\b(can't log in|cannot log in|login issue|locked out|2fa|mfa|password reset|reset password|account access)\b")
_FEATURE_RE = re.compile(r"(?i)\b(feature request|please add|can you add|it would be great if|support .* ability to|request.*feature)\b")
_BUG_RE = re.compile(r"(?i)\b(bug|error|crash|broken|fails to|not working as expected|stack trace|500\b|503\b)\b")
_SPAM_RE = re.compile(r"(?i)\b(buy now|free money|crypto|porn|viagra)\b")


def classify_request_type(inp: PreprocessedInput) -> str:
    t = inp.text.strip()
    if not t:
        return "invalid"
    if _SPAM_RE.search(t):
        return "invalid"
    if _FEATURE_RE.search(t):
        return "feature_request"
    if _BUG_RE.search(t):
        return "bug"
    return "product_issue"


def infer_product_area(inp: PreprocessedInput) -> str:
    t = inp.text.lower()
    if _PAYMENT_RE.search(t):
        return "billing"
    if _ACCESS_RE.search(t):
        return "account access"
    if any(k in t for k in ["sso", "saml", "scim", "jit"]):
        return "sso"
    if any(k in t for k in ["interview", "scorecard", "candidate", "coding interview"]):
        return "interviews"
    if any(k in t for k in ["test", "assessment", "question", "plagiarism", "proctor", "screen"]):
        return "assessments"
    if _FRAUD_RE.search(t):
        return "fraud"
    if any(k in t for k in ["api", "webhook", "integration", "ats", "greenhouse", "lever", "workday"]):
        return "integrations"
    return "general"


def classify(inp: PreprocessedInput) -> Classification:
    rt = classify_request_type(inp)
    pa = infer_product_area(inp)
    if rt not in REQUEST_TYPES:
        rt = "invalid"
    return Classification(request_type=rt, product_area=pa)

