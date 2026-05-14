from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from classifier import Classification
from preprocessor import PreprocessedInput
from retriever import RetrievedChunk, is_retrieval_weak


@dataclass(frozen=True)
class EscalationDecision:
    status: str  # replied | escalated
    reason: str
    user_message: str


_FRAUD_RE = re.compile(r"(?i)\b(fraud|scam|chargeback|unauthorized|hacked|stolen)\b")
_PAYMENT_RE = re.compile(r"(?i)\b(payment|charged|charge|refund|billing|invoice|vat|tax)\b")
_ACCESS_RE = re.compile(r"(?i)\b(can't log in|cannot log in|login|locked out|2fa|mfa|password reset|reset password|account access)\b")
_MALICIOUS_RE = re.compile(r"(?i)\b(sql injection|drop table|rm -rf|ddos|exploit)\b")


def decide_escalation(
    inp: PreprocessedInput,
    cls: Classification,
    retrieved: Sequence[RetrievedChunk],
) -> EscalationDecision:
    # Hard safety/risk rules first.
    if inp.multi_intent:
        return EscalationDecision(
            status="escalated",
            reason="Multiple intents detected; cannot choose safe single action deterministically.",
            user_message="Thanks—your request appears to include multiple issues. I’m routing this to a human support agent to ensure each part is handled correctly.",
        )

    if _MALICIOUS_RE.search(inp.text):
        return EscalationDecision(
            status="escalated",
            reason="Potentially harmful/malicious content detected.",
            user_message="I can’t help with that request. I’m routing this to a human reviewer.",
        )

    # Visa is treated as high-risk by policy in the prompt (financial domain).
    if inp.company == "visa":
        return EscalationDecision(
            status="escalated",
            reason="Visa/financial domain requires human review.",
            user_message="I’m routing this request to a human support agent who can securely assist you.",
        )

    # Financial, fraud, or account access ambiguity => escalate.
    if _FRAUD_RE.search(inp.text) or _PAYMENT_RE.search(inp.text) or _ACCESS_RE.search(inp.text):
        return EscalationDecision(
            status="escalated",
            reason="High-risk category (fraud/billing/account access).",
            user_message="I’m routing this request to a human support agent who can securely assist you.",
        )

    if cls.request_type == "invalid":
        return EscalationDecision(
            status="escalated",
            reason="Invalid/out-of-scope request.",
            user_message="I can only help with product-related support requests from the provided knowledge base. I’m routing this to a human reviewer.",
        )

    # Grounding gate: weak retrieval => escalate.
    if is_retrieval_weak(retrieved):
        return EscalationDecision(
            status="escalated",
            reason="Insufficient relevant grounding in the local corpus.",
            user_message="I couldn’t find enough relevant information in the provided knowledge base to answer safely. I’m routing this to a human support agent for review.",
        )

    return EscalationDecision(status="replied", reason="Low-risk and grounded.", user_message="")

