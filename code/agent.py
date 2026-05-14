from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from classifier import Classification, classify
from escalation import EscalationDecision, decide_escalation
from generator import ESCALATE_REQUIRED, GeneratedAnswer, generate_response_from_context
from ollama_client import OllamaConfig
from preprocessor import PreprocessedInput, preprocess
from retriever import RetrievedChunk, TfidfRetriever


@dataclass(frozen=True)
class TriageResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


def _evidence_block(retrieved: Sequence[RetrievedChunk], max_chars_per: int = 420) -> str:
    lines: List[str] = []
    for r in retrieved:
        snippet = r.chunk.text.strip().replace("\n", " ")
        if len(snippet) > max_chars_per:
            snippet = snippet[: max_chars_per - 3] + "..."
        lines.append(f"- score={r.score:.3f} source={r.chunk.source_path} text={snippet}")
    # Keep CSV rows single-line: do not emit newlines in justification.
    return " || ".join(lines).strip()


def triage_one(
    subject: str,
    issue: str,
    company: str | None,
    retriever: TfidfRetriever,
    ollama_cfg: OllamaConfig,
    k: int = 5,
) -> TriageResult:
    inp: PreprocessedInput = preprocess(subject=subject, issue=issue, company=company)
    cls: Classification = classify(inp)
    retrieved: List[RetrievedChunk] = retriever.retrieve(inp.text, k=k, company=inp.company)

    esc: EscalationDecision = decide_escalation(inp, cls, retrieved)
    if esc.status == "escalated":
        justification = (
            f"request_type={cls.request_type}; product_area={cls.product_area}; status=escalated. "
            f"reason={esc.reason} "
            f"Evidence: {_evidence_block(retrieved)}"
        )
        return TriageResult(
            status="escalated",
            product_area=cls.product_area,
            response=esc.user_message,
            justification=justification,
            request_type=cls.request_type,
        )

    gen: GeneratedAnswer = generate_response_from_context(inp.text, retrieved, cfg=ollama_cfg)
    if gen.not_found or gen.response.strip() == ESCALATE_REQUIRED:
        justification = (
            f"request_type={cls.request_type}; product_area={cls.product_area}; status=escalated. "
            "reason=Generator reported not_found (insufficient grounded steps). "
            f"Evidence: {_evidence_block(retrieved)}"
        )
        return TriageResult(
            status="escalated",
            product_area=cls.product_area,
            response="I couldn’t find enough relevant information in the provided knowledge base to answer safely. I’m routing this to a human support agent for review.",
            justification=justification,
            request_type=cls.request_type,
        )

    justification = (
        f"request_type={cls.request_type}; product_area={cls.product_area}; status=replied. "
        "reason=Low-risk and response generated strictly from retrieved context. "
        f"Evidence: {_evidence_block(retrieved)}"
    )
    return TriageResult(
        status="replied",
        product_area=cls.product_area,
        response=gen.response,
        justification=justification,
        request_type=cls.request_type,
    )

