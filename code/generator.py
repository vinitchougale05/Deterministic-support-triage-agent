from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ollama_client import OllamaConfig, generate_text, ollama_available
from retriever import RetrievedChunk


@dataclass(frozen=True)
class GeneratedAnswer:
    response: str
    not_found: bool


ESCALATE_REQUIRED = "ESCALATE_REQUIRED"

_SYSTEM = (
    "You are a helpful, polite support agent.\n"
    "You MUST answer the user's question using ONLY the provided context chunks.\n"
    "Do NOT use any outside knowledge.\n"
    f"If the context does not contain the answer, output exactly: {ESCALATE_REQUIRED}\n"
    "Keep the response concise and directly address the user.\n"
    "Return ONLY the final answer text (no JSON, no markdown, no extra labels)."
)


def _format_context(retrieved: Sequence[RetrievedChunk]) -> str:
    parts: List[str] = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(
            "\n".join(
                [
                    f"[DOC {i}] source={r.chunk.source_path}",
                    r.chunk.text.strip(),
                ]
            ).strip()
        )
    return "\n\n".join(parts).strip()


def generate_response_from_context(
    issue_text: str,
    retrieved: Sequence[RetrievedChunk],
    cfg: OllamaConfig,
) -> GeneratedAnswer:
    context = _format_context(retrieved)
    user = (
        "CONTEXT (retrieved documents):\n"
        f"{context}\n\n"
        "USER ISSUE:\n"
        f"{issue_text.strip()}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using ONLY the context.\n"
        f"- If missing, output exactly: {ESCALATE_REQUIRED}\n"
    )

    if not ollama_available(cfg):
        return GeneratedAnswer(response=ESCALATE_REQUIRED, not_found=True)

    text = generate_text(cfg, system=_SYSTEM, user=user)
    if not text:
        return GeneratedAnswer(response=ESCALATE_REQUIRED, not_found=True)

    resp = str(text).replace("\r", " ").replace("\n", " ").strip()
    if not resp:
        return GeneratedAnswer(response=ESCALATE_REQUIRED, not_found=True)
    if resp == ESCALATE_REQUIRED:
        return GeneratedAnswer(response=ESCALATE_REQUIRED, not_found=True)
    return GeneratedAnswer(response=resp, not_found=False)

