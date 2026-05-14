from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    company: str
    source_path: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _iter_md_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def _read_text(path: Path) -> str:
    # Deterministic: ignore decode errors consistently.
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    # Split on blank lines, then pack into chunks up to max_chars.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for p in paras:
        if size + len(p) + 2 > max_chars and buf:
            chunk = "\n\n".join(buf).strip()
            chunks.append(chunk)
            # Overlap: carry last overlap chars of previous chunk into next buffer
            tail = chunk[-overlap:] if overlap > 0 else ""
            buf = [tail] if tail else []
            size = len(tail)
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return [c for c in chunks if c]


def build_corpus(data_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    repo_root = data_dir.parent
    for company in ["hackerrank", "claude", "visa"]:
        root = data_dir / company
        if not root.exists():
            continue
        for fp in sorted(_iter_md_files(root)):
            raw = _read_text(fp)
            for i, chunk_text in enumerate(_chunk_text(raw)):
                try:
                    rel = fp.relative_to(repo_root).as_posix()
                except Exception:
                    rel = fp.as_posix()
                chunks.append(
                    Chunk(
                        chunk_id=f"{company}:{fp.as_posix()}:{i}",
                        company=company,
                        source_path=str(rel),
                        text=chunk_text,
                    )
                )
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self._chunks = list(chunks)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=60000,
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

    @property
    def chunks(self) -> Sequence[Chunk]:
        return self._chunks

    def retrieve(
        self,
        query: str,
        k: int = 5,
        company: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not query.strip() or not self._chunks:
            return []

        idxs = list(range(len(self._chunks)))
        if company in {"hackerrank", "claude", "visa"}:
            idxs = [i for i, c in enumerate(self._chunks) if c.company == company]
            if not idxs:
                return []

        qv = self._vectorizer.transform([query])
        mat = self._matrix[idxs, :]
        sims = cosine_similarity(qv, mat).flatten()

        scored: List[Tuple[int, float]] = list(zip(idxs, sims.tolist()))
        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[: max(0, k)]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(s)) for i, s in top]


def is_retrieval_weak(retrieved: Sequence[RetrievedChunk], min_score: float = 0.08) -> bool:
    if not retrieved:
        return True
    best = retrieved[0].score
    if math.isnan(best) or best < min_score:
        return True
    # Also treat retrieval as weak if all chunks are near-zero.
    if all((r.score < (min_score * 0.7)) for r in retrieved):
        return True
    return False

