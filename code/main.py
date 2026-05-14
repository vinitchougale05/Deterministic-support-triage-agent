from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from agent import triage_one
from ollama_client import OllamaConfig
from retriever import TfidfRetriever, build_corpus


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["status", "product_area", "response", "justification", "request_type"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    chunks = build_corpus(DATA_DIR)
    retriever = TfidfRetriever(chunks)

    ollama_cfg = OllamaConfig(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "llama3"),
        temperature=0.0,
        seed=int(os.environ.get("OLLAMA_SEED", "0")),
        timeout_s=int(os.environ.get("OLLAMA_TIMEOUT_S", "60")),
    )

    k = int(os.environ.get("RETRIEVER_TOP_K", "5"))
    per_ticket_sleep = float(os.environ.get("PER_TICKET_SLEEP_S", "0"))

    out_rows = []
    for row in _read_rows(INPUT_CSV):
        # Input headers may be title-cased per provided CSVs (Issue/Subject/Company).
        subject = row.get("subject") or row.get("Subject") or ""
        issue = row.get("issue") or row.get("Issue") or ""
        company = row.get("company") or row.get("Company")

        result = triage_one(
            subject=subject,
            issue=issue,
            company=company,
            retriever=retriever,
            ollama_cfg=ollama_cfg,
            k=k,
        )
        out_rows.append(
            {
                "status": result.status,
                "product_area": result.product_area,
                "response": result.response,
                "justification": result.justification,
                "request_type": result.request_type,
            }
        )

        if per_ticket_sleep > 0:
            time.sleep(per_ticket_sleep)

    _write_rows(OUTPUT_CSV, out_rows)


if __name__ == "__main__":
    main()
