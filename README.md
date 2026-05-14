# Deterministic Support Triage Agent (Offline RAG)

## Problem Summary
This project processes `support_tickets/support_tickets.csv` and produces a deterministic triage output at `support_tickets/output.csv` with the required columns:
`status, product_area, response, justification, request_type`.

Strict constraints:
- Uses **only** the local corpus under `data/` (`hackerrank/`, `claude/`, `visa/`)
- No external APIs or web access
- Responses are generated **only from retrieved corpus context**
- If uncertain, risky, ambiguous, or weakly grounded → **escalate**

## Architecture Overview
- **Preprocessor** (`code/preprocessor.py`): cleans `subject + issue`, normalizes `company`, detects multiple intents.
- **Classifier** (`code/classifier.py`): deterministic rules to set:
  - `request_type ∈ {product_issue, feature_request, bug, invalid}`
  - `product_area` (short label)
- **Risk & Escalation Engine** (`code/escalation.py`): hard escalation rules + retrieval grounding gate.
- **Retriever (STRICT RAG)** (`code/retriever.py`): chunks all `.md` docs in `data/`, TF‑IDF + cosine similarity, returns top‑k chunks.
- **Answer Generator** (`code/generator.py`): calls **local Ollama only** (if available) and forces JSON output; answers strictly from retrieved context; otherwise returns `not found` → escalate.
- **CSV Writer / Orchestrator** (`code/main.py`): deterministic read → triage → write.

## Design Decisions
- **TF‑IDF retrieval**: deterministic, fast, fully offline, and works well for support KB text.
- **Deterministic pipeline**: temperature \(=0\), seed \(=0\), and stable retrieval/sorting enable reproducibility.
- **Aggressive escalation**: wrong answers are worse than escalation; high‑risk domains (fraud/billing/account access/Visa) are escalated by rule.

## Setup Instructions
### Python
- Python 3.10+ recommended

### Install dependencies
From repo root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r code/requirements.txt
```

### Ollama (optional but recommended)
This agent can use **local** Ollama for response generation.
- Set environment variables (examples):
  - `OLLAMA_MODEL=llama3` (or `mistral`, `phi`, etc.)
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_SEED=0`

If Ollama is not running, the generator returns `not found` and the agent escalates (failsafe).

### Run
From repo root:

```bash
python3 code/main.py
```

Outputs:
- `support_tickets/output.csv`

## Output Format
Each output row contains:
- `status`: `replied` or `escalated`
- `product_area`: short label (e.g. `billing`, `account access`, `assessments`, etc.)
- `response`: user-facing response (grounded in retrieved context) or escalation message
- `justification`: internal rationale + retrieved evidence snippets
- `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`

## Limitations
- Ambiguous or multi-intent tickets are escalated.
- Retrieval quality depends on overlap between ticket language and the corpus.
- No external knowledge: if not in `data/`, the safe outcome is escalation.

## Future Improvements
- Local embeddings retriever (still offline) for semantic matching.
- Better multi-intent detection and decomposition.
- More granular confidence scoring using retrieval margins and rule signals.

