from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.0
    seed: int = 0
    timeout_s: int = 60


def ollama_available(cfg: OllamaConfig) -> bool:
    try:
        req = urllib.request.Request(f"{cfg.base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def generate_json(cfg: OllamaConfig, system: str, user: str) -> Optional[Dict[str, Any]]:
    """
    Calls Ollama's local HTTP API with temperature=0 for deterministic generation.
    Returns parsed JSON object if model output is valid JSON, else None.
    """
    payload = {
        "model": cfg.model,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {"temperature": cfg.temperature, "seed": cfg.seed},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{cfg.base_url}/api/generate",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None

    try:
        outer = json.loads(body)
        text = outer.get("response", "")
        return json.loads(text)
    except Exception:
        return None


def generate_text(cfg: OllamaConfig, system: str, user: str) -> Optional[str]:
    """
    Calls Ollama's local HTTP API with temperature=0 for deterministic generation.
    Returns raw text output from the model, or None on error.
    """
    payload = {
        "model": cfg.model,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {"temperature": cfg.temperature, "seed": cfg.seed},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{cfg.base_url}/api/generate",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None

    try:
        outer = json.loads(body)
        text = outer.get("response", "")
        return str(text)
    except Exception:
        return None

