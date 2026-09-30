"""Central configuration: file locations, thresholds and LLM settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(os.getenv("TICKET_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

HISTORICAL_TICKETS_CSV = DATA_DIR / "historical_tickets.csv"
AGENTS_JSON = DATA_DIR / "agents.json"
RESPONSE_TEMPLATES_JSON = DATA_DIR / "response_templates.json"

CATEGORIZER_MODEL_PATH = ARTIFACTS_DIR / "categorizer.joblib"
RESOLUTION_MODEL_PATH = ARTIFACTS_DIR / "resolution_model.joblib"
ESCALATION_MODEL_PATH = ARTIFACTS_DIR / "escalation_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Below this probability the ML categorizer defers to the keyword categorizer.
ML_CONFIDENCE_THRESHOLD = 0.55

# Escalation probability above which a ticket is flagged as "likely to escalate".
ESCALATION_THRESHOLD = 0.5

# Complexity score cut-offs (score is in [0, 1]).
COMPLEXITY_MEDIUM_THRESHOLD = 0.30
COMPLEXITY_HIGH_THRESHOLD = 0.60

# Number of pre-written responses suggested per ticket.
TOP_K_SUGGESTIONS = 3

# LLM drafting. "auto" enables Claude only when credentials are present in the
# environment; "true" forces it on (SDK resolves credentials); "false" disables it.
LLM_MODE = os.getenv("TICKET_LLM_MODE", "auto").lower()
LLM_MODEL = os.getenv("TICKET_LLM_MODEL", "claude-opus-5")
LLM_TIMEOUT_SECONDS = float(os.getenv("TICKET_LLM_TIMEOUT", "60"))


def llm_enabled() -> bool:
    if LLM_MODE == "true":
        return True
    if LLM_MODE == "false":
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
