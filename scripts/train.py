"""Train the categorization, resolution-time and escalation models.

Usage:  python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticket_router import config  # noqa: E402
from ticket_router.training import save_models, train_models  # noqa: E402


def main() -> None:
    models, metrics = train_models()
    save_models(models, metrics)
    print(f"Saved models to {config.ARTIFACTS_DIR}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
