"""Chat with customer support in the terminal.

Usage:  python scripts/chat.py [--agent-view] [--no-llm]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticket_router.chat_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
