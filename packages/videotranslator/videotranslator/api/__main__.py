"""API server entry: ``python -m videotranslator.api`` (uvicorn)."""

from __future__ import annotations

import os
import sys


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()  # Local .env; injected deployment environment takes precedence.

    from ..bootstrap import build_container
    from .app import create_app

    container = build_container()
    container.seed()
    app = create_app(container)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    sys.exit(main())
