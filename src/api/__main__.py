"""``python -m src.api`` — run the FastAPI app with uvicorn."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("ANEXUS_API_HOST", "127.0.0.1"),
        port=int(os.getenv("ANEXUS_API_PORT", "8000")),
        reload=os.getenv("ANEXUS_API_RELOAD", "0").strip().lower()
        in ("1", "true", "yes"),
        log_level=os.getenv("ANEXUS_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
