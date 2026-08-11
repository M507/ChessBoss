"""Configure Python logging for the app."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

from chess_boss.config import LoggingConfig


def setup_logging(config: LoggingConfig, session_id: Optional[str] = None) -> str:
    """
    Configure console + rotating file logging.

    Console uses the configured level (default INFO).
    The app log file always captures DEBUG so the full pipeline chain
    is available after the fact without noisy terminals.

    Returns the session id used for this run.
    """
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    console_level = getattr(logging, config.level.upper(), logging.INFO)
    session = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Stable rotating log (latest-ish history across runs).
    rolling = RotatingFileHandler(
        config.logs_dir / config.app_log_filename,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    rolling.setLevel(logging.DEBUG)
    rolling.setFormatter(fmt)
    root.addHandler(rolling)

    # Per-session detailed log.
    session_path = config.logs_dir / f"{session}_session.log"
    session_handler = logging.FileHandler(session_path, encoding="utf-8")
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(fmt)
    root.addHandler(session_handler)

    # Keep third-party noise down unless the user asked for DEBUG globally.
    for noisy in ("mss", "PIL", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if console_level <= logging.DEBUG else logging.WARNING
        )

    logging.getLogger(__name__).info(
        "Logging ready — console=%s file=%s session=%s",
        logging.getLevelName(console_level),
        config.logs_dir / config.app_log_filename,
        session_path,
    )
    return session
