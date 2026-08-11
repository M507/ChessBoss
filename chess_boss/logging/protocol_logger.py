"""Persist move chains in UCI, SAN/PGN, JSONL, and a human-readable chain log."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from chess_boss.config import LoggingConfig

logger = logging.getLogger(__name__)


class ProtocolLogger:
    """
    Multi-sink session logger for the coaching pipeline.

    Writes:
    - `*_moves.uci`  — engine-friendly UCI / FEN markers
    - `*_moves.pgn`  — SAN move text
    - `*_chain.log`  — human timeline (board → move → engine → voice)
    - `*_events.jsonl` — structured events for tooling
    """

    def __init__(self, config: LoggingConfig, session_id: Optional[str] = None) -> None:
        self.config = config
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stamp = self.session_id

        self.uci_path = config.logs_dir / f"{stamp}_{config.uci_filename}"
        self.pgn_path = config.logs_dir / f"{stamp}_{config.protocol_filename}"
        self.chain_path = config.logs_dir / f"{stamp}_chain.log"
        self.events_path = config.logs_dir / f"{stamp}_events.jsonl"

        self._uci = self.uci_path.open("a", encoding="utf-8")
        self._pgn = self.pgn_path.open("a", encoding="utf-8")
        self._chain = self.chain_path.open("a", encoding="utf-8")
        self._events = self.events_path.open("a", encoding="utf-8")
        self._closed = False

        self._write_headers()
        logger.info(
            "Protocol logs ready — UCI=%s PGN=%s chain=%s events=%s",
            self.uci_path.name,
            self.pgn_path.name,
            self.chain_path.name,
            self.events_path.name,
        )

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write_headers(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._uci.write(f"# Chess Boss UCI log session={self.session_id} {now}\n")
        self._uci.write("position startpos\n")
        self._uci.flush()

        self._pgn.write('[Event "Chess Boss Session"]\n')
        self._pgn.write(f'[Site "Local Screen / {self.session_id}"]\n')
        self._pgn.write(f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]\n')
        self._pgn.write('[White "?"]\n')
        self._pgn.write('[Black "?"]\n')
        self._pgn.write('[Result "*"]\n\n')
        self._pgn.flush()

        self._chain.write(f"# Chess Boss chain log session={self.session_id} {now}\n")
        self._chain.write("# format: TIME | KIND | details\n")
        self._chain.flush()

        self.event(
            "session_start",
            {
                "session_id": self.session_id,
                "uci_path": str(self.uci_path),
                "pgn_path": str(self.pgn_path),
            },
        )

    def event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Append one structured JSONL event + mirror important kinds to chain.log."""
        if self._closed:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **(payload or {}),
        }
        self._events.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._events.flush()

    def chain(self, kind: str, message: str) -> None:
        if self._closed:
            return
        line = f"{self._now()} | {kind:<10} | {message}"
        self._chain.write(line + "\n")
        self._chain.flush()
        # Also surface on the app logger so console/session.log stay in sync.
        logger.info("CHAIN %s | %s", kind, message)

    def log_reset(self, fen: str) -> None:
        self._uci.write(f"\n# RESET\nposition fen {fen}\n")
        self._uci.flush()
        self._pgn.write(f"\n; RESET FEN {fen}\n")
        self._pgn.flush()
        self.chain("RESET", f"fen={fen}")
        self.event("reset", {"fen": fen})

    def log_resync(self, fen: str, confidence: float) -> None:
        self._uci.write(f"\n# RESYNC conf={confidence:.2f}\nposition fen {fen}\n")
        self._uci.flush()
        self._pgn.write(f"\n; RESYNC conf={confidence:.2f} FEN {fen}\n")
        self._pgn.flush()
        self.chain("RESYNC", f"conf={confidence:.2f} fen={fen}")
        self.event("resync", {"fen": fen, "confidence": confidence})

    def log_illegal(self, fen: str, status: str, confidence: float) -> None:
        self._uci.write(f"# ILLEGAL conf={confidence:.2f} status={status} fen={fen}\n")
        self._uci.flush()
        self.chain("ILLEGAL", f"conf={confidence:.2f} status={status} fen={fen}")
        self.event(
            "illegal_observation",
            {"fen": fen, "status": status, "confidence": confidence},
        )

    def log_board(
        self,
        fen: str,
        *,
        score: float,
        confidence: float,
        side: str,
        bbox: Optional[tuple] = None,
    ) -> None:
        self.chain(
            "BOARD",
            f"side={side} score={score:.3f} conf={confidence:.3f} fen={fen}",
        )
        self.event(
            "board",
            {
                "fen": fen,
                "score": score,
                "confidence": confidence,
                "side": side,
                "bbox": list(bbox) if bbox else None,
            },
        )

    def log_side(self, side: str, white_at_bottom: bool, reason: str) -> None:
        self.chain(
            "SIDE",
            f"{side} white_at_bottom={white_at_bottom} ({reason})",
        )
        self.event(
            "side",
            {
                "side": side,
                "white_at_bottom": white_at_bottom,
                "reason": reason,
            },
        )

    def log_move(self, uci: str, san: str, fen_after: str, ply: int) -> None:
        self._uci.write(f"{uci}  # ply={ply} san={san} fen={fen_after}\n")
        self._uci.flush()

        if ply % 2 == 1:
            move_no = (ply + 1) // 2
            self._pgn.write(f"{move_no}. {san} ")
        else:
            self._pgn.write(f"{san} ")
        self._pgn.flush()

        self.chain("MOVE", f"ply={ply} UCI={uci} SAN={san} → {fen_after}")
        self.event(
            "move",
            {"ply": ply, "uci": uci, "san": san, "fen_after": fen_after},
        )

    def log_engine_line(
        self,
        pv_uci: str,
        pv_san: str,
        score_cp: Optional[int],
        *,
        best_uci: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> None:
        self._uci.write(
            f"# engine best={best_uci} depth={depth} score_cp={score_cp} pv {pv_uci}\n"
        )
        self._uci.flush()
        self._pgn.write(f"\n; engine {{%eval {score_cp}}} depth={depth} {pv_san}\n")
        self._pgn.flush()
        self.chain(
            "ENGINE",
            f"best={best_uci or '-'} depth={depth} cp={score_cp} pv_san={pv_san}",
        )
        self.event(
            "engine",
            {
                "best_uci": best_uci,
                "depth": depth,
                "score_cp": score_cp,
                "pv_uci": pv_uci,
                "pv_san": pv_san,
            },
        )

    def log_voice(self, text: str, voice: str) -> None:
        # Strip macOS say silence markers for readable logs.
        readable = text.replace("[[slnc 160]]", "").replace("[[slnc 240]]", "")
        readable = " ".join(readable.split())
        self.chain("VOICE", f"[{voice}] {readable}")
        self.event("voice", {"voice": voice, "text": readable, "raw": text})

    def log_screen(self, label: str, index: int) -> None:
        self.chain("SCREEN", f"index={index} {label}")
        self.event("screen", {"index": index, "label": label})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.chain("SESSION", "end")
            self.event("session_end", {"session_id": self.session_id})
            self._pgn.write("*\n")
        finally:
            for fh in (self._pgn, self._uci, self._chain, self._events):
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
        logger.info(
            "Protocol logs closed — %s | %s | %s | %s",
            self.uci_path.name,
            self.pgn_path.name,
            self.chain_path.name,
            self.events_path.name,
        )
