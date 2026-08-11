"""Stockfish wrapper via python-chess UCI protocol."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chess
import chess.engine

from chess_boss.config import EngineConfig

logger = logging.getLogger(__name__)


@dataclass
class EngineAdvice:
    """Best-move recommendation from the engine."""

    best_move: Optional[chess.Move]
    best_move_uci: Optional[str]
    ponder_uci: Optional[str]
    score_cp: Optional[int]
    mate: Optional[int]
    depth: int
    pv_uci: List[str] = field(default_factory=list)
    pv_san: List[str] = field(default_factory=list)


def discover_stockfish(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a Stockfish binary on PATH or common install locations."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        logger.warning("Configured stockfish path not found: %s", explicit)

    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env

    which = shutil.which("stockfish")
    if which:
        return which

    candidates = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        str(Path.home() / ".local" / "bin" / "stockfish"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def empty_advice(label_depth: int = 0) -> EngineAdvice:
    return EngineAdvice(
        best_move=None,
        best_move_uci=None,
        ponder_uci=None,
        score_cp=None,
        mate=None,
        depth=label_depth,
        pv_uci=[],
        pv_san=[],
    )


class StockfishEngine:
    """Thin UCI session around Stockfish with crash recovery."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self.path = discover_stockfish(config.path)

    @property
    def available(self) -> bool:
        return self.path is not None

    @property
    def is_running(self) -> bool:
        return self._engine is not None

    def start(self) -> None:
        if not self.path:
            raise FileNotFoundError(
                "Stockfish not found. Install it (e.g. `brew install stockfish`) "
                "or set STOCKFISH_PATH / --stockfish."
            )
        # Ensure we never leak a half-dead handle.
        self._force_drop()
        logger.info("Starting Stockfish at %s", self.path)
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        opts = {
            "Threads": self.config.threads,
            "Hash": self.config.hash_mb,
        }
        if self.config.skill_level is not None:
            opts["Skill Level"] = self.config.skill_level
        try:
            self._engine.configure(opts)
        except chess.engine.EngineError as exc:
            logger.warning("Could not apply all engine options: %s", exc)

    def close(self) -> None:
        if self._engine is None:
            return
        engine = self._engine
        self._engine = None
        try:
            engine.quit()
            logger.info("Stockfish closed")
        except chess.engine.EngineError as exc:
            logger.warning("Stockfish quit after crash/death: %s", exc)
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass

    def _force_drop(self) -> None:
        if self._engine is None:
            return
        engine = self._engine
        self._engine = None
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass

    def restart(self) -> bool:
        """Restart after a crash. Returns True if running again."""
        logger.warning("Restarting Stockfish…")
        try:
            self.start()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Stockfish restart failed: %s", exc)
            self._engine = None
            return False

    def __enter__(self) -> "StockfishEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def analyse(self, board: chess.Board) -> EngineAdvice:
        if self._engine is None:
            raise RuntimeError("Engine not started")

        if not board.is_valid():
            logger.warning(
                "Refusing to analyse illegal position status=%s fen=%s",
                board.status(),
                board.fen(),
            )
            return empty_advice()

        limit = (
            chess.engine.Limit(time=self.config.movetime_ms / 1000.0)
            if self.config.movetime_ms
            else chess.engine.Limit(depth=self.config.depth)
        )

        try:
            return self._analyse_once(board, limit)
        except chess.engine.EngineTerminatedError as exc:
            logger.error("Stockfish died during analyse: %s", exc)
            if not self.restart():
                return empty_advice()
            try:
                return self._analyse_once(board, limit)
            except chess.engine.EngineError as exc2:
                logger.error("Stockfish failed after restart: %s", exc2)
                return empty_advice()
        except chess.engine.EngineError as exc:
            logger.error("Stockfish analyse error: %s", exc)
            return empty_advice()

    def _analyse_once(
        self, board: chess.Board, limit: chess.engine.Limit
    ) -> EngineAdvice:
        assert self._engine is not None
        info = self._engine.analyse(board, limit, info=chess.engine.INFO_ALL)
        score = info.get("score")
        pov = score.pov(board.turn) if score is not None else None
        score_cp = pov.score(mate_score=100000) if pov is not None else None
        mate = pov.mate() if pov is not None else None

        pv: List[chess.Move] = list(info.get("pv") or [])
        best = pv[0] if pv else None
        if best is None:
            result = self._engine.play(board, limit)
            best = result.move
            pv = [best] if best else []

        pv_uci = [m.uci() for m in pv]
        tmp = board.copy(stack=False)
        pv_san: List[str] = []
        for move in pv:
            try:
                pv_san.append(tmp.san(move))
                tmp.push(move)
            except ValueError:
                pv_san.append(move.uci())
                break

        advice = EngineAdvice(
            best_move=best,
            best_move_uci=best.uci() if best else None,
            ponder_uci=pv_uci[1] if len(pv_uci) > 1 else None,
            score_cp=score_cp,
            mate=mate,
            depth=int(info.get("depth") or 0),
            pv_uci=pv_uci,
            pv_san=pv_san,
        )
        logger.info(
            "Engine advice: best=%s score_cp=%s mate=%s depth=%s pv_san=%s",
            advice.best_move_uci,
            advice.score_cp,
            advice.mate,
            advice.depth,
            " ".join(advice.pv_san[:6]),
        )
        return advice
