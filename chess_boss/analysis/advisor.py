"""Turn-aware move guidance using Stockfish."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

from chess_boss.detection.side_detector import PlayerSide
from chess_boss.engine.stockfish_engine import EngineAdvice, StockfishEngine

logger = logging.getLogger(__name__)


@dataclass
class GuideHint:
    """Visual + textual guidance for the next move."""

    our_turn: bool
    advice: Optional[EngineAdvice]
    from_square: Optional[chess.Square]
    to_square: Optional[chess.Square]
    label: str


class MoveAdvisor:
    """Ask Stockfish when it is our turn (or always, if configured)."""

    def __init__(
        self,
        engine: StockfishEngine,
        guide_when_our_turn: bool = True,
    ) -> None:
        self.engine = engine
        self.guide_when_our_turn = guide_when_our_turn
        self._last_fen: Optional[str] = None
        self._last_hint: Optional[GuideHint] = None

    def advise(self, board: chess.Board, our_side: PlayerSide) -> GuideHint:
        our_color = {
            PlayerSide.WHITE: chess.WHITE,
            PlayerSide.BLACK: chess.BLACK,
        }.get(our_side)

        our_turn = our_color is not None and board.turn == our_color
        if our_side == PlayerSide.UNKNOWN:
            our_turn = True  # still offer a best move for analysis

        fen = board.fen()
        if (
            self._last_hint is not None
            and self._last_fen == fen
            and self._last_hint.our_turn == our_turn
        ):
            return self._last_hint

        if self.guide_when_our_turn and not our_turn:
            hint = GuideHint(
                our_turn=False,
                advice=None,
                from_square=None,
                to_square=None,
                label="waiting for opponent",
            )
            self._last_fen = fen
            self._last_hint = hint
            return hint

        if not board.is_valid():
            hint = GuideHint(
                our_turn=our_turn,
                advice=None,
                from_square=None,
                to_square=None,
                label="invalid board",
            )
            # Do not cache forever on the same bad fen if we want retries after
            # vision recovers — but caching avoids spam while glitch persists.
            self._last_fen = fen
            self._last_hint = hint
            logger.warning("Skip advice; illegal board status=%s", board.status())
            return hint

        if not self.engine.available or not self.engine.is_running:
            hint = GuideHint(
                our_turn=our_turn,
                advice=None,
                from_square=None,
                to_square=None,
                label="engine offline",
            )
            self._last_fen = fen
            self._last_hint = hint
            return hint

        try:
            advice = self.engine.analyse(board)
        except (chess.engine.EngineError, RuntimeError) as exc:
            logger.error("Advise failed: %s", exc)
            hint = GuideHint(
                our_turn=our_turn,
                advice=None,
                from_square=None,
                to_square=None,
                label="engine error",
            )
            self._last_fen = fen
            self._last_hint = hint
            return hint

        if advice.best_move is None and not advice.pv_san:
            hint = GuideHint(
                our_turn=our_turn,
                advice=None,
                from_square=None,
                to_square=None,
                label="no-move",
            )
            self._last_fen = fen
            self._last_hint = hint
            return hint

        from_sq = advice.best_move.from_square if advice.best_move else None
        to_sq = advice.best_move.to_square if advice.best_move else None
        label = advice.best_move_uci or "no-move"
        if advice.pv_san:
            label = advice.pv_san[0]
        hint = GuideHint(
            our_turn=our_turn,
            advice=advice,
            from_square=from_sq,
            to_square=to_sq,
            label=label,
        )
        self._last_fen = fen
        self._last_hint = hint
        logger.info("Guide hint: %s (our_turn=%s)", label, our_turn)
        return hint
