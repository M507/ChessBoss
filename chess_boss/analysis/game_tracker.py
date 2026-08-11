"""Track position changes and emit UCI/SAN move chains."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chess

from chess_boss.logging.protocol_logger import ProtocolLogger

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    fen: str
    board: chess.Board
    confidence: float


@dataclass
class MoveEvent:
    move: chess.Move
    uci: str
    san: str
    fen_before: str
    fen_after: str


@dataclass
class GameTracker:
    """
    Maintain a live chess.Board, infer moves between vision snapshots,
    and mirror the game into protocol logs.
    """

    protocol: ProtocolLogger
    board: chess.Board = field(default_factory=chess.Board)
    last_fen_placement: Optional[str] = None
    move_history_uci: List[str] = field(default_factory=list)
    move_history_san: List[str] = field(default_factory=list)
    ply: int = 0

    def reset(self, board: Optional[chess.Board] = None) -> None:
        candidate = board if board is not None else chess.Board()
        if board is not None and not candidate.is_valid():
            logger.warning(
                "Reset ignored illegal board status=%s fen=%s",
                candidate.status(),
                candidate.fen(),
            )
            candidate = chess.Board()
        self.board = candidate
        self.last_fen_placement = self.board.board_fen()
        self.move_history_uci.clear()
        self.move_history_san.clear()
        self.ply = 0
        self.protocol.log_reset(self.board.fen())
        logger.info("GameTracker reset FEN=%s", self.board.fen())

    def update(self, observed: chess.Board, confidence: float) -> Optional[MoveEvent]:
        """
        Sync tracker with an observed board.

        Returns a MoveEvent when a single legal transition is detected.
        """
        if not observed.is_valid():
            # Vision glitches (e.g. rook→pawn) must never poison game state / Stockfish.
            status = str(observed.status())
            logger.warning(
                "Ignoring illegal observation (conf=%.2f) status=%s fen=%s",
                confidence,
                status,
                observed.fen(),
            )
            try:
                self.protocol.log_illegal(observed.fen(), status, confidence)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Illegal-observation protocol log failed: %s", exc)
            return None

        observed_placement = observed.board_fen()
        if self.last_fen_placement is None:
            self.reset(observed)
            return None

        if observed_placement == self.board.board_fen():
            return None

        # Try to find a single legal move explaining the change.
        move = self._infer_move(self.board, observed)
        if move is not None:
            fen_before = self.board.fen()
            san = self.board.san(move)
            self.board.push(move)
            event = MoveEvent(
                move=move,
                uci=move.uci(),
                san=san,
                fen_before=fen_before,
                fen_after=self.board.fen(),
            )
            self.move_history_uci.append(event.uci)
            self.move_history_san.append(event.san)
            self.ply += 1
            self.last_fen_placement = self.board.board_fen()
            self.protocol.log_move(event.uci, event.san, event.fen_after, self.ply)
            logger.info(
                "MOVE ply=%d UCI=%s SAN=%s → %s",
                self.ply,
                event.uci,
                event.san,
                event.fen_after,
            )
            return event

        # Multi-move jump or recognition glitch — resync only if the new board is legal.
        logger.warning(
            "Position jump (conf=%.2f). Resyncing board. old=%s new=%s",
            confidence,
            self.board.board_fen(),
            observed_placement,
        )
        self.board = observed
        self.last_fen_placement = observed_placement
        self.protocol.log_resync(self.board.fen(), confidence)
        return None

    @staticmethod
    def _infer_move(before: chess.Board, after: chess.Board) -> Optional[chess.Move]:
        target = after.board_fen()
        for move in before.legal_moves:
            before.push(move)
            match = before.board_fen() == target
            before.pop()
            if match:
                return move
        return None

    def chain_uci(self) -> str:
        return " ".join(self.move_history_uci)

    def chain_san(self) -> str:
        parts: List[str] = []
        for i, san in enumerate(self.move_history_san):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}. {san}")
            else:
                parts.append(san)
        return " ".join(parts)
