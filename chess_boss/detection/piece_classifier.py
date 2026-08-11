"""Classify pieces on a warped board using template matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess
import cv2
import numpy as np

from chess_boss.config import DetectionConfig
from chess_boss.detection.templates import PIECE_CODES, TemplateBank, square_image

logger = logging.getLogger(__name__)

CODE_TO_PIECE = {
    "wP": chess.Piece.from_symbol("P"),
    "wN": chess.Piece.from_symbol("N"),
    "wB": chess.Piece.from_symbol("B"),
    "wR": chess.Piece.from_symbol("R"),
    "wQ": chess.Piece.from_symbol("Q"),
    "wK": chess.Piece.from_symbol("K"),
    "bP": chess.Piece.from_symbol("p"),
    "bN": chess.Piece.from_symbol("n"),
    "bB": chess.Piece.from_symbol("b"),
    "bR": chess.Piece.from_symbol("r"),
    "bQ": chess.Piece.from_symbol("q"),
    "bK": chess.Piece.from_symbol("k"),
}


@dataclass
class BoardOccupation:
    """Recognized board contents."""

    grid: List[List[Optional[str]]]  # 8x8 piece codes or None
    fen_placement: str
    confidence: float


class PieceClassifier:
    """Template-match each square against the piece bank."""

    def __init__(self, config: DetectionConfig, bank: TemplateBank) -> None:
        self.config = config
        self.bank = bank

    def classify(self, warped_bgr: np.ndarray, white_at_bottom: bool = True) -> BoardOccupation:
        grid: List[List[Optional[str]]] = [[None] * 8 for _ in range(8)]
        scores: List[float] = []

        for r in range(8):
            for c in range(8):
                tile = square_image(warped_bgr, r, c)
                code, score = self._classify_tile(tile, r, c)
                # Chess rule: pawns never live on the 1st/8th rank (image rows 7/0
                # when White is at the bottom).
                if code and code[1] == "P" and r in (0, 7):
                    code = None
                    score = min(score, 0.2)
                grid[r][c] = code
                scores.append(score)

        if white_at_bottom:
            fen_rows = grid
        else:
            fen_rows = [[grid[7 - r][7 - c] for c in range(8)] for r in range(8)]

        fen_placement = self._grid_to_fen_placement(fen_rows)
        conf = float(np.mean(scores)) if scores else 0.0
        logger.debug("Classified board FEN placement=%s conf=%.3f", fen_placement, conf)
        return BoardOccupation(grid=grid, fen_placement=fen_placement, confidence=conf)

    def to_board(self, occupation: BoardOccupation, turn: chess.Color = chess.WHITE) -> chess.Board:
        turn_flag = "w" if turn == chess.WHITE else "b"
        fen = f"{occupation.fen_placement} {turn_flag} - - 0 1"
        board = chess.Board(fen)
        if board.is_valid():
            return board
        fen_castle = f"{occupation.fen_placement} {turn_flag} KQkq - 0 1"
        board_castle = chess.Board(fen_castle)
        if board_castle.is_valid():
            return board_castle
        logger.warning(
            "Built illegal board from vision status=%s fen=%s",
            board.status(),
            board.fen(),
        )
        return board

    def _classify_tile(self, tile: np.ndarray, row: int, col: int) -> Tuple[Optional[str], float]:
        # Match pieces first. Strong hits win even on last-move highlights
        # (destination squares like c3) and must not be wiped by "empty" logic
        # when light squares look slightly greenish (common with Black at bottom).
        best_code: Optional[str] = None
        best_score = -1.0
        tile_resized_cache: dict = {}

        for code in PIECE_CODES:
            for tmpl in self.bank.templates.get(code, []):
                key = tmpl.shape[:2]
                if key not in tile_resized_cache:
                    tile_resized_cache[key] = cv2.resize(tile, (tmpl.shape[1], tmpl.shape[0]))
                resized = tile_resized_cache[key]
                result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_code = code

        soft = self.config.template_match_threshold
        hard_min = max(0.35, soft - 0.12)

        if best_code is not None and best_score >= soft:
            return best_code, best_score

        highlighted = self._is_move_highlight(tile)
        if self._looks_empty(tile, row, col, highlighted=highlighted):
            return None, 1.0

        # Near-threshold: still trust the template (fixes h1 rook ~0.51).
        if best_code is not None and best_score >= hard_min:
            return best_code, best_score

        # Weak match on a highlighted square → almost always empty origin square.
        if highlighted and best_score < soft:
            return None, max(best_score, 0.0)

        # Very weak match → empty (do NOT invent pawns from brightness).
        return None, max(best_score, 0.0)

    def _looks_empty(
        self,
        tile: np.ndarray,
        row: int,
        col: int,
        highlighted: bool = False,
    ) -> bool:
        ref = self.bank.empty_light if (row + col) % 2 == 0 else self.bank.empty_dark
        if ref is None:
            return False
        ref_r = cv2.resize(ref, (tile.shape[1], tile.shape[0]))
        h, w = tile.shape[:2]
        a = tile[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5].astype(np.float32)
        b = ref_r[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5].astype(np.float32)
        diff = float(np.mean(np.abs(a - b)))
        threshold = self.config.empty_diff_threshold
        if highlighted:
            # Lichess green last-move overlay shifts square color a lot.
            threshold = max(threshold, 55.0)
        return diff < threshold

    @staticmethod
    def _is_move_highlight(tile: np.ndarray) -> bool:
        """Detect Lichess-style green/yellow last-move overlays."""
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        # Tighter band + higher coverage so normal light squares / white
        # pawns are not mistaken for move highlights (esp. Black-at-bottom).
        lower = np.array([28, 50, 90], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = float(mask.mean()) / 255.0
        return ratio > 0.22

    @staticmethod
    def _grid_to_fen_placement(rows: List[List[Optional[str]]]) -> str:
        parts = []
        for r in range(8):
            empty = 0
            rank = []
            for c in range(8):
                code = rows[r][c]
                if code is None:
                    empty += 1
                    continue
                if empty:
                    rank.append(str(empty))
                    empty = 0
                piece = CODE_TO_PIECE[code]
                rank.append(piece.symbol())
            if empty:
                rank.append(str(empty))
            parts.append("".join(rank))
        return "/".join(parts)
