"""Detect which color the local player is playing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from chess_boss.detection.piece_classifier import BoardOccupation

logger = logging.getLogger(__name__)


class PlayerSide(str, Enum):
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


@dataclass
class SideResult:
    side: PlayerSide
    white_at_bottom: bool
    confidence: float
    reason: str


class SideDetector:
    """
    Infer orientation and "our" side.

    Heuristics (in order):
    1. Explicit override from config (handled by caller).
    2. Piece placement: white pieces denser on bottom ranks ⇒ White at bottom.
    3. Coordinate glyph brightness near board edge (optional future).
    """

    def detect(
        self,
        warped_bgr: np.ndarray,
        occupation: Optional[BoardOccupation] = None,
    ) -> SideResult:
        white_bottom_score = 0.0
        black_bottom_score = 0.0

        if occupation is not None:
            for r in range(8):
                for c in range(8):
                    code = occupation.grid[r][c]
                    if code is None:
                        continue
                    weight = 1.5 if code[1] == "K" else 1.0
                    if code.startswith("w"):
                        if r >= 6:
                            white_bottom_score += weight
                        if r <= 1:
                            black_bottom_score += weight
                    else:
                        if r <= 1:
                            white_bottom_score += weight
                        if r >= 6:
                            black_bottom_score += weight

        # Brightness prior: white pieces are brighter.
        bottom = warped_bgr[int(warped_bgr.shape[0] * 0.75) :, :]
        top = warped_bgr[: int(warped_bgr.shape[0] * 0.25), :]
        bright_bottom = float(np.mean(cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)))
        bright_top = float(np.mean(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)))
        if bright_bottom > bright_top + 8:
            white_bottom_score += 1.0
        elif bright_top > bright_bottom + 8:
            black_bottom_score += 1.0

        if white_bottom_score == 0 and black_bottom_score == 0:
            return SideResult(
                side=PlayerSide.UNKNOWN,
                white_at_bottom=True,
                confidence=0.0,
                reason="no_signal",
            )

        white_at_bottom = white_bottom_score >= black_bottom_score
        total = white_bottom_score + black_bottom_score
        conf = abs(white_bottom_score - black_bottom_score) / max(total, 1e-6)

        # Convention: the local player sits at the bottom of the board view.
        side = PlayerSide.WHITE if white_at_bottom else PlayerSide.BLACK
        reason = (
            f"pieces+brightness white_bottom={white_bottom_score:.1f} "
            f"black_bottom={black_bottom_score:.1f}"
        )
        logger.info(
            "Side detected: %s (white_at_bottom=%s, conf=%.2f, %s)",
            side.value,
            white_at_bottom,
            conf,
            reason,
        )
        return SideResult(
            side=side,
            white_at_bottom=white_at_bottom,
            confidence=float(conf),
            reason=reason,
        )