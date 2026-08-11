"""Locate a chessboard in a screenshot and warp it to a canonical square."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from chess_boss.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectedBoard:
    """A board found in screen space."""

    corners: np.ndarray  # shape (4, 2) float32, order TL, TR, BR, BL
    warped: np.ndarray  # board_size x board_size BGR
    score: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h in source image


class BoardDetector:
    """
    Detect chessboards using contour candidates scored by checkerboard energy.

    Tuned against Lichess-style boards (see examples/).
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def detect(self, image_bgr: np.ndarray) -> Optional[DetectedBoard]:
        if image_bgr is None or image_bgr.size == 0:
            return None

        h, w = image_bgr.shape[:2]
        area = float(h * w)
        min_area = area * self.config.min_board_area_ratio
        max_area = area * self.config.max_board_area_ratio

        candidates = self._quad_candidates(image_bgr, min_area, max_area)
        if not candidates:
            # Fallback: color-clustered board blob (Lichess beige/brown).
            candidates = self._color_blob_candidates(image_bgr, min_area, max_area)

        best: Optional[DetectedBoard] = None
        for corners in candidates:
            warped = self._warp(image_bgr, corners, self.config.board_size)
            score = self._checkerboard_score(warped)
            if score < 0.18:
                continue
            x, y, bw, bh = cv2.boundingRect(corners.astype(np.float32))
            candidate = DetectedBoard(
                corners=corners,
                warped=warped,
                score=score,
                bbox=(x, y, bw, bh),
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            logger.debug(
                "Board detected score=%.3f bbox=%s",
                best.score,
                best.bbox,
            )
        return best

    def _quad_candidates(
        self,
        image_bgr: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads: List[np.ndarray] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            area = cv2.contourArea(approx)
            if area < min_area or area > max_area:
                continue
            corners = approx.reshape(4, 2).astype(np.float32)
            ordered = self._order_corners(corners)
            if not self._is_roughly_square(ordered):
                continue
            quads.append(ordered)

        # Largest first — boards dominate the UI.
        quads.sort(key=lambda c: -cv2.contourArea(c.astype(np.float32)))
        return quads[:12]

    def _color_blob_candidates(
        self,
        image_bgr: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> List[np.ndarray]:
        """Find beige/brown board regions typical of Lichess."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Warm wood tones (light + dark squares).
        lower = np.array([8, 25, 70], dtype=np.uint8)
        upper = np.array([35, 200, 245], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            ordered = self._order_corners(box)
            if not self._is_roughly_square(ordered, tol=0.35):
                continue
            quads.append(ordered)
        quads.sort(key=lambda c: -cv2.contourArea(c.astype(np.float32)))
        return quads[:8]

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order points as TL, TR, BR, BL."""
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _is_roughly_square(corners: np.ndarray, tol: float = 0.28) -> bool:
        def dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        sides = [
            dist(corners[0], corners[1]),
            dist(corners[1], corners[2]),
            dist(corners[2], corners[3]),
            dist(corners[3], corners[0]),
        ]
        mean = sum(sides) / 4.0
        if mean < 1e-3:
            return False
        if any(abs(s - mean) / mean > tol for s in sides):
            return False
        # Aspect via diagonals.
        d1 = dist(corners[0], corners[2])
        d2 = dist(corners[1], corners[3])
        if min(d1, d2) < 1e-3:
            return False
        return abs(d1 - d2) / max(d1, d2) < tol + 0.1

    @staticmethod
    def _warp(image_bgr: np.ndarray, corners: np.ndarray, size: int) -> np.ndarray:
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(image_bgr, matrix, (size, size))

    @staticmethod
    def _checkerboard_score(warped_bgr: np.ndarray) -> float:
        """
        Score how much an image looks like an 8x8 chessboard.

        Uses per-square mean luminance alternating correlation.
        """
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        size = gray.shape[0]
        sq = size // 8
        if sq < 4:
            return 0.0

        means = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                tile = gray[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq]
                # Center crop avoids coordinate labels / piece edges slightly.
                m = tile[sq // 4 : 3 * sq // 4, sq // 4 : 3 * sq // 4]
                means[r, c] = float(np.mean(m))

        # Expected pattern: (r+c) even is light OR dark; pick the better polarity.
        pattern = np.array([[(r + c) % 2 for c in range(8)] for r in range(8)], dtype=np.float32)
        flat = means.reshape(-1)
        # Normalize
        flat = (flat - flat.mean()) / (flat.std() + 1e-6)
        p = (pattern.reshape(-1) * 2 - 1)
        corr_a = float(np.mean(flat * p))
        corr_b = float(np.mean(flat * -p))
        return max(corr_a, corr_b, 0.0)