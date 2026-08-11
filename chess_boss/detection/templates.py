"""Piece template extraction and storage (seeded from examples/)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from chess_boss.config import DetectionConfig
from chess_boss.detection.board_detector import BoardDetector

logger = logging.getLogger(__name__)

# Starting-position map: board row 0 = rank 8 (top of image when White at bottom).
START_PIECES_WHITE_BOTTOM: List[List[Optional[str]]] = [
    ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
    ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    [None, None, None, None, None, None, None, None],
    ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
    ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
]

PIECE_CODES = [
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
]


class TemplateBank:
    """In-memory + on-disk piece templates for matching."""

    def __init__(self, templates_dir: Path) -> None:
        self.templates_dir = templates_dir
        self.templates: Dict[str, List[np.ndarray]] = {code: [] for code in PIECE_CODES}
        self.empty_light: Optional[np.ndarray] = None
        self.empty_dark: Optional[np.ndarray] = None

    def load(self) -> bool:
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        loaded = 0
        for code in PIECE_CODES:
            paths = sorted(self.templates_dir.glob(f"{code}_*.png"))
            for path in paths:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[code].append(img)
                    loaded += 1
        light = self.templates_dir / "empty_light.png"
        dark = self.templates_dir / "empty_dark.png"
        if light.exists():
            self.empty_light = cv2.imread(str(light), cv2.IMREAD_COLOR)
        if dark.exists():
            self.empty_dark = cv2.imread(str(dark), cv2.IMREAD_COLOR)
        ok = loaded >= 12 and self.empty_light is not None and self.empty_dark is not None
        logger.info("TemplateBank load: %d piece crops, ready=%s", loaded, ok)
        return ok

    def save_from_start_position(self, warped_bgr: np.ndarray) -> None:
        """Extract templates assuming a standard starting position, White at bottom."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        size = warped_bgr.shape[0]
        sq = size // 8
        counts = {code: 0 for code in PIECE_CODES}

        for r in range(8):
            for c in range(8):
                tile = warped_bgr[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq].copy()
                code = START_PIECES_WHITE_BOTTOM[r][c]
                if code is None:
                    # Empty squares: a3/c3/e3/g3 are light or dark depending on a1.
                    # With White at bottom, a1 (r=7,c=0) is dark on Lichess? Actually
                    # standard: a1 is dark. (r+c) even from top-left of image:
                    # r=7,c=0 => odd => dark if top-left (0,0) is light.
                    if (r + c) % 2 == 0:
                        self.empty_light = tile
                        cv2.imwrite(str(self.templates_dir / "empty_light.png"), tile)
                    else:
                        self.empty_dark = tile
                        cv2.imwrite(str(self.templates_dir / "empty_dark.png"), tile)
                else:
                    idx = counts[code]
                    counts[code] = idx + 1
                    path = self.templates_dir / f"{code}_{idx}.png"
                    cv2.imwrite(str(path), tile)
                    self.templates[code].append(tile)

        logger.info("Extracted piece templates into %s", self.templates_dir)


def ensure_templates_from_example(
    config: DetectionConfig,
    example_path: Optional[Path] = None,
) -> TemplateBank:
    """
    Ensure templates exist, building them from the examples/ screenshot if needed.

    The provided Lichess starting-position screenshot is the seed.
    """
    bank = TemplateBank(config.templates_dir)
    if bank.load():
        return bank

    path = example_path
    if path is None:
        examples = sorted(config.examples_dir.glob("*.png")) + sorted(
            config.examples_dir.glob("*.jpg")
        )
        if not examples:
            raise FileNotFoundError(
                f"No example images in {config.examples_dir}; cannot seed templates."
            )
        path = examples[0]

    logger.info("Seeding templates from example: %s", path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read example image: {path}")

    detector = BoardDetector(config)
    board = detector.detect(image)
    if board is None:
        raise RuntimeError(f"Could not detect a chessboard in example image: {path}")

    # Confirm it looks like a starting position with White at bottom before extracting.
    bank.save_from_start_position(board.warped)
    if not bank.load():
        raise RuntimeError("Template extraction failed to produce a usable bank.")
    return bank


def square_image(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    size = warped.shape[0]
    sq = size // 8
    return warped[row * sq : (row + 1) * sq, col * sq : (col + 1) * sq]