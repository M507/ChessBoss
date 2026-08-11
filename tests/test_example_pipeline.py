"""Regression tests against Lichess example screenshots."""

from pathlib import Path

import chess
import cv2
import pytest

from chess_boss.config import DetectionConfig, EngineConfig
from chess_boss.detection import (
    BoardDetector,
    PieceClassifier,
    SideDetector,
    ensure_templates_from_example,
)
from chess_boss.engine import StockfishEngine
from chess_boss.analysis.advisor import MoveAdvisor
from chess_boss.detection.side_detector import PlayerSide

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
STARTPOS = EXAMPLES / "2026-08-11_13-42.png"
CARO_KANN = EXAMPLES / "2026-08-11_15-22.png"
BLACK_SIDE = EXAMPLES / "2026-08-11_16-06_black.png"


@pytest.fixture(scope="module")
def vision():
    cfg = DetectionConfig()
    bank = ensure_templates_from_example(cfg, example_path=STARTPOS)
    return cfg, BoardDetector(cfg), PieceClassifier(cfg, bank), SideDetector()


def test_example_board_to_startpos_fen(vision):
    cfg, detector, classifier, side_det = vision
    assert STARTPOS.exists()
    image = cv2.imread(str(STARTPOS))
    detected = detector.detect(image)
    assert detected is not None
    assert detected.score > 0.2

    occupation = classifier.classify(detected.warped, white_at_bottom=True)
    assert occupation.fen_placement == chess.Board().board_fen()
    assert occupation.confidence > 0.8

    side = side_det.detect(detected.warped, occupation)
    assert side.side.value == "white"
    assert side.white_at_bottom is True


def test_newest_screenshot_caro_kann_position(vision):
    """Green last-move highlights must not invent pawns; h1 stays a rook."""
    _cfg, detector, classifier, side_det = vision
    assert CARO_KANN.exists()
    image = cv2.imread(str(CARO_KANN))
    detected = detector.detect(image)
    assert detected is not None

    occupation = classifier.classify(detected.warped, white_at_bottom=True)
    expected = "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR"
    assert occupation.fen_placement == expected

    board = classifier.to_board(occupation, turn=chess.WHITE)
    assert board.is_valid()

    side = side_det.detect(detected.warped, occupation)
    assert side.side.value == "white"


def test_black_at_bottom_orientation(vision):
    """Flipped board (Black to move after 1.e4 e5 2.Nc3) must classify correctly."""
    _cfg, detector, classifier, side_det = vision
    assert BLACK_SIDE.exists()
    image = cv2.imread(str(BLACK_SIDE))
    detected = detector.detect(image)
    assert detected is not None

    provisional = classifier.classify(detected.warped, white_at_bottom=True)
    side = side_det.detect(detected.warped, provisional)
    assert side.side.value == "black"
    assert side.white_at_bottom is False

    occupation = classifier.classify(
        detected.warped, white_at_bottom=side.white_at_bottom
    )
    expected = "rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR"
    assert occupation.fen_placement == expected

    board = classifier.to_board(occupation, turn=chess.BLACK)
    assert board.is_valid()
    assert board.turn == chess.BLACK


def test_all_examples_detect_without_crash(vision):
    _cfg, detector, classifier, _side = vision
    paths = sorted(EXAMPLES.glob("*.png"))
    assert paths, "no example screenshots"
    for path in paths:
        image = cv2.imread(str(path))
        assert image is not None, path.name
        detected = detector.detect(image)
        assert detected is not None, f"no board in {path.name}"
        occupation = classifier.classify(detected.warped, white_at_bottom=True)
        board = classifier.to_board(occupation, turn=chess.WHITE)
        # Illegal vision must be reported but must not raise.
        _ = board.is_valid()


def test_engine_survives_illegal_fen_and_advises_legal():
    eng = StockfishEngine(EngineConfig(depth=8))
    eng.start()
    try:
        bad = chess.Board(
            "rnbqkbnr/pp1Ppppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNP w - - 0 1"
        )
        assert not bad.is_valid()
        advice = eng.analyse(bad)
        assert advice.best_move is None

        good = chess.Board("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")
        hint = MoveAdvisor(eng, guide_when_our_turn=False).advise(good, PlayerSide.WHITE)
        assert hint.advice is not None
        assert hint.advice.best_move_uci is not None
    finally:
        eng.close()
