"""Computer-vision board and piece detection."""

from .board_detector import BoardDetector, DetectedBoard
from .piece_classifier import PieceClassifier, BoardOccupation
from .side_detector import SideDetector, PlayerSide
from .templates import TemplateBank, ensure_templates_from_example

__all__ = [
    "BoardDetector",
    "DetectedBoard",
    "PieceClassifier",
    "BoardOccupation",
    "SideDetector",
    "PlayerSide",
    "TemplateBank",
    "ensure_templates_from_example",
]