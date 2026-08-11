"""Runtime configuration for Chess Boss."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"
ASSETS_DIR = ROOT_DIR / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass
class CaptureConfig:
    """Screen capture settings."""

    monitor_index: int = 1  # mss: 0=all, 1=primary
    region: Optional[Tuple[int, int, int, int]] = None  # left, top, width, height
    target_fps: float = 8.0


@dataclass
class DetectionConfig:
    """Board / piece vision settings."""

    board_size: int = 512
    min_board_area_ratio: float = 0.04
    max_board_area_ratio: float = 0.85
    template_match_threshold: float = 0.50
    empty_diff_threshold: float = 22.0
    examples_dir: Path = field(default_factory=lambda: EXAMPLES_DIR)
    templates_dir: Path = field(default_factory=lambda: TEMPLATES_DIR)


@dataclass
class EngineConfig:
    """Stockfish / UCI engine settings."""

    path: Optional[str] = None  # auto-discover if None
    depth: int = 16
    movetime_ms: Optional[int] = None
    threads: int = 2
    hash_mb: int = 128
    skill_level: Optional[int] = None  # 0-20; None = full strength


@dataclass
class OverlayConfig:
    """HUD / hint rendering settings."""

    window_name: str = "Chess Boss"
    show_fps: bool = True
    show_fen: bool = True
    show_eval: bool = True
    hint_from_color: Tuple[int, int, int] = (40, 220, 40)  # BGR
    hint_to_color: Tuple[int, int, int] = (40, 180, 255)  # BGR
    hint_arrow_color: Tuple[int, int, int] = (0, 215, 255)
    hud_height: int = 100
    hud_bg_color: Tuple[int, int, int] = (24, 24, 24)
    hud_text_color: Tuple[int, int, int] = (230, 230, 230)
    hud_accent_color: Tuple[int, int, int] = (80, 200, 120)
    # Cap preview window so huge/ALL captures still fit on screen.
    max_window_width: int = 1600
    max_window_height: int = 1000


@dataclass
class LoggingConfig:
    """Protocol / application logging."""

    logs_dir: Path = field(default_factory=lambda: LOGS_DIR)
    level: str = "INFO"
    protocol_filename: str = "moves.pgn"
    uci_filename: str = "moves.uci"
    app_log_filename: str = "chess_boss.log"
    session_id: Optional[str] = None


@dataclass
class VoiceConfig:
    """Text-to-speech coaching."""

    enabled: bool = False
    voice_only: bool = False  # no preview window; audio + logs only
    voice: Optional[str] = None  # actor name, e.g. "Samantha", "Daniel"
    rate: int = 155  # words per minute (macOS say -r); slower helps ranks land clearly
    announce_suggestions: bool = True
    announce_played_moves: bool = True
    # Repeat the same suggestion if no move is played within this many seconds.
    # Set to 0 to disable reminders.
    repeat_seconds: float = 5.0


@dataclass
class AppConfig:
    """Top-level application configuration."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    preview_scale: float = 0.75
    calibrate_on_start: bool = True
    guide_when_our_turn: bool = True
    our_color_override: Optional[str] = None  # "white" | "black" | None=auto