"""CLI entrypoint: `python -m chess_boss`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from chess_boss import __version__
from chess_boss.app import ChessBossApp
from chess_boss.capture.screen import ScreenCapture
from chess_boss.config import AppConfig, CaptureConfig
from chess_boss.detection.templates import ensure_templates_from_example
from chess_boss.logging.setup import setup_logging
from chess_boss.voice.tts import list_voices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chess-boss",
        description="Screen-aware chess coach with Stockfish guidance and UCI/SAN logging.",
    )
    p.add_argument("--version", action="version", version=f"chess-boss {__version__}")
    p.add_argument(
        "-S",
        "-m",
        "--screen",
        "--monitor",
        dest="screen",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Capture screen index at startup (0=all displays, 1=primary, 2=…). "
            "Default: 1. Change live in the HUD."
        ),
    )
    p.add_argument(
        "--list-screens",
        action="store_true",
        help="List available capture screens (index, size, origin) and exit",
    )
    p.add_argument(
        "--voice-only",
        action="store_true",
        help="Headless mode: no preview window; announce moves by voice only",
    )
    p.add_argument(
        "--voice",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Voice actor / TTS voice (e.g. Samantha, Daniel, Albert). "
            "Enables speaking even with the window. See --list-voices"
        ),
    )
    p.add_argument(
        "--list-voices",
        action="store_true",
        help="List available TTS voices and exit",
    )
    p.add_argument(
        "--voice-rate",
        type=int,
        default=155,
        metavar="WPM",
        help="Speech rate in words per minute (default 155; lower = slower)",
    )
    p.add_argument(
        "--voice-repeat",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Repeat the suggested move every SEC seconds if you have not moved (default 5; 0=off)",
    )
    p.add_argument(
        "--speak",
        action="store_true",
        help="Enable voice announcements while keeping the preview window",
    )
    p.add_argument("--fps", type=float, default=8.0, help="Target capture FPS (default 8)")
    p.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to stockfish binary (or set STOCKFISH_PATH)",
    )
    p.add_argument("--depth", type=int, default=16, help="Stockfish depth (default 16)")
    p.add_argument(
        "--movetime",
        type=int,
        default=None,
        help="Stockfish movetime in ms (overrides --depth when set)",
    )
    p.add_argument(
        "--side",
        choices=("auto", "white", "black"),
        default="auto",
        help="Your side (default: auto-detect from board orientation)",
    )
    p.add_argument(
        "--always-guide",
        action="store_true",
        help="Show/speak best move even when it is not your turn",
    )
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Use a still image instead of live screen (great with examples/)",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Extract piece templates from examples/ and exit",
    )
    p.add_argument(
        "--example",
        type=str,
        default=None,
        help="Example image for --calibrate (default: first file in examples/)",
    )
    p.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Optional capture crop region",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, …")
    p.add_argument("--scale", type=float, default=0.75, help="Preview window scale")
    return p


def _print_screens() -> int:
    capture = ScreenCapture(CaptureConfig(monitor_index=0))
    try:
        monitors = capture.describe_monitors()
    finally:
        capture.close()

    print("Available screens (use with --screen N):")
    for mon in monitors:
        kind = "all displays" if mon.index == 0 else "display"
        print(
            f"  {mon.index}: {mon.width}x{mon.height} at ({mon.left},{mon.top})  [{kind}]"
        )
    print()
    print("Examples:")
    print("  python -m chess_boss --screen 2 --depth 18")
    print("  python -m chess_boss -S 1 --side auto")
    return 0


def _print_voices(filter_lang: Optional[str] = "en") -> int:
    voices = list_voices()
    if not voices:
        print("No voices found.")
        return 1
    print("Available voices (use with --voice NAME):")
    shown = 0
    for v in voices:
        if filter_lang and filter_lang not in v.lang.lower() and filter_lang not in v.name.lower():
            continue
        print(f"  {v.name:<40} {v.lang:<8} {v.sample}")
        shown += 1
    if shown == 0:
        # Fallback: show all
        for v in voices:
            print(f"  {v.name:<40} {v.lang:<8} {v.sample}")
    print()
    print("Examples:")
    print('  python -m chess_boss --voice-only --voice Samantha --screen 2')
    print('  python -m chess_boss --speak --voice Daniel --side white')
    print("Tip: run with --list-voices and omit the English filter by reading full `say -v ?`")
    return 0


def _validate_screen(index: int) -> None:
    capture = ScreenCapture(CaptureConfig(monitor_index=0))
    try:
        monitors = capture.describe_monitors()
    finally:
        capture.close()
    if index < 0 or index >= len(monitors):
        labels = ", ".join(m.label for m in monitors)
        raise SystemExit(
            f"Invalid --screen {index}. Available: {labels}\n"
            f"Run: python -m chess_boss --list-screens"
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_screens:
        return _print_screens()
    if args.list_voices:
        return _print_voices()

    _validate_screen(args.screen)

    config = AppConfig()
    config.capture.monitor_index = args.screen
    config.capture.target_fps = args.fps
    if args.region:
        config.capture.region = tuple(args.region)  # type: ignore[assignment]
    config.engine.path = args.stockfish
    config.engine.depth = args.depth
    config.engine.movetime_ms = args.movetime
    config.guide_when_our_turn = not args.always_guide
    config.our_color_override = None if args.side == "auto" else args.side
    config.preview_scale = args.scale
    config.logging.level = args.log_level
    config.voice.voice_only = args.voice_only
    config.voice.voice = args.voice
    config.voice.rate = args.voice_rate
    config.voice.repeat_seconds = max(0.0, float(args.voice_repeat))
    # Voice on when: voice-only, explicit --speak, or a voice actor was chosen.
    config.voice.enabled = bool(args.voice_only or args.speak or args.voice)

    config.logging.session_id = setup_logging(config.logging)

    if args.calibrate:
        example = Path(args.example) if args.example else None
        bank = ensure_templates_from_example(config.detection, example_path=example)
        print(f"Templates ready in {bank.templates_dir}")
        return 0

    app = ChessBossApp(config)
    if args.image:
        app.use_image(args.image)
    elif args.example:
        app.use_image(args.example)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
