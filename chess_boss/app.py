"""Main Chess Boss application loop."""

from __future__ import annotations

import logging
import time
from typing import Optional

import chess
import cv2

from chess_boss.analysis.advisor import GuideHint, MoveAdvisor
from chess_boss.analysis.game_tracker import GameTracker
from chess_boss.capture.screen import ScreenCapture
from chess_boss.config import AppConfig
from chess_boss.detection.board_detector import BoardDetector, DetectedBoard
from chess_boss.detection.piece_classifier import PieceClassifier
from chess_boss.detection.side_detector import PlayerSide, SideDetector
from chess_boss.detection.templates import ensure_templates_from_example
from chess_boss.engine.stockfish_engine import StockfishEngine
from chess_boss.logging.protocol_logger import ProtocolLogger
from chess_boss.overlay.renderer import HudStats, OverlayRenderer
from chess_boss.voice.announce import spoken_move, spoken_suggestion
from chess_boss.voice.tts import VoiceCoach, create_voice_coach

logger = logging.getLogger(__name__)


class ChessBossApp:
    """Orchestrates capture → detect → track → engine → overlay/voice."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.capture = ScreenCapture(config.capture)
        self.detector = BoardDetector(config.detection)
        self.side_detector = SideDetector()
        self.protocol = ProtocolLogger(
            config.logging, session_id=config.logging.session_id
        )
        self.tracker = GameTracker(protocol=self.protocol)
        self.voice_only = config.voice.voice_only
        self.overlay: Optional[OverlayRenderer] = (
            None if self.voice_only else OverlayRenderer(config.overlay)
        )
        self.engine = StockfishEngine(config.engine)
        self.advisor = MoveAdvisor(self.engine, config.guide_when_our_turn)
        self.voice: VoiceCoach = create_voice_coach(
            enabled=config.voice.enabled or config.voice.voice_only,
            voice=config.voice.voice,
            rate=config.voice.rate,
        )

        bank = ensure_templates_from_example(config.detection)
        self.classifier = PieceClassifier(config.detection, bank)

        self._fps = 0.0
        self._frames = 0
        self._fps_t0 = time.perf_counter()
        self._our_side = PlayerSide.UNKNOWN
        self._white_at_bottom = True
        self._static_image: Optional[str] = None
        self._last_frame_size: Optional[tuple] = None
        self._last_monitor_count: int = self.capture.monitor_count
        self._last_spoken_suggestion: Optional[str] = None
        self._suggestion_spoken_at: float = 0.0
        self._last_spoken_played: Optional[str] = None
        self._consecutive_errors = 0
        self._last_logged_fen: Optional[str] = None
        self._last_logged_engine: Optional[str] = None
        self._frames_seen = 0
        self._boards_seen = 0

    def use_image(self, path: str) -> None:
        """Debug / calibrate against a still image instead of the live screen."""
        self._static_image = path
        logger.info("Using static image source: %s", path)

    def run(self) -> int:
        engine_status = "off"
        if self.engine.available:
            try:
                self.engine.start()
                engine_status = "ready"
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to start Stockfish: %s", exc)
                engine_status = "error"
        else:
            logger.warning(
                "Stockfish not found — vision/logging still work; install via "
                "`brew install stockfish` or pass --stockfish"
            )
            engine_status = "missing"

        if self.config.our_color_override in ("white", "black"):
            self._our_side = PlayerSide(self.config.our_color_override)
            self._white_at_bottom = self._our_side == PlayerSide.WHITE
            logger.info("Side override: %s", self._our_side.value)

        try:
            monitors = self.capture.describe_monitors()
        except Exception as exc:  # noqa: BLE001
            logger.error("Monitor enumeration failed: %s", exc)
            monitors = []

        self.protocol.event(
            "runtime",
            {
                "voice_only": self.voice_only,
                "voice": self.voice.voice_name,
                "screen": self.capture.monitor_index,
                "depth": self.config.engine.depth,
                "side_override": self.config.our_color_override,
                "guide_when_our_turn": self.config.guide_when_our_turn,
            },
        )
        self.protocol.chain(
            "SESSION",
            f"start screen={self.capture.monitor_index} voice={self.voice.voice_name} "
            f"depth={self.config.engine.depth}",
        )

        if self.voice_only:
            logger.info(
                "Voice-only mode (no window) — screen=%s voice=%s | Ctrl+C to quit",
                self.capture.monitor_index,
                self.voice.voice_name,
            )
            try:
                self.voice.speak("Chess Boss voice mode ready", interrupt=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Startup speak failed: %s", exc)
        else:
            logger.info(
                "Chess Boss running — monitors: %s | click SCREEN chips or [ ] / 0-9",
                ", ".join(m.label for m in monitors),
            )
            logger.info("Keys: q quit · r reset · s re-detect side · [ ] cycle screen")
            if self.config.voice.enabled:
                logger.info("Voice enabled: %s", self.voice.voice_name)

        try:
            while True:
                try:
                    frame_bgr, source_label = self._next_frame()
                except Exception as exc:  # noqa: BLE001
                    self._consecutive_errors += 1
                    logger.exception("Frame capture failed (%s): %s", self._consecutive_errors, exc)
                    time.sleep(0.25)
                    if self._consecutive_errors >= 20:
                        logger.error("Too many capture failures — exiting loop")
                        break
                    continue

                try:
                    stats = self._process_frame(frame_bgr, engine_status, source_label)
                    self._consecutive_errors = 0
                    if not self.engine.is_running and engine_status == "ready":
                        engine_status = "restarting"
                    elif self.engine.is_running and engine_status in ("restarting", "error"):
                        engine_status = "ready"
                except Exception as exc:  # noqa: BLE001
                    self._consecutive_errors += 1
                    logger.exception(
                        "Frame processing failed (%s) — continuing: %s",
                        self._consecutive_errors,
                        exc,
                    )
                    try:
                        self._render_safe(
                            frame_bgr,
                            HudStats(
                                fps=self._fps,
                                engine_status=engine_status,
                                side=self._our_side,
                                status=f"error:{type(exc).__name__}",
                                monitor_index=self.capture.monitor_index,
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    # Recover engine if it died mid-frame.
                    if not self.engine.is_running and self.engine.available:
                        try:
                            if self.engine.restart():
                                engine_status = "ready"
                        except Exception as rexc:  # noqa: BLE001
                            logger.error("Engine recovery failed: %s", rexc)
                            engine_status = "error"
                    time.sleep(0.05)
                    continue

                try:
                    if self.voice_only:
                        time.sleep(0.01)
                    else:
                        assert self.overlay is not None
                        key = self.overlay.poll_key(1)
                        if self._handle_input(key):
                            break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Input handling failed: %s", exc)

                self._update_fps()
                stats.fps = self._fps
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        for name, closer in (
            ("voice", self.voice.close),
            ("engine", self.engine.close),
            ("protocol", self.protocol.close),
            ("capture", self.capture.close),
        ):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shutdown %s failed: %s", name, exc)
        if self.overlay is not None:
            try:
                self.overlay.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shutdown overlay failed: %s", exc)
        logger.info("Shutdown complete")

    def _handle_input(self, key: int) -> bool:
        """Handle keyboard/mouse. Returns True when the app should quit."""
        if self.overlay is None:
            return False

        try:
            clicked = self.overlay.consume_monitor_selection()
            if clicked is not None and self._static_image is None:
                self._switch_monitor(clicked)
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monitor click handling failed: %s", exc)

        if key == ord("q"):
            return True
        if key == ord("r"):
            try:
                self.tracker.reset()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reset failed: %s", exc)
            logger.info("Manual reset")
            return False
        if key == ord("s"):
            if self.config.our_color_override is None:
                self._our_side = PlayerSide.UNKNOWN
                logger.info("Side will re-detect on next board lock")
            return False

        if self._static_image is not None:
            return False

        try:
            if key in (ord("["), ord(",")):
                info = self.capture.cycle_monitor(-1)
                self._on_monitor_changed(info.label)
                return False
            if key in (ord("]"), ord("."), ord("m")):
                info = self.capture.cycle_monitor(1)
                self._on_monitor_changed(info.label)
                return False
            if ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < self.capture.monitor_count:
                    self._switch_monitor(idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monitor switch failed: %s", exc)
        return False

    def _switch_monitor(self, index: int) -> None:
        try:
            info = self.capture.set_monitor(index)
        except IndexError as exc:
            logger.warning("%s", exc)
            return
        self._on_monitor_changed(info.label)

    def _on_monitor_changed(self, label: str) -> None:
        if self.config.our_color_override is None:
            self._our_side = PlayerSide.UNKNOWN
        self._fps = 0.0
        self._frames = 0
        self._fps_t0 = time.perf_counter()
        self._last_frame_size = None
        self._last_monitor_count = self.capture.monitor_count
        self._last_spoken_suggestion = None
        self._suggestion_spoken_at = 0.0
        if self.overlay is not None:
            try:
                self.overlay.request_relayout()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Relayout request failed: %s", exc)
        logger.info("Capture screen set to %s", label)
        try:
            self.protocol.log_screen(label, self.capture.monitor_index)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Screen protocol log failed: %s", exc)
        self._last_logged_fen = None
        self._last_logged_engine = None

    def _next_frame(self):
        if self._static_image:
            img = cv2.imread(self._static_image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(self._static_image)
            time.sleep(1.0 / max(self.config.capture.target_fps, 0.1))
            self._note_frame_geometry(img.shape[1], img.shape[0])
            return img, "image"
        frame = self.capture.grab()
        self._note_frame_geometry(frame.image_bgr.shape[1], frame.image_bgr.shape[0])
        return frame.image_bgr, "screen"

    def _note_frame_geometry(self, width: int, height: int) -> None:
        size = (width, height)
        try:
            monitor_count = self.capture.monitor_count
        except Exception:  # noqa: BLE001
            monitor_count = self._last_monitor_count
        if size != self._last_frame_size or monitor_count != self._last_monitor_count:
            if self._last_frame_size is not None or monitor_count != self._last_monitor_count:
                logger.info(
                    "Capture geometry changed %s → %s (monitors=%s)",
                    self._last_frame_size,
                    size,
                    monitor_count,
                )
            self._last_frame_size = size
            self._last_monitor_count = monitor_count
            if self.overlay is not None:
                self.overlay.request_relayout()

    def _process_frame(self, frame_bgr, engine_status: str, source_label: str) -> HudStats:
        try:
            monitors = [] if self.voice_only else self.capture.describe_monitors()
        except Exception:  # noqa: BLE001
            monitors = []

        stats = HudStats(
            fps=self._fps,
            engine_status=engine_status,
            side=self._our_side,
            status=source_label,
            monitor_index=self.capture.monitor_index,
            monitors=monitors,
        )

        try:
            detected = self.detector.detect(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Board detection error: %s", exc)
            stats.board_detected = False
            stats.status = "detect-error"
            self._render_safe(frame_bgr, stats)
            return stats

        if detected is None:
            stats.board_detected = False
            stats.status = "scanning…"
            self._render_safe(frame_bgr, stats)
            return stats

        stats.board_detected = True
        stats.board_score = detected.score

        try:
            self._update_from_board(detected, stats)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vision/analysis pipeline error: %s", exc)
            stats.status = "pipeline-error"
            self._render_safe(frame_bgr, stats, board=detected)
            return stats

        hint: Optional[GuideHint] = None
        try:
            board = self.tracker.board
            stats.fen = board.fen()
            stats.move_chain_san = self.tracker.chain_san()
            hint = self.advisor.advise(board, self._our_side)
            stats.our_turn = hint.our_turn
            stats.best_move = hint.label
            if hint.advice is not None:
                if hint.advice.mate is not None:
                    stats.eval_text = f"M{hint.advice.mate}"
                elif hint.advice.score_cp is not None:
                    stats.eval_text = f"{hint.advice.score_cp / 100:+.2f}"
                engine_key = (
                    f"{hint.advice.best_move_uci}|{hint.advice.score_cp}|"
                    f"{' '.join(hint.advice.pv_san[:4])}"
                )
                if engine_key != self._last_logged_engine:
                    self._last_logged_engine = engine_key
                    try:
                        self.protocol.log_engine_line(
                            " ".join(hint.advice.pv_uci[:8]),
                            " ".join(hint.advice.pv_san[:8]),
                            hint.advice.score_cp,
                            best_uci=hint.advice.best_move_uci,
                            depth=hint.advice.depth,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Protocol engine log failed: %s", exc)
                stats.status = "guiding" if hint.our_turn else "watching"
                self._maybe_speak_suggestion(hint)
            else:
                stats.status = hint.label
        except Exception as exc:  # noqa: BLE001
            logger.exception("Advice stage failed: %s", exc)
            stats.status = "advice-error"
            hint = None

        self._render_safe(
            frame_bgr,
            stats,
            board=detected,
            hint=hint if hint and hint.from_square is not None else None,
        )
        return stats

    def _update_from_board(self, detected: DetectedBoard, stats: HudStats) -> None:
        self._boards_seen += 1
        provisional = self.classifier.classify(detected.warped, white_at_bottom=True)
        if self._our_side == PlayerSide.UNKNOWN:
            side_result = self.side_detector.detect(detected.warped, provisional)
            self._our_side = side_result.side
            self._white_at_bottom = side_result.white_at_bottom
            stats.status = f"side:{side_result.reason[:24]}"
            try:
                self.protocol.log_side(
                    side_result.side.value,
                    side_result.white_at_bottom,
                    side_result.reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Side protocol log failed: %s", exc)
            if self.config.voice.enabled or self.voice_only:
                try:
                    phrase = f"Playing as {self._our_side.value}"
                    self.voice.speak(phrase, interrupt=False)
                    self.protocol.log_voice(phrase, self.voice.voice_name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Side announce failed: %s", exc)
        else:
            self._white_at_bottom = self._our_side == PlayerSide.WHITE

        occupation = self.classifier.classify(
            detected.warped, white_at_bottom=self._white_at_bottom
        )
        stats.confidence = occupation.confidence
        stats.side = self._our_side

        turn = self._infer_turn(occupation.fen_placement)
        observed = self.classifier.to_board(occupation, turn=turn)

        # Log board lock / FEN changes once (not every frame).
        if occupation.fen_placement != self._last_logged_fen:
            self._last_logged_fen = occupation.fen_placement
            try:
                self.protocol.log_board(
                    observed.fen(),
                    score=detected.score,
                    confidence=occupation.confidence,
                    side=self._our_side.value,
                    bbox=detected.bbox,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Board protocol log failed: %s", exc)
            logger.debug(
                "Vision placement=%s conf=%.3f valid=%s",
                occupation.fen_placement,
                occupation.confidence,
                observed.is_valid(),
            )

        event = self.tracker.update(observed, occupation.confidence)
        if event is not None:
            # A real move was played — stop repeating the old suggestion.
            self._last_spoken_suggestion = None
            self._suggestion_spoken_at = 0.0
            self._last_logged_engine = None
            if self.config.voice.announce_played_moves and (
                self.config.voice.enabled or self.voice_only
            ):
                phrase = f"{spoken_move(event.san)}"
                if phrase != self._last_spoken_played:
                    self._last_spoken_played = phrase
                    try:
                        self.voice.speak(phrase, interrupt=True)
                        self.protocol.log_voice(phrase, self.voice.voice_name)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Move announce failed: %s", exc)

    def _infer_turn(self, fen_placement: str) -> chess.Color:
        """
        Side to move for the observed placement.

        Once the tracker is live, trust its turn clock. On first lock, prefer
        *our* side to move (users usually open the coach on their turn), except
        for the absolute starting position (always White).
        """
        if self.tracker.last_fen_placement is not None:
            return self.tracker.board.turn

        if fen_placement == chess.Board().board_fen():
            return chess.WHITE

        if self._our_side == PlayerSide.BLACK:
            return chess.BLACK
        if self._our_side == PlayerSide.WHITE:
            return chess.WHITE
        return chess.WHITE

    def _maybe_speak_suggestion(self, hint: GuideHint) -> None:
        """Speak the best move; repeat it every N seconds until the position changes."""
        try:
            if not self.config.voice.announce_suggestions:
                return
            if not (self.config.voice.enabled or self.voice_only):
                return
            if not hint.our_turn and self.config.guide_when_our_turn:
                return
            if hint.advice is None:
                return

            san = hint.advice.pv_san[0] if hint.advice.pv_san else None
            uci = hint.advice.best_move_uci
            phrase = spoken_suggestion(san, uci)
            now = time.monotonic()
            repeat_after = float(self.config.voice.repeat_seconds)

            is_new = phrase != self._last_spoken_suggestion
            due_for_repeat = (
                not is_new
                and repeat_after > 0
                and self._suggestion_spoken_at > 0
                and (now - self._suggestion_spoken_at) >= repeat_after
            )
            if not is_new and not due_for_repeat:
                return

            self._last_spoken_suggestion = phrase
            self._suggestion_spoken_at = now
            self.voice.speak(phrase, interrupt=True)
            tag = "repeat" if due_for_repeat else "suggest"
            self.protocol.log_voice(f"({tag}) {phrase}", self.voice.voice_name)
            if due_for_repeat:
                logger.info("VOICE repeat after %.1fs: %s", repeat_after, phrase)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Suggestion speak failed: %s", exc)

    def _render_safe(self, frame_bgr, stats, board=None, hint=None) -> None:
        try:
            self._render(frame_bgr, stats, board=board, hint=hint)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Render failed: %s", exc)

    def _render(self, frame_bgr, stats, board=None, hint=None) -> None:
        if self.overlay is None:
            return
        self.overlay.render(
            frame_bgr,
            stats,
            board=board,
            hint=hint,
            white_at_bottom=self._white_at_bottom,
            preview_scale=self.config.preview_scale,
        )

    def _update_fps(self) -> None:
        self._frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 0.5:
            self._fps = self._frames / elapsed
            self._frames = 0
            self._fps_t0 = now
