"""Compose the Chess Boss preview window: HUD strip + annotated board."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import chess
import cv2
import numpy as np

from chess_boss.analysis.advisor import GuideHint
from chess_boss.capture.screen import MonitorInfo
from chess_boss.config import OverlayConfig
from chess_boss.detection.board_detector import DetectedBoard
from chess_boss.detection.side_detector import PlayerSide

logger = logging.getLogger(__name__)


@dataclass
class HudStats:
    """Live telemetry shown in the top status strip (gaming-style FPS bar)."""

    fps: float = 0.0
    board_detected: bool = False
    board_score: float = 0.0
    side: PlayerSide = PlayerSide.UNKNOWN
    our_turn: bool = False
    fen: str = "-"
    best_move: str = "-"
    eval_text: str = "-"
    engine_status: str = "off"
    move_chain_san: str = ""
    confidence: float = 0.0
    status: str = "boot"
    monitor_index: int = 1
    monitors: List[MonitorInfo] = field(default_factory=list)


class OverlayRenderer:
    """Draw HUD + move hints onto frames and display via OpenCV."""

    # Keep HUD chrome readable/clickable; only the video pane is scaled.
    _MIN_HUD_WIDTH = 960

    def __init__(self, config: OverlayConfig) -> None:
        self.config = config
        self._monitor_hitboxes: List[Tuple[int, int, int, int, int]] = []
        self._pending_monitor: Optional[int] = None
        self._last_content_size: Optional[Tuple[int, int]] = None  # (w, h)
        self._force_relayout = True
        self._trackbar_max = -1
        self._trackbar_suppress = False
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(config.window_name, self._on_mouse)

    def close(self) -> None:
        cv2.destroyWindow(self.config.window_name)

    def request_relayout(self) -> None:
        """Force the OS window to resize on the next frame (monitor switch, etc.)."""
        self._force_relayout = True
        self._last_content_size = None

    def consume_monitor_selection(self) -> Optional[int]:
        """Return a monitor index chosen via chip click or trackbar, if any."""
        value = self._pending_monitor
        self._pending_monitor = None
        return value

    def render(
        self,
        frame_bgr: np.ndarray,
        stats: HudStats,
        board: Optional[DetectedBoard] = None,
        hint: Optional[GuideHint] = None,
        white_at_bottom: bool = True,
        preview_scale: float = 0.75,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        if board is not None:
            self._draw_board_outline(canvas, board)
            if hint is not None and hint.from_square is not None and hint.to_square is not None:
                self._draw_move_hint(canvas, board, hint, white_at_bottom)

        # Scale ONLY the video pane so HUD chips stay full-size and clickable.
        video_scale = self._fit_scale(
            canvas.shape[1],
            canvas.shape[0],
            preview_scale,
            hud_height=self.config.hud_height,
        )
        if abs(video_scale - 1.0) > 1e-6:
            video = cv2.resize(
                canvas,
                None,
                fx=video_scale,
                fy=video_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            video = canvas

        composed = self._compose(video, stats)
        self._ensure_monitor_trackbar(stats)
        self._sync_window_size(composed.shape[1], composed.shape[0])
        cv2.imshow(self.config.window_name, composed)
        return composed

    def _compose(self, video: np.ndarray, stats: HudStats) -> np.ndarray:
        strip_h = self.config.hud_height
        vid_h, vid_w = video.shape[:2]
        width = max(vid_w, self._MIN_HUD_WIDTH)
        out = np.zeros((strip_h + vid_h, width, 3), dtype=np.uint8)
        out[:strip_h] = self.config.hud_bg_color
        # Center the video under the HUD when HUD is wider.
        x0 = (width - vid_w) // 2
        out[strip_h : strip_h + vid_h, x0 : x0 + vid_w] = video
        self._paint_hud(out, stats, strip_h, width)
        return out

    def _fit_scale(
        self,
        width: int,
        height: int,
        preview_scale: float,
        hud_height: int,
    ) -> float:
        """Scale video so (scaled video + fixed HUD) fits the max window."""
        scale = max(float(preview_scale), 0.05)
        max_w = float(self.config.max_window_width)
        max_h = float(self.config.max_window_height)
        # Available height for video after reserving fixed HUD chrome.
        max_video_h = max(max_h - hud_height, 120.0)
        disp_w = width * scale
        disp_h = height * scale
        if disp_w > max_w or disp_h > max_video_h:
            scale *= min(max_w / max(disp_w, 1.0), max_video_h / max(disp_h, 1.0))
        return scale

    def _sync_window_size(self, width: int, height: int) -> None:
        """Resize the OpenCV window when capture/content dimensions change."""
        size = (int(width), int(height))
        if not self._force_relayout and self._last_content_size == size:
            return
        try:
            cv2.resizeWindow(self.config.window_name, size[0], size[1])
        except cv2.error as exc:
            logger.debug("resizeWindow failed: %s", exc)
        logger.debug("Preview window adjusted to %dx%d", size[0], size[1])
        self._last_content_size = size
        self._force_relayout = False

    def _ensure_monitor_trackbar(self, stats: HudStats) -> None:
        """Reliable screen picker (works even when mouse mapping is flaky)."""
        count = max(len(stats.monitors), 1)
        maxv = count - 1
        name = self.config.window_name
        if self._trackbar_max != maxv:
            self._trackbar_suppress = True
            cv2.createTrackbar(
                "screen",
                name,
                int(np.clip(stats.monitor_index, 0, maxv)),
                maxv,
                self._on_trackbar,
            )
            self._trackbar_max = maxv
            self._trackbar_suppress = False
            return

        # Keep the slider aligned when selection changes via chips / keys.
        try:
            pos = cv2.getTrackbarPos("screen", name)
        except cv2.error:
            return
        target = int(np.clip(stats.monitor_index, 0, maxv))
        if pos != target:
            self._trackbar_suppress = True
            cv2.setTrackbarPos("screen", name, target)
            self._trackbar_suppress = False

    def poll_key(self, delay_ms: int = 1) -> int:
        return cv2.waitKey(delay_ms) & 0xFF

    def _on_trackbar(self, pos: int) -> None:
        if self._trackbar_suppress:
            return
        self._pending_monitor = int(pos)
        logger.debug("Trackbar monitor → %s", pos)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ix, iy = self._map_mouse_to_image(x, y)
        for x1, y1, x2, y2, idx in self._monitor_hitboxes:
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                self._pending_monitor = idx
                logger.info("HUD monitor click → %s", idx)
                return

    def _map_mouse_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """
        Map window mouse coords → composed-image pixels.

        Needed when the OS window size diverges from the numpy image
        (Retina scaling, user drag-resize, letterboxing).
        """
        if not self._last_content_size:
            return x, y
        content_w, content_h = self._last_content_size
        try:
            _wx, _wy, win_w, win_h = cv2.getWindowImageRect(self.config.window_name)
        except Exception:  # noqa: BLE001
            return x, y
        if win_w <= 0 or win_h <= 0:
            return x, y
        if win_w == content_w and win_h == content_h:
            return x, y
        return int(x * content_w / win_w), int(y * content_h / win_h)

    def _paint_hud(
        self,
        canvas: np.ndarray,
        stats: HudStats,
        strip_h: int,
        width: int,
    ) -> None:
        def put(text: str, x: int, y: int, color=None, scale=0.55, thick=1) -> None:
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color or self.config.hud_text_color,
                thick,
                cv2.LINE_AA,
            )

        accent = self.config.hud_accent_color
        board_col = accent if stats.board_detected else (80, 80, 220)
        turn_col = accent if stats.our_turn else (180, 180, 80)

        put(f"FPS {stats.fps:5.1f}", 12, 24, accent, 0.6, 2)
        put(f"BOARD {'YES' if stats.board_detected else 'NO'}", 130, 24, board_col, 0.5, 2)
        put(f"SCR {stats.board_score:.2f}", 260, 24, scale=0.5)
        put(f"SIDE {stats.side.value.upper()}", 350, 24, turn_col, 0.5, 2)
        put(f"TURN {'US' if stats.our_turn else 'THEM'}", 490, 24, turn_col, 0.5, 2)
        put(f"ENG {stats.engine_status}", 610, 24, scale=0.5)
        put(f"CONF {stats.confidence:.2f}", 740, 24, scale=0.5)
        put(stats.status[:28], max(12, width - 260), 24, (200, 200, 120), 0.5)

        put(f"BEST {stats.best_move}", 12, 48, self.config.hint_to_color, 0.5, 2)
        put(f"EVAL {stats.eval_text}", 190, 48, scale=0.5)
        fen_short = stats.fen if len(stats.fen) < 48 else stats.fen[:45] + "..."
        put(f"FEN {fen_short}", 320, 48, scale=0.42)
        if stats.move_chain_san:
            chain = stats.move_chain_san
            if len(chain) > 70:
                chain = "…" + chain[-69:]
            put(chain, 12, 66, (160, 160, 160), 0.4)

        self._draw_monitor_selector(canvas, stats.monitors, stats.monitor_index, strip_h)

    def _draw_monitor_selector(
        self,
        canvas: np.ndarray,
        monitors: Sequence[MonitorInfo],
        active_index: int,
        strip_h: int,
    ) -> None:
        """Paint large monitor chips; hitboxes are final display pixels."""
        self._monitor_hitboxes = []
        y1 = strip_h - 34
        y2 = strip_h - 6
        x = 12
        put_scale = 0.48

        cv2.putText(
            canvas,
            "SCREEN",
            (x, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            put_scale,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        x = 90

        if not monitors:
            return

        for mon in monitors:
            label = mon.label
            (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, put_scale, 1)
            pad_x = 12
            x1, x2 = x, x + tw + pad_x * 2
            active = mon.index == active_index
            bg = self.config.hud_accent_color if active else (55, 55, 55)
            fg = (20, 20, 20) if active else self.config.hud_text_color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, thickness=-1, lineType=cv2.LINE_AA)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 90, 90), thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (x1 + pad_x, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                put_scale,
                fg,
                1,
                cv2.LINE_AA,
            )
            self._monitor_hitboxes.append((x1, y1, x2, y2, mon.index))
            x = x2 + 10

        cv2.putText(
            canvas,
            "click  ·  drag 'screen' slider  ·  [ ]  ·  0-9",
            (x + 6, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

    def _draw_board_outline(self, frame: np.ndarray, board: DetectedBoard) -> None:
        pts = board.corners.astype(np.int32)
        cv2.polylines(frame, [pts], True, (80, 200, 120), 2, cv2.LINE_AA)

    def _draw_move_hint(
        self,
        frame: np.ndarray,
        board: DetectedBoard,
        hint: GuideHint,
        white_at_bottom: bool,
    ) -> None:
        assert hint.from_square is not None and hint.to_square is not None
        p0 = self._square_center(board, hint.from_square, white_at_bottom)
        p1 = self._square_center(board, hint.to_square, white_at_bottom)
        self._fill_square(frame, board, hint.from_square, white_at_bottom, self.config.hint_from_color, 0.35)
        self._fill_square(frame, board, hint.to_square, white_at_bottom, self.config.hint_to_color, 0.35)
        cv2.arrowedLine(frame, p0, p1, self.config.hint_arrow_color, 3, cv2.LINE_AA, tipLength=0.18)
        if hint.label:
            cv2.putText(
                frame,
                hint.label,
                (p1[0] + 8, p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.config.hint_arrow_color,
                2,
                cv2.LINE_AA,
            )

    def _square_to_grid(
        self, square: chess.Square, white_at_bottom: bool
    ) -> Tuple[int, int]:
        file_idx = chess.square_file(square)  # 0=a
        rank_idx = chess.square_rank(square)  # 0=1
        if white_at_bottom:
            row = 7 - rank_idx
            col = file_idx
        else:
            row = rank_idx
            col = 7 - file_idx
        return row, col

    def _square_center(
        self,
        board: DetectedBoard,
        square: chess.Square,
        white_at_bottom: bool,
    ) -> Tuple[int, int]:
        row, col = self._square_to_grid(square, white_at_bottom)
        u = (col + 0.5) / 8.0
        v = (row + 0.5) / 8.0
        tl, tr, br, bl = board.corners
        top = tl + (tr - tl) * u
        bottom = bl + (br - bl) * u
        pt = top + (bottom - top) * v
        return int(pt[0]), int(pt[1])

    def _fill_square(
        self,
        frame: np.ndarray,
        board: DetectedBoard,
        square: chess.Square,
        white_at_bottom: bool,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        row, col = self._square_to_grid(square, white_at_bottom)
        corners = []
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            u = (col + du) / 8.0
            v = (row + dv) / 8.0
            tl, tr, br, bl = board.corners
            top = tl + (tr - tl) * u
            bottom = bl + (br - bl) * u
            pt = top + (bottom - top) * v
            corners.append(pt)
        pts = np.array(corners, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)