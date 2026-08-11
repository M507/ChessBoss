"""Cross-platform screen capture via mss."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mss
import numpy as np

from chess_boss.config import CaptureConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One captured frame with timing metadata."""

    image_bgr: np.ndarray
    timestamp: float
    monitor: dict


@dataclass(frozen=True)
class MonitorInfo:
    """Human-readable monitor descriptor for the HUD selector."""

    index: int
    label: str
    width: int
    height: int
    left: int
    top: int


class ScreenCapture:
    """Captures the desktop (or a crop) as BGR numpy arrays."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._sct = mss.mss()
        self._last_capture = 0.0
        self._min_interval = 1.0 / max(config.target_fps, 0.1)
        self._ensure_valid_monitor()
        logger.info(
            "ScreenCapture ready (monitor=%s, target_fps=%.1f, monitors=%s)",
            config.monitor_index,
            config.target_fps,
            len(self._sct.monitors),
        )

    def close(self) -> None:
        self._sct.close()

    @property
    def monitor_index(self) -> int:
        return self.config.monitor_index

    @property
    def monitor_count(self) -> int:
        return len(self._sct.monitors)

    def list_monitors(self) -> Tuple[dict, ...]:
        return tuple(self._sct.monitors)

    def describe_monitors(self) -> List[MonitorInfo]:
        self._refresh_monitors()
        infos: List[MonitorInfo] = []
        for idx, mon in enumerate(self._sct.monitors):
            if idx == 0:
                label = f"{idx}:ALL"
            else:
                label = f"{idx}:{mon['width']}x{mon['height']}"
            infos.append(
                MonitorInfo(
                    index=idx,
                    label=label,
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                )
            )
        return infos

    def _refresh_monitors(self) -> None:
        """Re-enumerate displays (hot-plug / resolution changes) and clamp index."""
        # Accessing .monitors forces mss to re-query the OS layout.
        count = len(self._sct.monitors)
        if count == 0:
            return
        if self.config.monitor_index >= count:
            fallback = 1 if count > 1 else 0
            logger.warning(
                "Monitor index %s gone (now %s displays); switching to %s",
                self.config.monitor_index,
                count,
                fallback,
            )
            self.config.monitor_index = fallback
            self.config.region = None

    def set_monitor(self, index: int) -> MonitorInfo:
        """Switch capture source to a monitor index (0 = virtual all-screens)."""
        monitors = self._sct.monitors
        if index < 0 or index >= len(monitors):
            raise IndexError(
                f"monitor_index {index} out of range (have 0..{len(monitors) - 1})"
            )
        self.config.monitor_index = index
        # Explicit monitor choice overrides any prior crop region.
        self.config.region = None
        info = self.describe_monitors()[index]
        logger.info(
            "Switched capture monitor → %s (%dx%d @ %d,%d)",
            info.label,
            info.width,
            info.height,
            info.left,
            info.top,
        )
        return info

    def cycle_monitor(self, delta: int = 1) -> MonitorInfo:
        count = self.monitor_count
        if count <= 0:
            raise RuntimeError("No monitors available")
        next_idx = (self.monitor_index + delta) % count
        return self.set_monitor(next_idx)

    def _ensure_valid_monitor(self) -> None:
        count = len(self._sct.monitors)
        if count == 0:
            raise RuntimeError("mss reported zero monitors")
        if self.config.monitor_index < 0 or self.config.monitor_index >= count:
            logger.warning(
                "monitor_index %s invalid (have %s); falling back to 1 or 0",
                self.config.monitor_index,
                count,
            )
            self.config.monitor_index = 1 if count > 1 else 0

    def _monitor_dict(self) -> dict:
        if self.config.region is not None:
            left, top, width, height = self.config.region
            return {"left": left, "top": top, "width": width, "height": height}
        return self._sct.monitors[self.config.monitor_index]

    def grab(self) -> Frame:
        """Grab a frame, optionally throttling to target FPS."""
        now = time.perf_counter()
        wait = self._min_interval - (now - self._last_capture)
        if wait > 0:
            time.sleep(wait)

        self._refresh_monitors()
        monitor = self._monitor_dict()
        shot = self._sct.grab(monitor)
        # mss returns BGRA
        bgra = np.asarray(shot, dtype=np.uint8)
        bgr = bgra[:, :, :3].copy()
        self._last_capture = time.perf_counter()
        return Frame(image_bgr=bgr, timestamp=time.time(), monitor=monitor)