"""Cross-platform-ish TTS. Prefers macOS `say`, falls back to pyttsx3 if present."""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceInfo:
    name: str
    lang: str
    sample: str = ""


class VoiceCoach(ABC):
    """Speaks coaching lines without blocking the capture loop."""

    @abstractmethod
    def speak(self, text: str, interrupt: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def voice_name(self) -> str:
        raise NotImplementedError


class NullVoiceCoach(VoiceCoach):
    def speak(self, text: str, interrupt: bool = True) -> None:
        logger.debug("TTS(disabled): %s", text)

    def close(self) -> None:
        return

    @property
    def voice_name(self) -> str:
        return "off"


class MacSayVoiceCoach(VoiceCoach):
    """Apple `say` — non-blocking, interruptible."""

    def __init__(self, voice: Optional[str] = None, rate: int = 155) -> None:
        if not shutil.which("say"):
            raise RuntimeError("`say` not found (macOS only)")
        self._voice = voice
        self._rate = rate
        self._proc: Optional[subprocess.Popen] = None
        if voice:
            self._validate_voice(voice)
        logger.info(
            "Voice coach ready via macOS say (voice=%s, rate=%s)",
            voice or "system-default",
            rate,
        )

    @property
    def voice_name(self) -> str:
        return self._voice or "system-default"

    def _validate_voice(self, voice: str) -> None:
        names = {v.name.lower(): v.name for v in list_macos_voices()}
        key = voice.lower()
        if key not in names:
            # Allow partial match: "Eddy (English (US))" or just "Samantha"
            matches = [n for n in names if key in n or n.startswith(key)]
            if len(matches) == 1:
                self._voice = names[matches[0]]
                logger.info("Resolved voice %r → %s", voice, self._voice)
                return
            if matches:
                preview = ", ".join(names[m] for m in matches[:8])
                raise ValueError(
                    f"Ambiguous voice {voice!r}. Matches: {preview}\n"
                    f"Run: python -m chess_boss --list-voices"
                )
            raise ValueError(
                f"Unknown voice {voice!r}. Run: python -m chess_boss --list-voices"
            )
        self._voice = names[key]

    def speak(self, text: str, interrupt: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if interrupt:
            self._stop()
        cmd = ["say", "-r", str(self._rate)]
        if self._voice:
            cmd.extend(["-v", self._voice])
        cmd.append(text)
        logger.info("SPEAK [%s]: %s", self.voice_name, text)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error("Failed to speak: %s", exc)

    def _stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def close(self) -> None:
        self._stop()


def list_macos_voices() -> List[VoiceInfo]:
    if not shutil.which("say"):
        return []
    try:
        raw = subprocess.check_output(["say", "-v", "?"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return []
    voices: List[VoiceInfo] = []
    # "Samantha             en_US    # Hello! My name is Samantha."
    pattern = re.compile(r"^(.+?)\s{2,}([a-z]{2}_[A-Z]{2})\s+#\s*(.*)$")
    for line in raw.splitlines():
        m = pattern.match(line.rstrip())
        if not m:
            continue
        voices.append(VoiceInfo(name=m.group(1).strip(), lang=m.group(2), sample=m.group(3)))
    return voices


def list_voices() -> List[VoiceInfo]:
    if platform.system() == "Darwin":
        return list_macos_voices()
    # Optional pyttsx3 discovery
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        out = []
        for v in engine.getProperty("voices") or []:
            out.append(VoiceInfo(name=getattr(v, "name", str(v.id)), lang=getattr(v, "id", "")))
        engine.stop()
        return out
    except Exception:  # noqa: BLE001
        return []


def create_voice_coach(
    enabled: bool,
    voice: Optional[str] = None,
    rate: int = 155,
) -> VoiceCoach:
    if not enabled:
        return NullVoiceCoach()
    system = platform.system()
    if system == "Darwin":
        return MacSayVoiceCoach(voice=voice, rate=rate)
    try:
        import pyttsx3  # type: ignore

        class PyttsxCoach(VoiceCoach):
            def __init__(self) -> None:
                self._engine = pyttsx3.init()
                self._name = voice or "pyttsx3-default"
                if voice:
                    for v in self._engine.getProperty("voices") or []:
                        if voice.lower() in (getattr(v, "name", "") or "").lower():
                            self._engine.setProperty("voice", v.id)
                            self._name = v.name
                            break
                self._engine.setProperty("rate", rate)

            @property
            def voice_name(self) -> str:
                return self._name

            def speak(self, text: str, interrupt: bool = True) -> None:
                logger.info("SPEAK [%s]: %s", self._name, text)
                if interrupt:
                    self._engine.stop()
                self._engine.say(text)
                self._engine.runAndWait()

            def close(self) -> None:
                try:
                    self._engine.stop()
                except Exception:  # noqa: BLE001
                    pass

        return PyttsxCoach()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No TTS backend available. On macOS use the built-in `say` command; "
            "elsewhere install pyttsx3 (`pip install pyttsx3`)."
        ) from exc