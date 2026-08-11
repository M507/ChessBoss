"""Turn SAN / UCI into short phrases suitable for TTS."""

from __future__ import annotations

import re
from typing import Optional

# macOS `say` embedded silence (milliseconds) — keeps file/rank from blending.
_PAUSE_SHORT = "[[slnc 160]]"
_PAUSE_MED = "[[slnc 240]]"

_PIECE = {
    "K": "King",
    "Q": "Queen",
    "R": "Rook",
    "B": "Bishop",
    "N": "Knight",
}

# Digits as words so TTS doesn't drop or rush past "4".
_RANK_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
}

# Spoken letter names (clearer than a bare "E" which many voices swallow).
_FILE_WORDS = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e": "E",
    "f": "F",
    "g": "G",
    "h": "H",
}


def spoken_square(square: str) -> str:
    """e4 → 'E [[pause]] four' so the rank is clearly heard."""
    square = square.strip().lower()
    if len(square) != 2:
        return square
    file_ch, rank_ch = square[0], square[1]
    file_word = _FILE_WORDS.get(file_ch, file_ch.upper())
    rank_word = _RANK_WORDS.get(rank_ch, rank_ch)
    return f"{file_word} {_PAUSE_SHORT} {rank_word}"


def spoken_move(san: str) -> str:
    """
    Convert SAN to a spoken phrase.

    Examples:
      e4      → "E [[pause]] four"
      Nf3     → "Knight to F [[pause]] three"
      Qxd5    → "Queen takes D [[pause]] five"
      O-O     → "castle kingside"
    """
    san = san.strip().rstrip("+#!?")
    if not san:
        return "unknown move"

    if san in ("O-O", "0-0"):
        return "castle kingside"
    if san in ("O-O-O", "0-0-0"):
        return "castle queenside"

    promo = ""
    if "=" in san:
        san, promo_piece = san.split("=", 1)
        promo = f", promotes to {_PIECE.get(promo_piece[:1], promo_piece)}"

    takes = " takes " if "x" in san else " to "
    body = san.replace("x", "")

    piece = ""
    if body and body[0] in _PIECE:
        piece = _PIECE[body[0]]
        body = body[1:]

    # Drop disambiguation file/rank (e.g. Nbd7, R1e2) — keep destination square.
    match = re.search(r"([a-h][1-8])$", body)
    if not match:
        return f"{piece} {san}".strip() + promo

    dest = spoken_square(match.group(1))
    if piece:
        return f"{piece}{takes}{dest}{promo}"
    if takes.strip() == "takes":
        return f"Pawn takes {dest}{promo}"
    return f"{dest}{promo}"


def spoken_suggestion(san: Optional[str], uci: Optional[str] = None) -> str:
    """Phrase for a coach suggestion, with a beat after 'Play'."""
    if san:
        return f"Play {_PAUSE_MED} {spoken_move(san)}"
    if uci and len(uci) >= 4:
        return (
            f"Play {_PAUSE_MED} {spoken_square(uci[:2])} "
            f"to {_PAUSE_SHORT} {spoken_square(uci[2:4])}"
        )
    return "No move"
