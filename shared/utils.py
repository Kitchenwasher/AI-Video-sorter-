"""Shared utility functions – previously duplicated across 3-5 modules."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Type-safe conversion helpers.
# ---------------------------------------------------------------------------

def safe_int(value: Any, default: int = 0) -> int:
    """Convert *value* to ``int``, returning *default* on any error."""
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to ``float``, returning *default* on any error."""
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Filesystem helpers.
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temporary file."""
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def move_video_collision_safe(src: Path, dst_dir: Path) -> Path:
    """Move *src* into *dst_dir*, appending a counter suffix on collision.

    If *src* already resides at the destination (same resolved path) the
    function returns *src* without moving.
    """
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        if src.resolve() == dst.resolve():
            return src
    except Exception:
        pass
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while True:
            candidate = dst_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            counter += 1
    shutil.move(str(src), str(dst))
    return dst


# ---------------------------------------------------------------------------
# Timestamp helpers.
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Normalisation helpers.
# ---------------------------------------------------------------------------

def normalize_cache_key(path_text: str) -> str:
    """Normalise a filesystem path for use as a cache dictionary key."""
    return os.path.normcase(os.path.normpath(str(path_text).strip()))


def clamp_confidence(value: float) -> float:
    """Clamp a confidence score to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Download helpers.
# ---------------------------------------------------------------------------

def download_file(url: str, dst: Path) -> None:
    """Download *url* to *dst*, skipping if the file already exists."""
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"Downloading model asset: {dst.name}")
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    if tmp_dst.exists():
        tmp_dst.unlink()
    try:
        urllib.request.urlretrieve(url, tmp_dst)
        tmp_dst.replace(dst)
    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise


def download_first_available(urls: Sequence[str], dst: Path) -> None:
    """Try each URL in *urls* until a download succeeds."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            download_file(url, dst)
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to download {dst.name} from all known sources. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Console helpers.
# ---------------------------------------------------------------------------

def safe_console_print(message: Any, *, flush: bool = False) -> None:
    """Print *message* with graceful fallback on encoding errors."""
    text = str(message)
    try:
        print(text, flush=flush)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        line = text if text.endswith("\n") else f"{text}\n"
        payload = line.encode(encoding, errors="replace")
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(payload)
            if flush:
                buffer.flush()
            return
        sys.stdout.write(payload.decode(encoding, errors="replace"))
        if flush:
            sys.stdout.flush()


def configure_console_encoding() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement errors."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
