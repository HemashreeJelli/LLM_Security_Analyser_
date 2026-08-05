"""
Layer 1 — Pre-processing / Deobfuscation
=========================================
Pipeline (in order):
  1. HTML entity decode (up to 2 passes for nested entities)
  2. Unicode NFKC normalise
  3. Zero-width char strip
  4. Base64 detect + decode (up to 2 passes)
  5. Whitespace normalise
"""

import base64
import html
import re
import unicodedata
from typing import TypedDict


class PreprocessResult(TypedDict):
    clean_text: str
    original_text: str
    was_base64: bool
    was_html_encoded: bool
    had_zero_width: bool


def _decode_html(text: str, max_passes: int = 2) -> tuple[str, bool]:
    any_decoded = False
    current = text
    for _ in range(max_passes):
        decoded = html.unescape(current)
        if decoded != current:
            any_decoded = True
            current = decoded
        else:
            break
    return current, any_decoded


_ZERO_WIDTH = re.compile(
    r"[\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\u206a-\u206f\ufeff\u00ad]"
)


def _strip_zero_width(text: str) -> tuple[str, bool]:
    stripped = _ZERO_WIDTH.sub("", text)
    return stripped, stripped != text


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


_B64_RE = re.compile(r"(?<!\w)([A-Za-z0-9+/\-_]{16,}={0,2})(?!\w)")


def _try_b64_decode(token: str) -> str | None:
    normalised = token.replace("-", "+").replace("_", "/")
    pad = (4 - len(normalised) % 4) % 4
    try:
        decoded = base64.b64decode(normalised + "=" * pad)
        return decoded.decode("utf-8")
    except Exception:
        return None


def _decode_base64_pass(text: str) -> tuple[str, bool]:
    changed = False

    def replace(m: re.Match) -> str:
        nonlocal changed
        result = _try_b64_decode(m.group(1))
        if result is not None and result.isprintable():
            changed = True
            return result
        return m.group(0)

    return _B64_RE.sub(replace, text), changed


def _decode_base64(text: str, max_passes: int = 2) -> tuple[str, bool]:
    any_decoded = False
    for _ in range(max_passes):
        text, changed = _decode_base64_pass(text)
        if changed:
            any_decoded = True
        else:
            break
    return text, any_decoded


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> PreprocessResult:
    original = text
    text, was_html = _decode_html(text)
    text = _nfkc(text)
    text, had_zw = _strip_zero_width(text)
    text, was_b64 = _decode_base64(text)
    text = _normalise_whitespace(text)

    return PreprocessResult(
        clean_text=text,
        original_text=original,
        was_base64=was_b64,
        was_html_encoded=was_html,
        had_zero_width=had_zw,
    )
