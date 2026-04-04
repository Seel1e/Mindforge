"""
src/preprocessing/clean_text.py
────────────────────────────────
Reusable text-cleaning helpers used by every other preprocessing step.

Plain English: this file has a collection of small functions that
take messy text (typos, HTML tags, emojis, etc.) and return clean text.
"""

import re
import unicodedata
import html
from typing import Optional


# ── Compile patterns once (faster in loops) ──────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"\[.*?\]|\(.*?\)")     # [text] or (text)
_EXTRA_SPACE_RE = re.compile(r"\s{2,}")
_NON_ASCII_PUNCT_RE = re.compile(r"[^\x00-\x7F]+")


def remove_urls(text: str) -> str:
    """Remove http/https URLs and www links."""
    return _URL_RE.sub(" ", text)


def remove_html(text: str) -> str:
    """Decode HTML entities (&amp; → &) then strip tags."""
    text = html.unescape(text)
    return _HTML_TAG_RE.sub(" ", text)


def remove_brackets(text: str) -> str:
    """Remove content inside [] and ()."""
    return _BRACKET_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Convert fancy quotes, em-dashes etc. to ASCII equivalents."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def collapse_whitespace(text: str) -> str:
    """Replace multiple spaces/newlines with a single space."""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return _EXTRA_SPACE_RE.sub(" ", text).strip()


def clean(
    text: str,
    remove_urls_: bool = True,
    remove_html_: bool = True,
    remove_brackets_: bool = False,
    normalize_unicode_: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Master cleaner — applies all selected steps in order.

    Args:
        text:               Raw input string.
        remove_urls_:       Strip URLs.          Default True.
        remove_html_:       Strip HTML tags.     Default True.
        remove_brackets_:   Strip [..] (..))     Default False.
        normalize_unicode_: ASCII-ify text.      Default True.
        lowercase:          Lowercase text.      Default False
                            (we leave case for the LLM tokeniser).
    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        return ""

    if remove_html_:
        text = remove_html(text)
    if remove_urls_:
        text = remove_urls(text)
    if remove_brackets_:
        text = remove_brackets(text)
    if normalize_unicode_:
        text = normalize_unicode(text)
    if lowercase:
        text = text.lower()

    return collapse_whitespace(text)


def truncate(text: str, max_chars: int = 2000) -> str:
    """Hard-truncate to max_chars characters (keeps whole words)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def label_to_int(label: str, mapping: Optional[dict] = None) -> int:
    """
    Convert a string label like 'High' → integer.
    Uses default mental-health-risk mapping if none provided.
    """
    if mapping is None:
        mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(label.strip().lower(), -1)
