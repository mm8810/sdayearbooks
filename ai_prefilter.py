# ai_prefilter.py
import re
from pathlib import Path
from typing import List, Tuple

from PyPDF2 import PdfReader

ROLE_TOKENS = [
    "president", "vice-president", "vice president",
    "secretary", "treasurer", "auditor", "librarian",
    "agent", "committee", "editor", "board", "director",
]

DIRECTORY_HINTS = [
    "DIRECTORY", "CONFERENCE DIRECTORIES", "GENERAL DIRECTORIES",
    "SABBATH-SCHOOL", "TRACT AND MISSIONARY", "PUBLISHING ASSOCIATION",
    "MINISTERS' DIRECTORY", "EDUCATIONAL SOCIETY", "HEALTH REFORM INSTITUTE", 
    "MINISTERIAL DIRECTORY", "GENERAL ORGANIZATIONS"
]

# Very lightweight state/territory abbreviations commonly found in 1880–1920
LOC_ABBREV_RE = re.compile(
    r"\b(Ala\.|Ark\.|Cal\.|Col\.|Dak\.|D\. T\.|Ill\.|Ind\.|Iowa|Kan\.|Ky\.|Me\.|Mass\.|Mich\.|Minn\.|Mo\.|Neb\.|N\. Y\.|N\. H\.|Ohio|Ore\.|Pa\.|Tenn\.|Tex\.|Vt\.|Wis\.|W\. T\.)\b"
)

STATE_HEADER_RE = re.compile(r"^[A-Z][A-Z '\.-]{2,}$")  # e.g., MAINE., CALIFORNIA.

# Obvious “not interesting” (tune freely)
DOCTRINE_BANLIST = [
    "FUNDAMENTAL PRINCIPLES",
    "STATEMENT OF BELIEF",
    "OBITUAR",  # obituary sections (if you don't want them)
    "SERMON",
    "APPEAL TO",
    "TESTIMONY",
    "MISSION REPORT",
]

NAME_TOKENS_RE = re.compile(r"\b(Mr\.|Mrs\.|Miss|Eld\.|Dr\.)\b")
INITIALS_RE = re.compile(r"\b[A-Z]\.\s*(?:[A-Z]\.)?")  # J. H. or J.
ROLE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z '\-]*:\s+.+$")  # President: Name, Place

def read_pdf(page_path: Path) -> List[str]:
    reader = PdfReader(str(page_path))
    out = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        t = t.replace("\r", "\n")
        # unwrap common hyphenation
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        out.append(t)
    return out

def is_doctrine_page(text: str) -> bool:
    up = text.upper()
    return any(tok in up for tok in DOCTRINE_BANLIST)

def score_line(ln: str) -> int:
    """
    Return an integer score; keep lines with score >= 1.
    """
    s = ln.strip()
    if not s:
        return 0
    up = s.upper()

    score = 0
    # directory headers
    if any(h in up for h in DIRECTORY_HINTS):
        score += 3
    # state header
    if STATE_HEADER_RE.match(s) and len(s.split()) <= 4:
        score += 2
    # role lines
    if ROLE_LINE_RE.match(s):
        score += 3
    # mentions role tokens inline
    if any(tok in up for tok in ROLE_TOKENS):
        score += 1
    # name-ish: honorifics or initials, plus location-ish
    if NAME_TOKENS_RE.search(s) or INITIALS_RE.search(s):
        score += 1
    if LOC_ABBREV_RE.search(s) or s.count(",") >= 2:
        score += 1
    return score

def filter_page(text: str, context: int = 0) -> str:
    """
    Keep only "interesting" lines + optional context window (0 = none).
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    keep = [False] * len(lines)

    for idx, ln in enumerate(lines):
        if score_line(ln) >= 1:
            keep[idx] = True
            # include neighbors for context
            for j in range(max(0, idx - context), min(len(lines), idx + context + 1)):
                keep[j] = True

    kept_lines = [ln for ln, k in zip(lines, keep) if k and ln.strip()]
    # Optional de-dup / compact
    deduped = []
    seen = set()
    for ln in kept_lines:
        key = ln.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(ln)
    return "\n".join(deduped)

def prefilter_pdf_for_ai(pdf_path: str, context_window: int = 0) -> List[Tuple[int, str]]:
    """
    Returns list of (page_num, filtered_text) for pages that contain
    interesting lines. Drops pages that look doctrinal or produce empty output.
    """
    pages = read_pdf(Path(pdf_path))
    out: List[Tuple[int, str]] = []
    for i, raw in enumerate(pages, start=1):
        if is_doctrine_page(raw):
            continue
        ft = filter_page(raw, context=context_window)
        if ft.strip():
            out.append((i, ft))
    return out
