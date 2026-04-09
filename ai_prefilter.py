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

# Common state and territory abbreviations seen in the 1880-1920 yearbooks.
LOC_ABBREV_RE = re.compile(
    r"\b(Ala\.|Ark\.|Cal\.|Col\.|Dak\.|D\. T\.|Ill\.|Ind\.|Iowa|Kan\.|Ky\.|Me\.|Mass\.|Mich\.|Minn\.|Mo\.|Neb\.|N\. Y\.|N\. H\.|Ohio|Ore\.|Pa\.|Tenn\.|Tex\.|Vt\.|Wis\.|W\. T\.)\b"
)

STATE_HEADER_RE = re.compile(r"^[A-Z][A-Z '\.-]{2,}$")  # e.g. MAINE., CALIFORNIA.

# Skip pages that are clearly doctrinal or narrative rather than directory material.
DOCTRINE_BANLIST = [
    "FUNDAMENTAL PRINCIPLES",
    "STATEMENT OF BELIEF",
    "OBITUAR",
    "SERMON",
    "APPEAL TO",
    "TESTIMONY",
    "MISSION REPORT",
]

NAME_TOKENS_RE = re.compile(r"\b(Mr\.|Mrs\.|Miss|Eld\.|Dr\.)\b")
INITIALS_RE = re.compile(r"\b[A-Z]\.\s*(?:[A-Z]\.)?")
ROLE_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z '\-]*:\s+.+$")

def read_pdf(page_path: Path) -> List[str]:
    reader = PdfReader(str(page_path))
    out = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        t = t.replace("\r", "\n")
        # Rejoin words split across a line break in the source PDF.
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
        out.append(t)
    return out

def is_doctrine_page(text: str) -> bool:
    up = text.upper()
    return any(tok in up for tok in DOCTRINE_BANLIST)

def score_line(ln: str) -> int:
    """
    Score a line for directory-like content.
    Lines with a score of at least 1 are kept.
    """
    s = ln.strip()
    if not s:
        return 0
    up = s.upper()

    score = 0
    if any(h in up for h in DIRECTORY_HINTS):
        score += 3
    if STATE_HEADER_RE.match(s) and len(s.split()) <= 4:
        score += 2
    if ROLE_LINE_RE.match(s):
        score += 3
    if any(tok in up for tok in ROLE_TOKENS):
        score += 1
    if NAME_TOKENS_RE.search(s) or INITIALS_RE.search(s):
        score += 1
    if LOC_ABBREV_RE.search(s) or s.count(",") >= 2:
        score += 1
    return score

def filter_page(text: str, context: int = 0) -> str:
    """
    Keep the likely directory lines plus an optional context window.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    keep = [False] * len(lines)

    for idx, ln in enumerate(lines):
        if score_line(ln) >= 1:
            keep[idx] = True
            for j in range(max(0, idx - context), min(len(lines), idx + context + 1)):
                keep[j] = True

    kept_lines = [ln for ln, k in zip(lines, keep) if k and ln.strip()]
    # Deduplicate repeated lines introduced by extraction quirks.
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
    Return the pages whose filtered text still contains likely directory content.
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
