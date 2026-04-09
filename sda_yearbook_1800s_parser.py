#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os, csv, glob, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

try:
    from PyPDF2 import PdfReader
except Exception:
    raise SystemExit("PyPDF2>=3.0.0 required. pip install PyPDF2")

# Internal row model used by the rule-based extractor.
@dataclass
class Row:
    conference: Optional[str] = None
    region: Optional[str] = None
    institution_name: Optional[str] = None
    organization: Optional[str] = None
    group: Optional[str] = None
    position_information: Optional[str] = None
    position: Optional[str] = None
    prefix: Optional[str] = None
    name: Optional[str] = None
    lastname: Optional[str] = None
    suffix: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    yearbook_year: Optional[int] = None
    page: Optional[int] = None
    source_pdf: Optional[str] = None

# Default CSV schema used when no sample output file is provided.
DEFAULT_SCHEMA = [
    ("conference","conference"),
    ("region","region"),
    ("institution-name","institution_name"),
    ("organization","organization"),
    ("group","group"),
    ("position-information","position_information"),
    ("position","position"),
    ("prefix","prefix"),
    ("name","name"),
    ("lastname","lastname"),
    ("suffix","suffix"),
    ("gender","gender"),
    ("location","location"),
    ("yearbook-year","yearbook_year"),
    ("page","page"),
    ("source-pdf","source_pdf"),
]

# PDF helpers.
def read_pdf_text_per_page(path: str) -> List[Tuple[int, str]]:
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.replace("\r", "\n")
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        # Rejoin words split by end-of-line hyphenation in the source PDF.
        txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
        pages.append((i+1, txt))
    return pages

# Section detection for the 1880s books.
SECTION_PATTERNS = [
    r"MINISTERS['’]\s+DIRECTORY",
    r"STATE\s+SABBATH-?SCHOOL\s+ASSOCIATION\s+DIRECTORIES",
    r"STATE\s+TRACT\s+AND\s+MISSIONARY\s+SOCIETY\s+DIRECTORIES",
    r"STATE\s+CONFERENCE\s+DIRECTORIES",
    r"GENERAL\s+SABBATH-?SCHOOL\s+ASSOCIATION\s+DIRECTORY",
    r"(?:INTERNATIONAL|GENERAL)\s+TRACT\s+AND\s+MISSIONARY(?:\s+SOCIETY)?\s+DIRECTORY",
    r"SEVENTH-?DAY\s+ADVENTIST\s+PUBLISHING\s+ASSOCIATION\s+DIRECTORY",
    r"PACIFIC\s+SEVENTH-?DAY\s+ADVENTIST\s+PUBLISHING\s+ASSOCIATION\s+DIRECTORY",
    r"HEALTH\s+REFORM\s+INSTITUTE\s+DIRECTORY",
    r"EDUCATIONAL\s+SOCIETY\s+DIRECTORY",
    r"GENERAL\s+CONFERENCE\s+DIRECTORY",
    r"GENERAL\s+DIRECTORIES",
]
SECTION_RE = re.compile("|".join(f"(?:{p})" for p in SECTION_PATTERNS), re.I)
STATE_HEADER_RE = re.compile(r"^[A-Z][A-Z '\.-]{2,}$")

def normalize_section_name(h: str) -> str:
    h = re.sub(r"\s+", " ", h.upper()).strip()
    if "MINISTERS" in h and "DIRECTORY" in h:
        return "MINISTERS' DIRECTORY"
    if "GENERAL DIRECTORIES" in h:
        return "GENERAL DIRECTORIES"
    return h

def split_sections(pages: List[Tuple[int,str]]) -> List[Dict]:
    sections = []
    current = None
    for pg, text in pages:
        for ln in [x.strip() for x in text.split("\n") if x.strip()]:
            if SECTION_RE.search(ln):
                if current:
                    sections.append(current)
                current = {"name": normalize_section_name(ln), "page": pg, "text": ln + "\n"}
            elif current:
                current["text"] += ln + "\n"
    if current:
        sections.append(current)
    return sections

# Parsing helpers.
NAME_TITLES = r"(?:Mrs\.?|Miss|Mr\.?|Eld\.?|Dr\.?)"
INITIAL = r"(?:[A-Z]\.)"
WORD = r"(?:[A-Za-z][A-Za-z'\-]*)"
ROLE_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z '\-]*?)\s*:\s*(.+)$")

def looks_like_location(s: str) -> bool:
    return bool(re.search(r"\b(Ala\.|Ark\.|Cal\.|Col\.|Dak\.|D\. T\.|Ill\.|Ind\.|Iowa|Kan\.|Ky\.|Me\.|Mass\.|Mich\.|Minn\.|Mo\.|Neb\.|N\. Y\.|N\. H\.|Ohio|Ore\.|Pa\.|Tenn\.|Tex\.|Vt\.|Wis\.|W\. T\.)\b", s))

def looks_like_name_token(s: str) -> bool:
    return bool(re.fullmatch(rf"(?:{INITIAL}|{WORD})", s.replace(" ", "")))

def split_name_and_location(chunk: str) -> Tuple[str, str]:
    parts = [p.strip(" ;") for p in chunk.split(",")]
    if len(parts) <= 1:
        return chunk.strip(), ""
    if looks_like_location(", ".join(parts[1:])):
        return parts[0].strip(), ", ".join(parts[1:]).strip()
    if len(parts) >= 3 and looks_like_name_token(parts[1]):
        return (", ".join(parts[:2])).strip(), ", ".join(parts[2:]).strip()
    return parts[0].strip(), ", ".join(parts[1:]).strip()

def split_prefix_core_last(name: str) -> Tuple[Optional[str], str, str, Optional[str]]:
    s = name.strip().strip(",;")
    prefix = None
    suffix = None
    m = re.match(rf"^\s*({NAME_TITLES})\s+(.*)$", s, re.I)
    if m:
        prefix = m.group(1)
        s = m.group(2).strip()
    toks = s.split()
    if not toks:
        return prefix, "", "", suffix
    lastname = toks[-1].rstrip(",.;")
    core = " ".join(toks[:-1])
    return prefix, core, lastname, suffix

def derive_gender(prefix: Optional[str]) -> Optional[str]:
    if not prefix:
        return None
    p = prefix.lower().replace(".", "")
    if p in ("mrs","miss"):
        return "F"
    if p in ("mr","eld","dr"):
        return "M"
    return None

def infer_conference_from_section(section_name: str) -> Optional[str]:
    s = section_name.upper()
    if "GENERAL CONFERENCE" in s or "GENERAL DIRECTORIES" in s:
        return "General"
    return None

def map_section_to_org(section_name: str) -> Optional[str]:
    s = section_name.upper()
    if "TRACT AND MISSIONARY" in s:
        return "International Tract and Missionary Society"
    if "SABBATH-SCHOOL" in s:
        return "General Sabbath-school Association"
    if "PUBLISHING ASSOCIATION" in s:
        return "Publishing Association"
    if "HEALTH REFORM INSTITUTE" in s:
        return "Health Reform Institute"
    if "EDUCATIONAL SOCIETY" in s:
        return "Educational Society"
    if "GENERAL CONFERENCE" in s or "GENERAL DIRECTORIES" in s:
        return "General Conference"
    if "STATE CONFERENCE DIRECTORIES" in s:
        return "State Conference"
    if "STATE TRACT AND MISSIONARY SOCIETY" in s:
        return "State Tract and Missionary Society"
    if "STATE SABBATH-SCHOOL ASSOCIATION" in s:
        return "State Sabbath-school Association"
    if "MINISTERS' DIRECTORY" in s:
        return "Ministers"
    return None

def map_section_to_group(section_name: str) -> Optional[str]:
    s = section_name.upper()
    if "MINISTERS' DIRECTORY" in s:
        return "ministers"
    return "officers"

# Section parsers.
def parse_directory_block(section_name: str, text: str, year: Optional[int], src_pdf: str, start_page: int) -> List[Row]:
    rows: List[Row] = []
    current_state = None
    for raw in text.splitlines():
        line = raw.strip().strip("·•—-")
        if not line:
            continue
        if STATE_HEADER_RE.match(line) and len(line.split()) <= 4:
            current_state = line.strip(". ")
            continue
        m = ROLE_LINE_RE.match(line)
        if m:
            role = m.group(1).strip().rstrip(":")
            val = m.group(2).strip()
            parts = re.split(r"\s*;\s*|\s+—\s+", val)
            for part in parts:
                name_chunk, loc = split_name_and_location(part)
                prefix, core, lastname, suffix = split_prefix_core_last(name_chunk)
                gender = derive_gender(prefix)
                rows.append(Row(
                    conference=current_state if current_state else infer_conference_from_section(section_name),
                    organization=map_section_to_org(section_name),
                    group=map_section_to_group(section_name),
                    position=role.lower() if role else None,
                    prefix=prefix,
                    name=core if core else name_chunk,
                    lastname=lastname or None,
                    suffix=suffix,
                    gender=gender,
                    location=loc or None,
                    yearbook_year=year,
                    page=start_page,
                    source_pdf=os.path.basename(src_pdf)
                ))
    return rows

def parse_ministers_directory(section_name: str, text: str, year: Optional[int], src_pdf: str, start_page: int) -> List[Row]:
    rows: List[Row] = []
    current_state = None
    current_label = None
    for raw in text.splitlines():
        line = raw.strip().strip("·•—-")
        if not line:
            continue
        if STATE_HEADER_RE.match(line) and len(line.split()) <= 4:
            current_state = line.strip(". ")
            current_label = None
            continue
        if re.fullmatch(r"(MINISTERS\.|LICENTIATES\.)", line, re.I):
            current_label = line.rstrip(".").upper()
            continue
        if current_state:
            name_chunk, loc = split_name_and_location(line)
            if not name_chunk:
                continue
            prefix, core, lastname, suffix = split_prefix_core_last(name_chunk)
            gender = derive_gender(prefix)
            rows.append(Row(
                conference=current_state,
                organization="General",
                group="ministers" if current_label == "MINISTERS" else ("licentiates" if current_label == "LICENTIATES" else None),
                position="minister" if current_label == "MINISTERS" else ("licentiate" if current_label == "LICENTIATES" else None),
                prefix=prefix,
                name=core if core else name_chunk,
                lastname=lastname or None,
                suffix=suffix,
                gender=gender,
                location=loc or None,
                yearbook_year=year,
                page=start_page,
                source_pdf=os.path.basename(src_pdf),
            ))
    return rows

# High-level extraction entry point.
def guess_year_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"(18\d{2})", base)
    return int(m.group(1)) if m else None

def extract_from_pdf(path: str) -> List[Row]:
    pages = read_pdf_text_per_page(path)
    year = guess_year_from_filename(path)
    sections = split_sections(pages)
    out: List[Row] = []
    for sec in sections:
        name, start_page, text = sec["name"], sec["page"], sec["text"]
        if "MINISTERS' DIRECTORY" in name:
            out.extend(parse_ministers_directory(name, text, year, path, start_page))
        else:
            out.extend(parse_directory_block(name, text, year, path, start_page))
    return out

# CSV writing helpers.
def load_schema_from_csv(sample_csv: Optional[str]) -> List[Tuple[str,str]]:
    """
    Copy the exact headers from a sample CSV when one is provided.
    Internal fields are matched to those headers with a best-effort mapping.
    """
    if not sample_csv:
        return DEFAULT_SCHEMA

    hdrs: List[str]
    with open(sample_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdrs = next(r)

    # Match headers case-insensitively and ignore punctuation differences.
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    candidates = {
        "conference": ["conference"],
        "region": ["region", "state", "territory"],
        "institution_name": ["institutionname", "institution_name"],
        "organization": ["organization", "org"],
        "group": ["group"],
        "position_information": ["positioninformation", "position_info", "notes"],
        "position": ["position", "title", "role"],
        "prefix": ["prefix", "honorific"],
        "name": ["name", "given_names", "first_middle"],
        "lastname": ["lastname", "surname", "last_name"],
        "suffix": ["suffix"],
        "gender": ["gender", "sex"],
        "location": ["location", "address", "citystate"],
        "yearbook_year": ["yearbook", "yearbookyear", "year"],
        "page": ["page", "pagenum", "page_number"],
        "source_pdf": ["sourcepdf", "source", "file"],
    }

    mapping: List[Tuple[str,str]] = []
    for h in hdrs:
        hnorm = norm(h)
        matched_internal = None
        for internal, keys in candidates.items():
            if any(hnorm == k for k in keys):
                matched_internal = internal
                break
        mapping.append((h, matched_internal))

    return mapping

def write_csv(rows: List[Row], out_csv: str, sample_csv: Optional[str] = None):
    schema = load_schema_from_csv(sample_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([col for col, _ in schema])
        for r in rows:
            d = asdict(r)
            row = []
            for _, internal in schema:
                row.append(d.get(internal, "") if internal else "")
            w.writerow(row)

# CLI entry point.
def main():
    ap = argparse.ArgumentParser(description="Rule-based extractor for 1880s SDA Yearbooks (e.g., 1883, 1885).")
    ap.add_argument("inputs", nargs="*", help="PDFs or glob patterns (default: YB188*.pdf)")
    ap.add_argument("-o", "--out", default="yearbook_officers_1800s.csv", help="Output CSV")
    ap.add_argument("--schema-csv", help="Optional: copy column headers (order & names) from this CSV (e.g., 'YB1883 - Sheet1.csv')")
    args = ap.parse_args()

    files: List[str] = []
    if not args.inputs:
        files = glob.glob("YB188*.pdf")
        if not files:
            raise SystemExit("No inputs. Provide PDFs or place YB188*.pdf in the current directory.")
    else:
        for patt in args.inputs:
            m = glob.glob(patt)
            if not m and os.path.isfile(patt):
                m = [patt]
            files.extend(m)

    all_rows: List[Row] = []
    for pdf in sorted(set([f for f in files if f.lower().endswith(".pdf")])):
        print(f"Parsing {os.path.basename(pdf)} …")
        try:
            rows = extract_from_pdf(pdf)
            all_rows.extend(rows)
            print(f"  + {len(rows)} rows")
        except Exception as e:
            print(f"[WARN] Failed on {pdf}: {e}")

    write_csv(all_rows, args.out, sample_csv=args.schema_csv)
    print(f"wrote {len(all_rows)} rows → {args.out}")

if __name__ == "__main__":
    main()
