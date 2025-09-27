
import re
import os
import csv
import glob
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import PyPDF2
except Exception as e:
    PyPDF2 = None

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

ROLE_CORE = r'(?:President|Vice-?President|Associate Vice-?President|Secretary|Treasurer|Assistant Treasurer|Second Assistant Treasurer|Auditor|Assistant Auditor|Statistical Secretary|Associate Secretary|Field Secretary|Field Secretaries|Office Secretary)'

NAME_REGEX = re.compile(r'\b((?:[A-Z]\.\s*){1,4}[A-Z][a-zA-Z-]+)\b')  # e.g., "W. A. Spicer", "L. H. Christian"

def read_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is required but not available.")
    reader = PyPDF2.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i+1, text))
    return pages

def normalize_text(raw: str) -> str:
    txt = raw.replace("-\n", "")
    txt = re.sub(r'\n(?!\n)', ' ', txt)
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = txt.replace("—", "-").replace("’", "'").replace("`", "'").strip()
    return txt

OFFICER_GROUP_ALIASES = [
    ("OFFICERS", "officers"),
    ("APPOINTED ASSISTANTS", "appointed assistants"),
    ("GENERAL CONFERENCE COMMITTEE", "general conference committee"),
    ("PRESIDENTS OF UNION CONFERENCES", "presidents of union conferences"),
    ("PRESIDENTS OF UNION CONFER- ENCES", "presidents of union conferences"),
    ("PRESIDENTS OF UNION CONFER-ENCES", "presidents of union conferences"),
    ("PRESIDENTS OF UNION CONFE R- ENCES", "presidents of union conferences"),
    ("PRESIDENTS OF UNION CONFER-ENCES AND SUPERINTENDENTS", "presidents of union conferences"),
    ("SUPERINTENDENTS OF UNION MISSIONS", "superintendents of union missions"),
]

def detect_groups(fulltext: str) -> List[Tuple[str, int]]:
    anchors = []
    for header, normalized in OFFICER_GROUP_ALIASES:
        for m in re.finditer(re.escape(header), fulltext, flags=re.IGNORECASE):
            anchors.append( (normalized, m.start()) )
    anchors = sorted(anchors, key=lambda x: x[1])
    return anchors

def split_sections(fulltext: str) -> List[Tuple[str, str]]:
    anchors = detect_groups(fulltext)
    if not anchors:
        return [("unlabeled", fulltext)]
    chunks = []
    for i, (g, idx) in enumerate(anchors):
        end = anchors[i+1][1] if i+1 < len(anchors) else len(fulltext)
        chunks.append( (g, fulltext[idx:end]) )
    return chunks

def iter_role_blocks(section_text: str) -> List[Tuple[str, str]]:
    role_label = re.compile(rf'(?P<role>{ROLE_CORE}(?: [^:{{}}]{{0,80}})?)\s*:', flags=re.IGNORECASE)
    blocks = []
    positions = [(m.group('role').strip(), m.start(), m.end()) for m in role_label.finditer(section_text)]
    for i, (role, s, e) in enumerate(positions):
        end = positions[i+1][1] if i+1 < len(positions) else len(section_text)
        rest = section_text[e:end].strip()
        blocks.append((role, rest))
    return blocks

def split_names_and_location(rest: str) -> Tuple[List[str], Optional[str]]:
    # Find all name occurrences; assume everything after the last name's ending comma is location
    names = [m.group(1).strip().replace("  ", " ") for m in NAME_REGEX.finditer(rest)]
    # Determine cut point: end position of last match
    last = None
    for m in NAME_REGEX.finditer(rest):
        last = m
    if last:
        after = rest[last.end():].strip(" ,.;")
        location = after if after else None
    else:
        # fallback: treat first comma as split
        if "," in rest:
            nm, loc = rest.split(",", 1)
            names = [nm.strip()]
            location = loc.strip(" ,.;")
        else:
            names = [rest.strip()]
            location = None
    # Clean stray trailing page numbers like "... D. C. 5"
    if location and re.search(r'\b\d{1,2}\s*$', location):
        location = re.sub(r'\b\d{1,2}\s*$', '', location).strip()
    return names, location

def parse_prefix_lastname(fullname: str) -> Tuple[Optional[str], Optional[str]]:
    tokens = fullname.split()
    initials = [t for t in tokens if t.endswith(".")]
    non_initials = [t for t in tokens if not t.endswith(".")]
    prefix = " ".join(initials) if initials else None
    lastname = non_initials[-1] if non_initials else (tokens[-1] if tokens else None)
    return prefix, lastname

def year_from_filename(path: str) -> Optional[int]:
    m = re.search(r'YB(\d{4})\.pdf$', os.path.basename(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_from_pdf(pdf_path: str) -> List[Row]:
    rows: List[Row] = []
    pages = read_pdf_text(pdf_path)
    year = year_from_filename(pdf_path)
    for pageno, raw in pages:
        if not raw:
            continue
        text = normalize_text(raw)
        sections = split_sections(text)
        for group, section in sections:
            if group == "unlabeled":
                continue
            for role, rest in iter_role_blocks(section):
                names, location = split_names_and_location(rest)
                # Extract optional "for <region>" info
                position_information = None
                m = re.match(rf'(?i)({ROLE_CORE})(?:\s+for\s+(.+))?$', role.strip())
                position = role.strip()
                if m:
                    position = m.group(1).strip().lower()
                    if m.group(2):
                        position_information = m.group(2).strip()
                for fullname in names:
                    prefix, lastname = parse_prefix_lastname(fullname)
                    rows.append(Row(
                        conference="General",
                        region=None,
                        institution_name=None,
                        organization=None,
                        group=group,
                        position_information=position_information,
                        position=position,
                        prefix=prefix,
                        name=fullname,
                        lastname=lastname,
                        suffix=None,
                        gender=None,
                        location=location,
                        yearbook_year=year,
                        page=pageno,
                        source_pdf=os.path.basename(pdf_path)
                    ))
    # Deduplicate
    seen = set()
    uniq = []
    for r in rows:
        key = (r.yearbook_year, r.page, r.group, r.position, r.name, r.location)
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(r)
    return uniq

def read_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    reader = PyPDF2.PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i+1, text))
    return pages

def write_csv(rows: List[Row], out_csv: str, append: bool = False):
    field_map = [
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
    headers = [fm[0] for fm in field_map]

    mode = "a" if append else "w"
    write_header = True
    if append and os.path.exists(out_csv):
        try:
            write_header = os.path.getsize(out_csv) == 0
        except OSError:
            write_header = True  # if in doubt, write header

    with open(out_csv, mode, newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        if not append or write_header:
            w.writerow(headers)
        for r in rows:
            d = asdict(r)
            w.writerow([d[fm[1]] for fm in field_map])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="SDA Yearbook officer extractor (1908-1921 style).")
    ap.add_argument("inputs", nargs="+", help="PDF files or glob patterns (e.g., '/path/YB19*.pdf')")
    ap.add_argument("-o","--out", default="yearbook_officers.csv", help="Output CSV path")
    ap.add_argument("--append", action="store_true",
                help="Append to existing CSV instead of overwriting")

    args = ap.parse_args()

    files = []
    for patt in args.inputs:
        matches = glob.glob(patt)
        if not matches and os.path.isfile(patt):
            matches = [patt]
        files.extend(matches)
    files = [f for f in files if f.lower().endswith(".pdf")]
    if not files:
        raise SystemExit("No PDF inputs matched.")

    all_rows: List[Row] = []
    for f in sorted(set(files)):
        try:
            rows = extract_from_pdf(f)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed on {f}: {e}")

    write_csv(all_rows, args.out, append=args.append)
    print(f"Wrote {len(all_rows)} rows to {args.out} ({'append' if args.append else 'overwrite'})")

if __name__ == "__main__":
    main()
