
#!/usr/bin/env python3
"""

This script scans the entire PDF, detects directory-like sections, tracks hierarchy
(Section/Directory -> Subsection/Conference/Region -> Group), and extracts roster lines.

It writes two sheets:
- "yearbook_directory": unified, normalized rows
- "parse_errors": raw lines that couldn't parse

USAGE:
  python parse_sda_yearbook_all_dirs.py YB1883.pdf -o YB1883_all_dirs.xlsx
  python parse_sda_yearbook_all_dirs.py YB1883.pdf -o out.xlsx --template-xlsx YB1884.xlsx
"""

import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pdfplumber
import pandas as pd
import ollama

# ------------------------------------------------------------
# Config: output schema (will realign to template if provided)
# ------------------------------------------------------------
BASE_COLUMNS = [
    "directory",              # NEW: high-level section, e.g., 'CONFERENCE DIRECTORY'
    "conference",             # e.g., 'California Conference'
    "region",                 # e.g., 'California', 'District No. 1'
    "institution-name",       # e.g., 'Pacific Press'
    "organization",           # umbrella org, if applied
    "group",                  # e.g., 'Officers', 'Executive Committee'
    "position-information",   # text payload on/near position (e.g., 'acting', 'corresponding')
    "position",               # President, Secretary, Treasurer, Member, etc.
    "prefix",
    "name",
    "lastname",
    "suffix",
    "location",
    "yearbook-year",
    "page",
    "raw_line",               # keep a copy of the source line that produced the row
]

# Common title/role tokens you’ll see across directories
ROLE_TOKENS = [
    "President", "Vice-President", "Vice President", "Secretary", "Treasurer",
    "Recording Secretary", "Corresponding Secretary", "Gen. Secretary", "General Secretary",
    "State Agent", "Executive Committee", "Committee", "Member", "Editor",
    "Business Manager", "Manager", "Superintendent", "Director",
    "Assistant", "Asst.", "Advisory Board", "Trustee", "Board of Trustees",
    "Directors", "Publishing Committee", "Librarian",
]

# Name affixes
PREFIXES = {
    "Eld.", "Elder", "Bro.", "Brother", "Mr.", "Mrs.", "Miss", "Dr.", "Dr",
    "Pastor", "Pr.", "Prof.", "Sister", "Sr."
}
SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV", "M.D.", "MD", "D.D.", "Ph.D."}

# Headings detection
ALLCAPS_LINE = re.compile(r"^[A-Z0-9 ,.'&()\/\-]+$")

# Lines that strongly indicate a *section* heading for a directory
SECTION_KEYS = [
    "DIRECTORY", "DIRECTORIES",
    "CONFERENCE", "STATE", "GENERAL CONFERENCE",
    "SABBATH", "TRACT", "MISSION", "PUBLISH", "INSTITUTION", "SANITARIUM", "COLLEGE", "SCHOOL",
    "BIBLE", "EDUCATION", "ASSOCIATION", "SOCIETY", "SOCIETIES", "DISTRICT"
]

# Splitters between position and the rest
POS_SPLIT = re.compile(r"\s*(?:—|–|-|:)\s*")

# Split on first comma (Name, Location)
FIRST_COMMA_SPLIT = re.compile(r"\s*,\s*")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm(s: str) -> str:
    return " ".join(s.replace("—", "-").replace("–", "-").split())

def is_allcaps_heading(line: str) -> bool:
    L = line.strip()
    if len(L) < 2 or len(L) > 80:
        return False
    if ALLCAPS_LINE.match(L) is None:
        return False
    # mixed cases like 'No.' allowed; keep permissive
    # avoid pure page numbers
    if re.fullmatch(r"\d{1,4}", L):
        return False
    return True

def looks_like_section_header(line: str) -> bool:
    if not is_allcaps_heading(line):
        return False
    L = line.upper()
    # A directory section header usually mentions one of our keys
    return any(k in L for k in SECTION_KEYS)

def looks_like_subheader(line: str) -> bool:
    # Subheaders (Conference/Region/Institution names) tend to be ALLCAPS but
    # without the big "DIRECTORY" keyword.
    if not is_allcaps_heading(line):
        return False
    L = line.upper()
    if any(k in L for k in ["DIRECTORY", "DIRECTORIES"]):
        return False
    return True

def split_name_block(name_block: str) -> Tuple[Optional[str], str, str, Optional[str]]:
    s = norm(name_block)
    tokens = s.split()
    if not tokens:
        return None, "", "", None

    prefix = None
    suffix = None

    # prefix
    if tokens[0] in PREFIXES or (tokens[0].rstrip(".") + ".") in PREFIXES:
        prefix = tokens.pop(0)

    # suffix
    if tokens and tokens[-1] in SUFFIXES:
        suffix = tokens.pop()

    if not tokens:
        return prefix, "", "", suffix

    if len(tokens) == 1:
        return prefix, "", tokens[0].rstrip(",").strip(), suffix

    lastname = tokens[-1].rstrip(",").strip()
    given = " ".join(tokens[:-1])
    return prefix, given, lastname, suffix

def extract_position(line_left: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Attempt to split 'Position — Name, Location' patterns.
    Returns (position, position_info, remainder_after_position)
    """
    # Examples:
    # 'President—J. H. Waggoner, Oakland, Cal.'
    # 'Recording Secretary: A. Smith, Battle Creek, Mich.'
    # 'Executive Committee—J. Doe; R. Roe; A. Poe, City, ST'
    L = line_left
    # Try to split on the first dash/colon that looks like a position separator
    m = POS_SPLIT.split(L, maxsplit=1)
    if len(m) == 2:
        maybe_pos, rest = m[0].strip(), m[1].strip()
        # position-info heuristic: in cases like 'Assistant Secretary'
        # we treat whole left side as position (we’ll not overfit here)
        position = maybe_pos
        pos_info = None
        return position, pos_info, rest
    # Could be 'President, J. H. Waggoner, Oakland...' (rare)
    if "," in L:
        left, rest = L.split(",", 1)
        if any(tok.lower() in left.lower() for tok in ["president", "secretary", "treasurer", "committee", "editor", "manager", "trustee", "director", "agent"]):
            return left.strip(), None, rest.strip()
    return None, None, L

def split_name_and_location(s: str) -> Tuple[str, str]:
    """
    Split 'Name, Location' on first comma. If no comma, return (s, "").
    """
    parts = FIRST_COMMA_SPLIT.split(s, maxsplit=1)
    if len(parts) == 1:
        return parts[0].strip(), ""
    return parts[0].strip(), parts[1].strip()

def clean_text_lines(page_text: str) -> List[str]:
    lines = []
    for raw in (page_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"\d{1,4}", line):  # likely an isolated page number
            continue
        lines.append(line)
    return lines

def infer_year_from_filename(path: str) -> Optional[int]:
    m = re.search(r"(18|19)\d{2}", path)
    return int(m.group(0)) if m else None

# ------------------------------------------------------------
# Parser state
# ------------------------------------------------------------

@dataclass
class Ctx:
    year: Optional[int]
    directory: Optional[str] = None   # e.g., 'CONFERENCE DIRECTORY'
    conference: Optional[str] = None  # 'California Conference'
    region: Optional[str] = None      # 'California' or 'District No. 1'
    organization: Optional[str] = None
    institution_name: Optional[str] = None
    group: Optional[str] = None       # 'Officers', 'Executive Committee', etc.

    def reset_section(self):
        self.directory = None
        self.conference = None
        self.region = None
        self.organization = None
        self.institution_name = None
        self.group = None

    def reset_subheaders(self):
        self.conference = None
        self.region = None
        self.organization = None
        self.institution_name = None
        self.group = None

# ------------------------------------------------------------
# Core parsing
# ------------------------------------------------------------

def parse_yearbook(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    errors: List[Dict] = []

    ctx = Ctx(year=infer_year_from_filename(pdf_path))

    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            lines = clean_text_lines(text)

            for raw_line in lines:
                line = norm(raw_line)

                # 1) Detect NEW DIRECTORY SECTION (top-level)
                if looks_like_section_header(line):
                    # If it's clearly a directory-like section, treat as new section
                    if any(k in line.upper() for k in SECTION_KEYS):
                        ctx.reset_section()
                        ctx.directory = line.upper()
                        # Often next lines are subheaders like conference/region/institution
                        continue

                # 2) Subheaders (conference/region/institution names)
                if looks_like_subheader(line):
                    U = line.upper()
                    # Guess what this subheader is:
                    # Heuristics (order matters)
                    if "CONFERENCE" in U and len(U) < 70:
                        ctx.conference = line.title()
                        ctx.region = None
                        ctx.organization = None
                        ctx.institution_name = None
                        ctx.group = None
                        continue
                    if any(w in U for w in ["PRESS", "PUBLISH", "INSTITUTION", "SANITARIUM", "COLLEGE", "SCHOOL", "INSTITUTE"]):
                        ctx.institution_name = line.title()
                        ctx.organization = None
                        ctx.group = None
                        continue
                    if any(w in U for w in ["DISTRICT", "PROVINCE", "STATE", "TERRITORY", "MISSION"]):
                        ctx.region = line.title()
                        ctx.group = None
                        continue
                    # Fallback: treat as a region-ish grouping
                    ctx.region = line.title()
                    ctx.group = None
                    continue

                # 3) Group labels (not all-caps but look like 'Officers:', 'Executive Committee', etc.)
                lower = line.lower().rstrip(":")
                if any(tok.lower() in lower for tok in ["officers", "executive committee", "committee", "board of trustees", "directors", "publishing committee", "faculty", "teachers", "agents"]):
                    ctx.group = line.rstrip(":")
                    continue

                # 4) Entry lines (general pattern):
                # Try to find 'Position — Rest'
                position, position_info, rest = extract_position(line)

                # If it *starts* with role info, we’re in 'role — person, location' shape
                if position:
                    # Some lines contain multiple names (e.g., Exec Committee—A; B; C; City, ST)
                    # Split on semicolons; last part might hold a shared location.
                    parts = [p.strip() for p in re.split(r"\s*;\s*", rest) if p.strip()]
                    # If the final part clearly looks like a location (has a comma and few words), we'll treat as shared location
                    shared_location = ""
                    if parts and ("," in parts[-1]) and (len(parts[-1].split()) <= 6):
                        # This might also be 'Name, Location' though; we’ll not force it—handle per part.
                        pass

                    # Emit one row per part; parse "Name, Location" for each.
                    for part in parts if parts else [rest]:
                        if not part:
                            continue
                        name_str, loc = split_name_and_location(part)
                        prefix, given, lastname, suffix = split_name_block(name_str)

                        if not lastname and not given:
                            # Might be a continuation line; log as error
                            errors.append({
                                "page": pageno,
                                "directory": ctx.directory,
                                "conference": ctx.conference,
                                "region": ctx.region,
                                "institution-name": ctx.institution_name,
                                "group": ctx.group,
                                "line": raw_line,
                                "reason": "No name parsed after position"
                            })
                            continue

                        rows.append({
                            "directory": ctx.directory,
                            "conference": ctx.conference,
                            "region": ctx.region,
                            "institution-name": ctx.institution_name,
                            "organization": ctx.organization,
                            "group": ctx.group,
                            "position-information": position_info,
                            "position": position,
                            "prefix": prefix,
                            "name": given,
                            "lastname": lastname,
                            "suffix": suffix,
                            "location": loc,
                            "yearbook-year": ctx.year,
                            "page": pageno,
                            "raw_line": raw_line,
                        })
                    continue

                # 5) Non-position lines may still be roster rows like 'J. H. Waggoner, Oakland, Cal.'
                # or 'Pacific Press—Oakland, Cal.' or 'Address: ...'
                # We first check if it looks like 'Name, Location'
                if "," in line:
                    name_str, loc = split_name_and_location(line)
                    prefix, given, lastname, suffix = split_name_block(name_str)

                    # If there's clearly a person's name, accept
                    if lastname or given:
                        rows.append({
                            "directory": ctx.directory,
                            "conference": ctx.conference,
                            "region": ctx.region,
                            "institution-name": ctx.institution_name,
                            "organization": ctx.organization,
                            "group": ctx.group,
                            "position-information": None,
                            "position": None,
                            "prefix": prefix,
                            "name": given,
                            "lastname": lastname,
                            "suffix": suffix,
                            "location": loc,
                            "yearbook-year": ctx.year,
                            "page": pageno,
                            "raw_line": raw_line,
                        })
                        continue

                    # If not a person, treat as organization + location (e.g., 'Pacific Press, Oakland, Cal.')
                    if not (lastname or given) and ctx.institution_name is None:
                        # store as institution row
                        rows.append({
                            "directory": ctx.directory,
                            "conference": ctx.conference,
                            "region": ctx.region,
                            "institution-name": name_str.title(),
                            "organization": ctx.organization,
                            "group": ctx.group,
                            "position-information": None,
                            "position": None,
                            "prefix": None,
                            "name": "",
                            "lastname": "",
                            "suffix": None,
                            "location": loc,
                            "yearbook-year": ctx.year,
                            "page": pageno,
                            "raw_line": raw_line,
                        })
                        continue

                # 6) Continuations (wrapped locations or addenda)
                # If previous row exists and this line looks short and address-like, append to location.
                if rows and (len(line) <= 64) and (" " in line) and not looks_like_section_header(line) and not looks_like_subheader(line):
                    rows[-1]["location"] = (rows[-1]["location"] + " " + line).strip()
                    continue

                # 7) Otherwise, we don't know what this is—log as unparsed
                errors.append({
                    "page": pageno,
                    "directory": ctx.directory,
                    "conference": ctx.conference,
                    "region": ctx.region,
                    "institution-name": ctx.institution_name,
                    "group": ctx.group,
                    "line": raw_line,
                    "reason": "Unrecognized pattern",
                })

    return rows, errors

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def align_columns(df: pd.DataFrame, template_path: Optional[str]) -> pd.DataFrame:
    if template_path:
        try:
            # Use first sheet columns as template order
            tdf = pd.read_excel(template_path, nrows=0)
            template_cols = list(tdf.columns)
            # ensure all template columns exist
            for c in template_cols:
                if c not in df.columns:
                    df[c] = None
            # keep extra columns at the end
            extras = [c for c in df.columns if c not in template_cols]
            df = df[template_cols + extras]
            return df
        except Exception:
            pass  # fallback to base order if template not loadable

    # no template: use base order; add any missing; append extras
    for c in BASE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    ordered = BASE_COLUMNS + [c for c in df.columns if c not in BASE_COLUMNS]
    return df[ordered]

def write_excel(rows: List[Dict], errors: List[Dict], out_path: str, template_path: Optional[str]):
    df = pd.DataFrame(rows)
    df = align_columns(df, template_path)

    # sort for consistency
    sort_cols = [c for c in ["directory", "conference", "region", "institution-name", "group", "position", "lastname", "name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last", kind="mergesort")

    err_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["page","directory","conference","region","institution-name","group","line","reason"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="yearbook_directory")
        err_df.to_excel(xw, index=False, sheet_name="parse_errors")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse ALL directories from SDA Yearbook PDF into Excel.")
    ap.add_argument("pdf", help="Path to Yearbook PDF (e.g., YB1883.pdf)")
    ap.add_argument("-o", "--out", default=None, help="Output .xlsx path (default: <pdf>_alldirs.xlsx)")
    ap.add_argument("--template-xlsx", default=None, help="Optional: an Excel file whose column order to mirror (e.g., YB1884.xlsx)")
    args = ap.parse_args()

    out = args.out
    if not out:
        base = re.sub(r"\.pdf$", "", args.pdf, flags=re.IGNORECASE)
        out = f"{base}_alldirs.xlsx"

    rows, errors = parse_yearbook(args.pdf)
    write_excel(rows, errors, out, args.template_xlsx)

    print(f"[OK] Wrote {len(rows)} rows to: {out}")
    if errors:
        print(f"[NOTE] Logged {len(errors)} lines in parse_errors sheet for follow-up tuning.")

if __name__ == "__main__":
    import re
    main()
