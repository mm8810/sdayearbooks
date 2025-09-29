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

NAME_REGEX = re.compile(r'\b((?:[A-Z]\.\s*){1,4}[A-Z][a-zA-Z-]+)\b')

# Pattern to detect conference headers
CONFERENCE_PATTERN = re.compile(
    r'\b([A-Z][A-Z\s-]+)\s+(?:UNION\s+)?CONFERENCE\b',
   # flags=re.IGNORECASE
)

# Pattern to detect mission headers
MISSION_PATTERN = re.compile(
    r'\b([A-Z][A-Z\s-]+)\s+MISSION\b',
   # flags=re.IGNORECASE
)

# Pattern for list-style entries: "Region: Name, Location"
REGION_LIST_PATTERN = re.compile(
    r'^([A-Z][A-Za-z\s\-]+?):\s*([A-Z][A-Za-z\s\.\-]+?),\s*(.+?)$',
    flags=re.MULTILINE
)

# Extended role pattern with qualifiers
ROLE_WITH_QUALIFIER = re.compile(
    rf'(?P<role>{ROLE_CORE})\s*(?:for\s+(?P<qualifier>[^:]+))?:',
    flags=re.IGNORECASE
)

# Pattern to detect section headers that indicate list-style content
LIST_SECTION_HEADERS = [
    "PRESIDENTS OF UNION CONFERENCES",
    "SUPERINTENDENTS OF UNION MISSIONS",
    "SECRETARIES OF DEPARTMENTS",
    "GENERAL MISSIONARY AGENTS",
    "GENERAL MISSIONARY SECRETARIES",
    "PUBLISHING HOUSE MANAGERS",
    "MANAGERS CIRCULATING DEPARTMENTS",
]

MEMBER_SECTION_HEADERS = [
    "OTHER MEMBERS",
    "GENERAL MEMBERS",
    "GENERAL CONFERENCE COMMITTEE",
]

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
    txt = txt.replace("—", "-").replace("'", "'").replace("`", "'").strip()
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
    ("SECRETARIES OF DEPARTMENTS", "secretaries of departments"),
    ("GENERAL MISSIONARY AGENTS", "general missionary agents"),
    ("GENERAL MISSIONARY SECRETARIES", "general missionary secretaries"),
    ("PUBLISHING HOUSE MANAGERS", "publishing house managers"),
    ("MANAGERS CIRCULATING DEPARTMENTS", "managers circulating departments"),
    ("OTHER MEMBERS", "other members"),
    ("GENERAL MEMBERS", "general members"),
    ("MINISTERS", "ministers"),
    ("LICENTIATES", "licentiates"),
    ("MISSIONARY LICENTIATES", "missionary licentiates"),
]

def detect_conference(text: str, start_pos: int) -> Optional[Tuple[str, int]]:
    """Detect conference name near the given position."""
    # Look backwards up to 500 chars for a conference header
    search_start = max(0, start_pos - 1000)
    search_text = text[search_start:start_pos + 100]
    
    # Try to find conference pattern
    matches = list(CONFERENCE_PATTERN.finditer(search_text))
    if matches:
        last_match = matches[-1]
        conf_name = last_match.group(1).strip().title()
        return (conf_name, search_start + last_match.start())
    
    # Try to find mission pattern
    matches = list(MISSION_PATTERN.finditer(search_text))
    if matches:
        last_match = matches[-1]
        conf_name = last_match.group(1).strip().title()
        return (conf_name, search_start + last_match.start())
    
    return None

def detect_groups(fulltext: str) -> List[Tuple[str, int]]:
    anchors = []
    for header, normalized in OFFICER_GROUP_ALIASES:
        for m in re.finditer(re.escape(header), fulltext, flags=re.IGNORECASE):
            anchors.append( (normalized, m.start()) )
    anchors = sorted(anchors, key=lambda x: x[1])
    return anchors

def split_sections(fulltext: str) -> List[Tuple[str, str, int]]:
    """Returns (group_name, section_text, start_position)"""
    anchors = detect_groups(fulltext)
    if not anchors:
        return [("unlabeled", fulltext, 0)]
    chunks = []
    for i, (g, idx) in enumerate(anchors):
        end = anchors[i+1][1] if i+1 < len(anchors) else len(fulltext)
        chunks.append( (g, fulltext[idx:end], idx) )
    return chunks

def detect_section_type(section_text: str, group_name: str) -> str:
    """
    Determine if this section uses list-style formatting or role-based formatting.
    Returns: 'list', 'role', or 'members'
    """
    upper_text = section_text[:300].upper()
    upper_group = group_name.upper()
    
    # Check for list-style sections
    for header in LIST_SECTION_HEADERS:
        if header in upper_text or header in upper_group:
            return 'list'
    
    # Check for member list sections
    for header in MEMBER_SECTION_HEADERS:
        if header in upper_text or header in upper_group:
            return 'members'
    
    return 'role'

def parse_list_style_entries(section_text: str, group_name: str) -> List[Tuple[Optional[str], str, str, str]]:
    """
    Parse entries in format: "Region: Name, Location"
    Returns: [(region, position, name, location), ...]
    """
    entries = []
    
    # Determine position type based on section header
    position = "member"  # default
    if "PRESIDENTS" in group_name.upper():
        position = "president"
    elif "SUPERINTENDENTS" in group_name.upper():
        position = "superintendent"
    elif "SECRETARIES" in group_name.upper():
        position = "secretary"
    elif "MANAGERS" in group_name.upper() or "MANAGER" in group_name.upper():
        position = "manager"
    elif "AGENTS" in group_name.upper() or "AGENT" in group_name.upper():
        position = "agent"
    
    # Find all region: name, location patterns
    for match in REGION_LIST_PATTERN.finditer(section_text):
        region = match.group(1).strip()
        name = match.group(2).strip()
        location = match.group(3).strip()
        
        # Clean up location (remove trailing page numbers, etc.)
        location = re.sub(r'\s+\d{1,3}\s*$', '', location).strip()
        location = re.sub(r'\s*\.\s*$', '', location).strip()
        
        entries.append((region, position, name, location))
    
    return entries

def parse_member_list(section_text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse simple member lists (no explicit positions).
    Returns: [(name, location), ...]
    """
    entries = []
    
    # Split by lines and look for name, location patterns
    lines = section_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        
        # Skip lines that look like headers
        if line.isupper() or line.endswith(':'):
            continue
        
        # Look for pattern: Name, Location
        names = [m.group(1).strip() for m in NAME_REGEX.finditer(line)]
        if names and ',' in line:
            # Get everything after the first name as location
            first_name_match = NAME_REGEX.search(line)
            if first_name_match:
                remaining = line[first_name_match.end():].strip()
                if remaining.startswith(','):
                    location = remaining[1:].strip()
                    # Clean location
                    location = re.sub(r'\s+\d{1,3}\s*$', '', location).strip()
                    location = re.sub(r'\s*\.\s*$', '', location).strip()
                    if location:
                        entries.append((names[0], location))
                    else:
                        entries.append((names[0], None))
    
    return entries

def iter_role_blocks_enhanced(section_text: str, section_type: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Enhanced version that handles different section types.
    Returns: [(position, position_qualifier, content), ...]
    """
    if section_type == 'list':
        # Signal to caller to use list parsing
        return [('__LIST__', None, section_text)]
    
    if section_type == 'members':
        # Signal to caller to use member list parsing
        return [('__MEMBERS__', None, section_text)]
    
    # Original role-based parsing with enhanced qualifier extraction
    blocks = []
    positions = []
    
    for m in ROLE_WITH_QUALIFIER.finditer(section_text):
        role = m.group('role').strip()
        qualifier = m.group('qualifier').strip() if m.group('qualifier') else None
        positions.append((role, qualifier, m.start(), m.end()))
    
    for i, (role, qualifier, s, e) in enumerate(positions):
        end = positions[i+1][2] if i+1 < len(positions) else len(section_text)
        rest = section_text[e:end].strip()
        blocks.append((role, qualifier, rest))
    
    return blocks

def split_names_and_location(rest: str) -> Tuple[List[str], Optional[str]]:
    names = [m.group(1).strip().replace("  ", " ") for m in NAME_REGEX.finditer(rest)]
    last = None
    for m in NAME_REGEX.finditer(rest):
        last = m
    if last:
        after = rest[last.end():].strip(" ,.;")
        location = after if after else None
    else:
        if "," in rest:
            nm, loc = rest.split(",", 1)
            names = [nm.strip()]
            location = loc.strip(" ,.;")
        else:
            names = [rest.strip()]
            location = None
    if location and re.search(r'\b\d{1,2}\s*$', location):
        location = re.sub(r'\b\d{1,2}\s*$', '', location).strip()
    return names, location

def determine_gender(fullname: str) -> Optional[str]:
    """Determine gender based on title prefixes (Mrs., Miss, Mr., Dr.)"""
    name_lower = fullname.lower()
    
    if name_lower.startswith("mrs.") or name_lower.startswith("miss"):
        return "Female"
    
    if name_lower.startswith("mr."):
        return "Male"
    
    return None

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
        
        for group, section, section_start_pos in sections:
            if group == "unlabeled":
                continue
            
            # Detect which conference this section belongs to
            conf_info = detect_conference(text, section_start_pos)
            conference_name = conf_info[0] if conf_info else "General"
            
            # Determine section type
            section_type = detect_section_type(section, group)
            
            # Get role blocks with enhanced parsing
            role_blocks = iter_role_blocks_enhanced(section, section_type)
            
            for position, qualifier, content in role_blocks:
                # Handle list-style sections
                if position == '__LIST__':
                    list_entries = parse_list_style_entries(content, group)
                    
                    for region, pos, fullname, location in list_entries:
                        prefix, lastname = parse_prefix_lastname(fullname)
                        gender = determine_gender(fullname)
                        
                        rows.append(Row(
                            conference=conference_name,
                            region=region,
                            institution_name=None,
                            organization=None,
                            group=group,
                            position_information=region,
                            position=pos,
                            prefix=prefix,
                            name=fullname,
                            lastname=lastname,
                            suffix=None,
                            gender=gender,
                            location=location,
                            yearbook_year=year,
                            page=pageno,
                            source_pdf=os.path.basename(pdf_path)
                        ))
                
                # Handle member list sections
                elif position == '__MEMBERS__':
                    member_entries = parse_member_list(content)
                    
                    for fullname, location in member_entries:
                        prefix, lastname = parse_prefix_lastname(fullname)
                        gender = determine_gender(fullname)
                        
                        rows.append(Row(
                            conference=conference_name,
                            region=None,
                            institution_name=None,
                            organization=None,
                            group=group,
                            position_information=None,
                            position="member",
                            prefix=prefix,
                            name=fullname,
                            lastname=lastname,
                            suffix=None,
                            gender=gender,
                            location=location,
                            yearbook_year=year,
                            page=pageno,
                            source_pdf=os.path.basename(pdf_path)
                        ))
                
                # Handle traditional role-based sections
                else:
                    names, location = split_names_and_location(content)
                    
                    pos = position.strip().lower()
                    position_information = qualifier
                    
                    for fullname in names:
                        prefix, lastname = parse_prefix_lastname(fullname)
                        gender = determine_gender(fullname)
                        
                        rows.append(Row(
                            conference=conference_name,
                            region=None,
                            institution_name=None,
                            organization=None,
                            group=group,
                            position_information=position_information,
                            position=pos,
                            prefix=prefix,
                            name=fullname,
                            lastname=lastname,
                            suffix=None,
                            gender=gender,
                            location=location,
                            yearbook_year=year,
                            page=pageno,
                            source_pdf=os.path.basename(pdf_path)
                        ))
    
    # Deduplicate
    seen = set()
    uniq = []
    for r in rows:
        key = (r.yearbook_year, r.page, r.conference, r.region, r.group, r.position, r.name, r.location)
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(r)
    return uniq

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
            write_header = True

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
    ap.add_argument("inputs", nargs="*", help="PDF files or glob patterns (default: all YB19*.pdf in current directory)")
    ap.add_argument("-o","--out", default="yearbook_officers.csv", help="Output CSV path")
    ap.add_argument("--append", action="store_true",
                help="Append to existing CSV instead of overwriting")

    args = ap.parse_args()

    # If no inputs specified, automatically find all YB19*.pdf files
    if not args.inputs:
        current_dir = os.getcwd()
        pattern = os.path.join(current_dir, "YB19*.pdf")
        files = glob.glob(pattern)
        if not files:
            print(f"No YB19*.pdf files found in {current_dir}")
            print("Please specify input files or ensure YB19*.pdf files exist in the current directory.")
            return
        print(f"Found {len(files)} yearbook PDF(s) to process:")
        for f in sorted(files):
            print(f"  - {os.path.basename(f)}")
    else:
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
            print(f"Processing {os.path.basename(f)}...")
            rows = extract_from_pdf(f)
            all_rows.extend(rows)
            print(f"  Extracted {len(rows)} officer records")
        except Exception as e:
            print(f"[WARN] Failed on {f}: {e}")

    write_csv(all_rows, args.out, append=args.append)
    print(f"\nWrote {len(all_rows)} total rows to {args.out} ({'append' if args.append else 'overwrite'})")
    
    # Summary statistics
    from collections import Counter
    print(f"\nSummary by year:")
    year_counts = Counter(r.yearbook_year for r in all_rows)
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]} records")
    
    print(f"\nSummary by conference:")
    conf_counts = Counter(r.conference for r in all_rows)
    for conf in sorted(conf_counts.keys()):
        print(f"  {conf}: {conf_counts[conf]} records")

if __name__ == "__main__":
    main()
