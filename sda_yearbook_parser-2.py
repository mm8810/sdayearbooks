
from PyPDF2 import PdfReader
import re, os, csv, unicodedata
from typing import List, Dict

STATE_ALIASES = {
    "Mich.":"MI","Mass.":"MA","Cal.":"CA","Cal":"CA","Wis.":"WI","Vt.":"VT","Vt":"VT",
    "Me.":"ME","Me":"ME","Ill.":"IL","Ind.":"IN","Iowa":"IA","Iowa.":"IA","Iowa,":"IA",
    "Kan.":"KS","Ky.":"KY","Mo.":"MO","Neb.":"NE","Nebr.":"NE","N. Y.":"NY","N. Y":"NY","N.Y.":"NY",
    "N. H.":"NH","R. I.":"RI","Minn.":"MN","Col.":"CO","Col":"CO","Colo.":"CO","Ohio":"OH","0.":"OH","O.":"OH",
    "Oreg.":"OR","Oregon":"OR","W. T.":"WA","W. T":"WA","Wash.":"WA","Tenn.":"TN","Tex.":"TX","Ala.":"AL",
    "D. C.":"DC","D. T.":"Dakota Terr.","P. Q.":"Quebec","Ont.":"Ontario","Que.":"Quebec","Canada":"Canada",
    "Prussia":"Prussia","Norway":"Norway","England":"England","Scotland":"Scotland","Sweden":"Sweden",
    "Denmark":"Denmark","Switzerland":"Switzerland","Germany":"Germany","Prussia.":"Prussia",
    "Dakota":"Dakota","Iowa;":"IA","Iowa,":"IA","Mo":"MO"
}

PREFIX_GENDER = {
    "Mrs.":"female","Miss":"female","Ms.":"female","Sister":"female","Sis.":"female",
    "Mr.":"male","Elder":"male","Eld.":"male","Br.":"male","Brother":"male",
    "Dr.":"unknown","Prof.":"unknown","Rev.":"unknown","Madam":"female","Mme.":"female","Sir":"male"
}
PREFIXES = sorted(PREFIX_GENDER.keys(), key=len, reverse=True)
SUFFIXES = ["M. D.","M.D.","MD","Jr.","Sr.","Esq.","Esq","Ph.D.","D.D."]

ALLOWED_ROLE_PREFIXES = tuple([
    "President","Vice","Secretary","Treasurer","Assistant","Associate","Auditor","Executive",
    "Directors","Board","Committee","Agents","Superintendent","Superintendents","Field",
    "Publishing","Medical","Educational","Missionary","Manager","Managers","Supt.","Negro",
    "Bureau","Home Missionary","Sabbath","Missionary Volunteer","Religious Liberty"
])
DENY_ROLES = {"Territory","Cable Address","Telegraphic Address","Express and Freight Address",
              "Postal Address","Organized","Telegraphic","Express","Freight","Address",
              "Directory","Officers","Elective Members","General Members","Headquarters",
              "Office Address","Office","General"}

ROLE_SEGMENT_PATTERN = re.compile(r'([A-Za-z][A-Za-z \-\.\(\)&/]*?)\s*:\s*(.*?)(?=(?: [A-Z][A-Za-z \-\.\(\)&/]{0,40}\s*:\s)|$)')

STATE_HEADINGS = {h.upper() for h in [
    "California","Canada","Colorado","Dakota Territory","Denmark","Illinois","Indiana","Iowa","Kansas","Kentucky",
    "Maine","Michigan","Minnesota","Missouri","Nebraska","New England","New York","North Pacific","Ohio",
    "Pennsylvania","Sweden","Tennessee","Texas","Upper Columbia","Vermont","Wisconsin"
]}

def extract_pages_text(path: str) -> List[str]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages

def clean_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("—","-").replace("•","")
    s = re.sub(r'\s+',' ', s).strip()
    return s

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or any(k in s for k in ["Page", "PREFACE"]): 
        return False
    letters = re.sub(r'[^A-Za-z]','',s)
    if not letters: 
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1,len(letters))
    if upper_ratio > 0.9 and len(s) >= 4:
        return True
    if "DIRECTORY" in s.upper():
        return True
    return False

def is_personnel_context(major: str) -> bool:
    if not major:
        return False
    up = major.upper()
    if any(bad in up for bad in ["PROCEEDINGS","STATISTICS","CONSTITUTION","CATALOGUE","CALENDAR","POSTAL RATES"]):
        return False
    return any(tok in up for tok in ["DIRECTORY","OFFICERS","DEPARTMENT","ASSOCIATION","CONFERENCE","PUBLISHING","SANITARIUM","SOCIETY"])

def detect_year_from_path(path: str):
    m = re.search(r'YB(\d{4})\.pdf$', os.path.basename(path))
    return int(m.group(1)) if m else None

def split_role_entries(content: str):
    return [p.strip(" ;") for p in re.split(r';\s*', content) if p.strip(" ;")]

def parse_person_chunk(chunk: str) -> Dict[str, str]:
    original = chunk
    prefix = None
    for pf in PREFIXES:
        if chunk.startswith(pf+" "):
            prefix = pf
            chunk = chunk[len(pf):].strip()
            break
    location_raw = None
    if "," in chunk:
        name_part, loc_part = chunk.split(",", 1)
        name = name_part.strip()
        location_raw = loc_part.strip().strip(",")
    else:
        name = chunk.strip()
    suffix_found = None
    for suf in SUFFIXES:
        if name.endswith(" "+suf):
            suffix_found = suf
            name = name[:-(len(suf)+1)].strip()
    gender = PREFIX_GENDER.get(prefix, "unknown")
    return {
        "name_prefix": prefix,
        "full_name": name,
        "name_suffix": suffix_found,
        "gender_inferred": gender,
        "location_raw": location_raw,
        "original_chunk": original
    }

def parse_location_fields(location_raw: str):
    if not location_raw:
        return {"city": None, "state_or_region": None, "country": None}
    parts = [p.strip() for p in location_raw.split(",")]
    city = parts[0] if parts else None
    region = None
    country = None
    if len(parts) >= 2:
        token = parts[-1]
        mapped = STATE_ALIASES.get(token, token)
        if mapped in {"England","Scotland","Prussia","Norway","Denmark","Sweden","Switzerland","Germany","Canada","Quebec","Ontario","Dakota Terr.","Dakota"}:
            country = mapped
            if len(parts) >= 2:
                region = parts[-2]
        else:
            region = mapped
            if mapped in {"MI","MA","CA","WI","VT","ME","IL","IN","IA","KS","KY","MO","NE","NY","NH","RI","MN","CO","OH","OR","WA","TN","TX","AL","DC"}:
                country = "USA"
            elif mapped in {"Ontario","Quebec","Canada"}:
                country = "Canada"
    return {"city": city, "state_or_region": region, "country": country}

def parse_pdf(path: str):
    year = detect_year_from_path(path)
    pages = extract_pages_text(path)
    records = []
    major = None
    suborg = None
    buffer_role = None
    buffer_text = None
    for page in pages:
        lines = [clean_line(ln) for ln in (page.split("\\n") if page else [])]
        for line in lines:
            if not line:
                continue
            if is_heading(line):
                text = line.strip(" .")
                up = text.upper()
                if up in STATE_HEADINGS:
                    suborg = text.title()
                else:
                    if "CALENDAR" not in up:
                        major = text.title()
                        if "STATE CONFERENCE DIRECTORIES" not in up:
                            suborg = None
                continue
            segments = list(ROLE_SEGMENT_PATTERN.finditer(line))
            if segments:
                for seg in segments:
                    if buffer_role is not None and is_personnel_context(major):
                        for chunk in split_role_entries(buffer_text.strip()):
                            person = parse_person_chunk(chunk)
                            loc = parse_location_fields(person["location_raw"])
                            records.append({
                                "source_file": os.path.basename(path),
                                "source_year": year,
                                "section": major,
                                "organization": suborg if suborg else major,
                                "role_title": buffer_role.strip(),
                                **person, **loc
                            })
                    buffer_role = seg.group(1).strip()
                    buffer_text = seg.group(2).strip()
            else:
                if buffer_text is not None:
                    buffer_text += " " + line.strip()
    if buffer_role is not None and buffer_text is not None and is_personnel_context(major):
        for chunk in split_role_entries(buffer_text.strip()):
            person = parse_person_chunk(chunk)
            loc = parse_location_fields(person["location_raw"])
            records.append({
                "source_file": os.path.basename(path),
                "source_year": year,
                "section": major,
                "organization": suborg if suborg else major,
                "role_title": buffer_role.strip(),
                **person, **loc
            })
    filtered = []
    for r in records:
        role = r["role_title"]
        if any(role.startswith(p) for p in ALLOWED_ROLE_PREFIXES) and role not in DENY_ROLES:
            if any(c.isalpha() for c in r["full_name"]):
                filtered.append(r)
    return filtered

def main():
    inputs = ["/mnt/data/YB1883.pdf", "/mnt/data/YB1921.pdf"]
    out_path = "/mnt/data/yearbook_people.csv"
    fieldnames = [
        "source_file","source_year","section","organization","role_title",
        "name_prefix","full_name","name_suffix","gender_inferred",
        "location_raw","city","state_or_region","country"
    ]
    total = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for path in inputs:
            if not os.path.exists(path):
                print(f"WARNING: file not found: {path}")
                continue
            rows = parse_pdf(path)
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
            print(f"Parsed {len(rows)} people from {os.path.basename(path)}")
            total += len(rows)
    print(f"Wrote {total} rows to {out_path}")

if __name__ == "__main__":
    main()
