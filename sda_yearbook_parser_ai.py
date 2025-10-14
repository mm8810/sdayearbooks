#!/usr/bin/env python3
import argparse, os, re, json, csv, time, math, sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dateutil.parser import parse as dtparse

# --- PDF extraction ---
try:
    from PyPDF2 import PdfReader
except Exception as e:
    print("Please `pip install PyPDF2`", file=sys.stderr)
    raise

# --- LLM providers ---
# Optional; install whichever you plan to use
try:
    import anthropic  # Claude
except Exception:
    anthropic = None

try:
    from openai import OpenAI  # OpenAI
except Exception:
    OpenAI = None

# --- Validation ---
from pydantic import BaseModel, Field, ValidationError

# -----------------------------
# Schema you requested (column order preserved in CSV)
# -----------------------------
CSV_COLUMNS = [
    "Conference",
    "Gender",
    "Group",
    "Institution-Name",
    "Last-Name",
    "Location",
    "Name",
    "Organization",
    "Page",
    "Position",
    "Position-Information",
    "Prefix",
    "Region",
    "Suffix",
    "Yearbook",
]

class RowModel(BaseModel):
    Conference: Optional[str] = None
    Gender: Optional[str] = None
    Group: Optional[str] = None
    Institution_Name: Optional[str] = Field(None, alias="Institution-Name")
    Last_Name: Optional[str] = Field(..., alias="Last-Name")
    Location: Optional[str] = None
    Name: Optional[str] = Field(..., alias="Name")
    Organization: Optional[str] = None
    Page: int
    Position: Optional[str] = None
    Position_Information: Optional[str] = Field(None, alias="Position-Information")
    Prefix: Optional[str] = None
    Region: Optional[str] = None
    Suffix: Optional[str] = None
    Yearbook: int

    class Config:
        populate_by_name = True


# -----------------------------
# Prompt Template
# -----------------------------
PROMPT_TEMPLATE = """You are a meticulous historian-assistant extracting structured records from a historical yearbook page.

Return ONLY a JSON object with this exact shape and keys:

{{
  "rows": [
    {{
      "Conference": null | string,
      "Gender": null | "Male" | "Female" | "Unknown",
      "Group": null | string,
      "Institution-Name": null | string,
      "Last-Name": string,                # REQUIRED
      "Location": null | string,
      "Name": string,                     # REQUIRED, as printed (e.g., "J. H. Kellogg")
      "Organization": null | string,
      "Page": INTEGER,                    # REQUIRED
      "Position": null | string,          # e.g., President, Secretary, Treasurer, etc.
      "Position-Information": null | string, # free-form details after the title
      "Prefix": null | string,            # e.g., Eld., Dr., Mrs., Mr., Miss
      "Region": null | string,
      "Suffix": null | string,            # e.g., Jr., Sr., M.D., Ph.D., Esq.
      "Yearbook": INTEGER                 # REQUIRED (supplied in system context)
    }},
    ...
  ]
}}

Extraction rules:
- Parse *names and titles/roles* mentioned on the page. Ignore doctrinal text or narrative paragraphs.
- If a line lists an *organization directory* (e.g., “GENERAL SABBATH-SCHOOL ASSOCIATION DIRECTORY”), set Organization to that text; Position to the role (President, etc).
- If a *state conference* or *association* is indicated (e.g., “MAINE.”, “STATE CONFERENCE DIRECTORIES”), set Region (state/territory/country) and/or Conference (named conference). Use Group for sub-bodies (e.g., “Camp-Meeting Committee”).
- Prefix examples: Eld., Dr., Mrs., Mr., Miss; Suffix examples: Jr., Sr., M.D.
- Derive Gender conservatively from honorifics (e.g., Mrs./Miss → Female; Eld./Mr. → Male; otherwise “Unknown”).
- Last-Name is the surname (strip punctuation). “Name” is the full printed name including initials.
- Position-Information: anything immediately following the title that carries details (e.g., addresses).
- Location: city/state/country as printed on the line, if present.
- Keep diacritics; don’t modernize spellings.
- If the page has no valid entries, return {{"rows":[]}}.

Return ONLY JSON. No markdown, no commentary.

YEARBOOK YEAR: {year}
PAGE NUMBER: {page}

PAGE TEXT:
---
{page_text}
---
"""

# -----------------------------
# Provider Adapters
# -----------------------------
class ProviderError(Exception):
    pass

class ProviderBase:
    def __init__(self, model: str):
        self.model = model

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

class AnthropicProvider(ProviderBase):
    def __init__(self, model: str):
        super().__init__(model)
        if anthropic is None:
            raise ProviderError("anthropic package not installed. `pip install anthropic`")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.0, min=1, max=30),
           retry=retry_if_exception_type(Exception))
    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Anthropic returns a list of content blocks; join text parts
        parts = []
        for block in msg.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts).strip()

class OpenAIProvider(ProviderBase):
    def __init__(self, model: str):
        super().__init__(model)
        if OpenAI is None:
            raise ProviderError("openai package not installed. `pip install openai`")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1.0, min=1, max=30),
           retry=retry_if_exception_type(Exception))
    def complete(self, prompt: str) -> str:
        # Use simple text completion; ask for raw JSON
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()


# -----------------------------
# Utilities
# -----------------------------
def infer_year_from_name(pdf_path: Path) -> Optional[int]:
    m = re.search(r'(18|19|20)\d{2}', pdf_path.stem)
    return int(m.group(0)) if m else None

def clean_text(t: str) -> str:
    # Normalize spaces and common hyphenation breaks
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)  # collapse huge gaps
    # Unwrap common hyphen-breaks like "Associa-\ntion"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    return t.strip()

def extract_pdf_text_per_page(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(clean_text(txt))
    return pages

def json_only(s: str) -> str:
    # Try to isolate a top-level JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

def coerce_rows_to_schema(rows: List[Dict[str, Any]], year: int, page: int) -> List[Dict[str, Any]]:
    fixed = []
    for r in rows:
        # Normalize keys the model might return
        r = {k.replace("_", "-"): v for k, v in r.items()}
        # Force requireds and types
        r.setdefault("Yearbook", year)
        r.setdefault("Page", page)
        try:
            obj = RowModel.model_validate(r)
        except ValidationError:
            # attempt soft fixes for common fields
            r["Page"] = page
            r["Yearbook"] = year
            if not r.get("Last-Name") and r.get("Name"):
                # naive guess: last token (strip punctuation)
                lname = re.sub(r"[^\w’'-]", "", r["Name"].split()[-1])
                r["Last-Name"] = lname
            # retry validation once
            obj = RowModel.model_validate(r)
        # Convert back to your exact column names (aliases preserved)
        out = obj.model_dump(by_alias=True)
        # Ensure all CSV columns are present
        for c in CSV_COLUMNS:
            out.setdefault(c, None)
        fixed.append(out)
    return fixed


# -----------------------------
# Core runner
# -----------------------------
def run(pdf: Path, provider_name: str, model: str, out_csv: Path, year: Optional[int], start: int, end: Optional[int], max_pages: Optional[int]):
    # Provider
    if provider_name.lower() == "anthropic":
        provider = AnthropicProvider(model)
    elif provider_name.lower() == "openai":
        provider = OpenAIProvider(model)
    else:
        raise ProviderError("Unknown provider: choose 'anthropic' or 'openai'")

    # Pages
    pages = extract_pdf_text_per_page(pdf)
    if end is None:
        end = len(pages)
    page_indices = list(range(max(1, start), min(end, len(pages)) + 1))
    if max_pages:
        page_indices = page_indices[:max_pages]

    # Year
    inferred = infer_year_from_name(pdf)
    yearbook_year = year or inferred
    if not yearbook_year:
        raise ValueError("Could not infer year from filename. Pass --year explicitly.")

    all_rows: List[Dict[str, Any]] = []

    for page_num in page_indices:
        text = pages[page_num - 1]
        if not text.strip():
            continue  # skip blank

        prompt = PROMPT_TEMPLATE.format(
            year=yearbook_year,
            page=page_num,
            page_text=text[:12000]  # keep context reasonably sized
        )

        try:
            raw = provider.complete(prompt)
        except Exception as e:
            print(f"[warn] provider error on page {page_num}: {e}", file=sys.stderr)
            continue

        raw_json = json_only(raw)

        # Try parse; if malformed, ask the model once to repair
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(raw_json)
        except Exception:
            # Repair prompt
            repair_prompt = f"""The following is malformed JSON. Return a corrected JSON with the same keys and structure, and nothing else:"""

            try:
                repaired = provider.complete(repair_prompt)
                parsed = json.loads(json_only(repaired))
            except Exception as e:
                print(f"[warn] JSON repair failed on page {page_num}: {e}", file=sys.stderr)
                continue

        rows = parsed.get("rows", [])
        if not isinstance(rows, list):
            print(f"[warn] Page {page_num}: 'rows' is not a list; skipping.", file=sys.stderr)
            continue

        fixed_rows = coerce_rows_to_schema(rows, yearbook_year, page_num)
        all_rows.extend(fixed_rows)

        # Polite pacing to reduce rate limiting
        time.sleep(0.3)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in CSV_COLUMNS})

    print(f"Wrote {len(all_rows)} rows → {out_csv}")


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Extract structured rows from SDA yearbooks using Claude or OpenAI.")
    p.add_argument("pdf", type=Path, help="Path to the yearbook PDF")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="LLM provider")
    p.add_argument("--model", default="claude-3-5-sonnet-20240620", help="Model name (e.g., claude-3-5-sonnet-20240620 or gpt-4o-mini)")
    p.add_argument("--out", type=Path, default=Path("yearbook_rows.csv"), help="Output CSV path")
    p.add_argument("--year", type=int, help="Yearbook year (overrides filename inference)")
    p.add_argument("--start", type=int, default=1, help="Start page (1-based)")
    p.add_argument("--end", type=int, help="End page (inclusive, 1-based)")
    p.add_argument("--max-pages", type=int, help="Process at most N pages (from the start bound)")
    args = p.parse_args()

    run(
        pdf=args.pdf,
        provider_name=args.provider,
        model=args.model,
        out_csv=args.out,
        year=args.year,
        start=args.start,
        end=args.end,
        max_pages=args.max_pages
    )

if __name__ == "__main__":
    main()
