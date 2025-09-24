#!/usr/bin/env python3
"""

Usage:
  python sda_yearbooks_parser_ollama.py YB1883.pdf -o YB1883.csv
  python sda_yearbooks_parser_ollama.py YB1883.pdf YB1884.pdf -o out.csv
  python sda_yearbooks_parser_ollama.py YB1921.pdf -o 1921.csv --model llama3.1
  python sda_yearbooks_parser_ollama.py YB1883.pdf -o out.csv --max-pages 10

Requirements (install as needed):
  pip install pdfplumber pydantic ollama tqdm

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Any, Dict

import pdfplumber  # type: ignore
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm
import ollama

# ---------------------------
# Pydantic models / schema
# ---------------------------

class YearbookRow(BaseModel):
    directory: Optional[str] = None
    conference: Optional[str] = None
    region: Optional[str] = None
    # schema**

    institution_name: Optional[str] = None  # normalized from "institution-name"
    organization: Optional[str] = None
    group: Optional[str] = None
    position_information: Optional[str] = None
    position: Optional[str] = None
    prefix: Optional[str] = None
    name: Optional[str] = None
    lastname: Optional[str] = None
    suffix: Optional[str] = None
    location: Optional[str] = None
    yearbook_year: Optional[int] = None
    page: Optional[int] = None
    raw_line: Optional[str] = None

    model_config = ConfigDict(extra='ignore')

    @field_validator('yearbook_year', mode='before')
    @classmethod
    def clamp_year(cls, v):
        if v is None:
            return v
        if 1800 <= v <= 2100:
            return v
        return None


class PageExtraction(BaseModel):
    """Model for page-level extraction: list of YearbookRow entries."""
    rows: List[YearbookRow]


# ---------------------------
# Helpers
# ---------------------------

def extract_year_from_filename(path: Path) -> Optional[int]:
    m = re.search(r'(18|19|20)\d{2}', path.name)
    return int(m.group(0)) if m else None


def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def default_headers() -> List[str]:
    # Strict output headers the user asked for; we map our field names accordingly.
    return [
        "directory",
        "conference",
        "region",
        "institution-name",
        "organization",
        "group",
        "position-information",
        "position",
        "prefix",
        "name",
        "lastname",
        "suffix",
        "location",
        "yearbook-year",
        "page",
        "raw_line",
    ]


def to_csv_row(y: YearbookRow) -> List[Any]:
    # Map pydantic fields -> requested CSV header names
    return [
        y.directory or None,
        y.conference or None,
        y.region or None,
        y.institution_name or None,  # "institution-name"
        y.organization or None,
        y.group or None,
        y.position_information or None,  # "position-information"
        y.position or None,
        y.prefix or None,
        y.name or None,
        y.lastname or None,
        y.suffix or None,
        y.location or None,
        y.yearbook_year,
        y.page,
        y.raw_line or None,
    ]


def is_probably_heading(line: str) -> bool:
    """Heuristic to mark likely headings/directories (big caps, short lines)."""
    clean = re.sub(r'[^A-Za-z0-9 &\-\.,]', ' ', line).strip()
    if not clean:
        return False
    # uppercase bias + shortish line
    caps_ratio = sum(1 for c in clean if c.isupper()) / max(1, sum(1 for c in clean if c.isalpha()))
    return caps_ratio > 0.7 and len(clean) <= 60


def prune_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def page_text_lines(page) -> List[str]:
    text = page.extract_text() or ""
    lines = [prune_whitespace(ln) for ln in text.splitlines()]
    # discard empties
    return [ln for ln in lines if ln]


def build_system_instruction() -> str:
    return (
        "You are a meticulous archivist and data normalizer extracting structured records "
        "from historical Seventh-day Adventist Yearbooks. You must return STRICT JSON "
        "matching the given schema. Only include plausible person/institution roster rows; "
        "skip editorial, prefaces, poetry, constitutions, calendars, postal tables, etc."
    )


def build_user_prompt_for_page(context: Dict[str, Optional[str]], year: Optional[int], page_index: int, raw_lines: List[str]) -> str:
    """
    Provide the model with page-level context (recent headings) + all lines.
    Ask it to infer directory/conference/region and emit only valid 'rows'.
    """
    ctx_str = json.dumps(context, ensure_ascii=False)
    joined = "\n".join(raw_lines[:400])  # safety cap

    instructions = f"""
Extract roster-like ENTRIES from this page of an SDA Yearbook as an array of objects named "rows".
Infer missing context (directory, conference, region, organization, group) from headings when possible.

Rules:
- Only output entries (people/positions/institutions) that belong to directories/rosters/lists.
- If a line is a heading, use it to update context for subsequent lines, but DO NOT emit a row for the heading.
- Parse names into prefix / name / lastname / suffix when possible (e.g., "Eld. J. N. Andrews, D.D.S." -> prefix="Eld.", name="J. N.", lastname="Andrews", suffix="D.D.S.").
- Extract locations like city/state/country when present (e.g., "Battle Creek, Mich." -> location="Battle Creek, Mich.").
- If you can't confidently parse a line, skip it (do NOT guess wildly).
- For institutions/boards/directories, set "institution-name" and/or "organization" and only add name fields when a person appears.
- Keep text concise; don't invent data.
- Always include "raw_line" with the source line content.
- Always include "page" (1-based page number).

Context (carry forward into your inferences):
{ctx_str}

Page meta:
- yearbook_year: {year if year is not None else "unknown"}
- page: {page_index + 1}

Page text lines:
{joined}
"""
    return instructions.strip()


def call_ollama_page(model: str, schema: Dict[str, Any], system_instruction: str, prompt: str) -> Optional[PageExtraction]:
    ollama.pull(model)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            format=schema,
            options={"temperature": 0},
        )
        content = response["message"]["content"]
        # Ensure content is a JSON string
        if isinstance(content, dict):
            content = json.dumps(content)
        return PageExtraction.model_validate_json(content)
    except Exception as e:
        # Allow the caller to decide whether to retry or skip
        sys.stderr.write(f" {e}\n")
        return None
        


def ensure_cache_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class Cache:
    dir: Path

    def load(self, key: str) -> Optional[dict]:
        p = self.dir / f"{key}.json"
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save(self, key: str, obj: dict) -> None:
        p = self.dir / f"{key}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_context_from_lines(lines: List[str]) -> Dict[str, Optional[str]]:
    """
    Very light heuristics to seed context before first call per page.
    We lean on the LLM to refine during extraction.
    """
    directory = conference = region = organization = group = None
    # Simple patterns
    for ln in lines[:15]:
        u = ln.upper()
        if "DIRECTORY" in u and not directory:
            # e.g., "GENERAL CONFERENCE DIRECTORY", "STATE CONFERENCE DIRECTORIES"
            directory = prune_whitespace(ln.title())
        if ("CONFERENCE" in u and not conference and "DIRECTORY" in u) or (re.search(r"\bConference\b", ln) and not conference):
            conference = prune_whitespace(ln)
        if re.search(r"\bDistrict\b|\bRegion\b|\bNo\.\s*\d+\b", ln, re.I) and not region:
            region = prune_whitespace(ln)
        if re.search(r"\bASSOCIATION\b|\bSOCIETY\b|\bINSTITUTE\b|\bPUBLISHING\b", u) and not organization:
            organization = prune_whitespace(ln.title())
        if re.search(r"\bCOMMITTEE\b|\bBOARD\b|\bMINISTERS\b", u) and not group:
            group = prune_whitespace(ln.title())
    return {
        "directory": directory,
        "conference": conference,
        "region": region,
        "organization": organization,
        "group": group,
    }


def write_csv(out_path: Path, rows: List[YearbookRow]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(default_headers())
        for r in rows:
            writer.writerow(to_csv_row(r))


def main():
    parser = argparse.ArgumentParser(description="Extract structured rows from SDA Yearbooks via Ollama structured outputs.")
    parser.add_argument("pdfs", nargs="+", help="One or more PDF paths (Yearbooks).")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path.")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name (default: llama3.1).")
    parser.add_argument("--cache-dir", default=".ollama_cache_yearbooks", help="Directory to cache per-page JSON results.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional limit: only process up to N pages per PDF (for testing).")
    parser.add_argument("--start-page", type=int, default=1, help="1-based page index to start (default: 1).")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    cache = Cache(ensure_cache_dir(Path(args.cache_dir)))

    all_rows: List[YearbookRow] = []
    system_instruction = build_system_instruction()
    schema = PageExtraction.model_json_schema()

    for pdf_path_str in args.pdfs:
        pdf_path = Path(pdf_path_str).resolve()
        if not pdf_path.exists():
            print(f"[WARN] Missing file: {pdf_path}")
            continue

        year = extract_year_from_filename(pdf_path)

        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            start_idx = max(0, args.start_page - 1)
            end_idx = total_pages if args.max_pages is None else min(total_pages, start_idx + args.max_pages)
            rng = range(start_idx, end_idx)

            pbar = tqdm(rng, desc=f"Processing {pdf_path.name}", unit="page")
            for i in pbar:
                page = pdf.pages[i]
                lines = page_text_lines(page)
                if not lines:
                    continue

                # Build context heuristics
                init_context = infer_context_from_lines(lines[:25])

                cache_key = md5(f"{pdf_path.name}:{i}:{init_context}")
                cached = cache.load(cache_key)
                if cached is not None:
                    try:
                        pe = PageExtraction.model_validate(cached)
                        for r in pe.rows:
                            # Fill page + year if missing
                            if r.page is None:
                                r.page = i + 1
                            if r.yearbook_year is None:
                                r.yearbook_year = year
                        all_rows.extend(pe.rows)
                        continue
                    except Exception:
                        pass  # fall through to re-call

                user_prompt = build_user_prompt_for_page(init_context, year, i, lines)
                pe = call_ollama_page(args.model, schema, system_instruction, user_prompt)
                if pe is None:
                    # Save a negative cache entry to skip on next run, but don't break.
                    cache.save(cache_key, {"rows": []})
                    continue

                # Post-process rows
                for r in pe.rows:
                    # normalize a few fields
                    if r.institution_name is None and r.organization is None:
                        # Try a quick institution cue from capitals + keywords
                        pass
                    if r.yearbook_year is None:
                        r.yearbook_year = year
                    if r.page is None:
                        r.page = i + 1
                    # Title-case institution names
                    if r.institution_name:
                        r.institution_name = r.institution_name.strip()
                    # Normalize whitespace in location
                    if r.location:
                        r.location = prune_whitespace(r.location)

                # Cache
                cache.save(cache_key, pe.model_dump())
                all_rows.extend(pe.rows)

    # Deduplicate rows by (year, page, raw_line) to avoid duplicates on retries
    seen = set()
    deduped: List[YearbookRow] = []
    for r in all_rows:
        key = (r.yearbook_year, r.page, (r.raw_line or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    write_csv(out_path, deduped)
    print(f"[DONE] Wrote {len(deduped)} rows to {out_path}")
    print("Tip: re-run with --start-page and/or --max-pages to iterate quickly.")


if __name__ == "__main__":
    main()
