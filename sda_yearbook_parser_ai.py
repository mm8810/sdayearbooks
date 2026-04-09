import argparse, os, re, json, csv, time, math, sys, tempfile, hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

# Read directly from the PDF when no prefiltered text bundle is supplied.
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

from rate_limiter import TokenBucket, estimate_tokens_from_text

# Import providers lazily so the module still loads if only one SDK is installed.
OpenAI = None
anthropic = None
try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
except Exception:
    pass
try:
    import anthropic as _anthropic
    anthropic = _anthropic
except Exception:
    pass

@dataclass
class Row:
    yearbook_year: Optional[int] = None
    page: Optional[int] = None
    name: Optional[str] = None
    last_name: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    gender: Optional[str] = None
    position: Optional[str] = None
    position_information: Optional[str] = None
    organization: Optional[str] = None
    group: Optional[str] = None
    conference: Optional[str] = None
    institution_name: Optional[str] = None
    location: Optional[str] = None
    region: Optional[str] = None

class ProviderError(RuntimeError): ...
class ProviderBase:
    def __init__(self, model: str, limiter: TokenBucket):
        self.model = model
        self.limiter = limiter
    def complete(self, system_prompt: str, user_text: str, max_output_tokens: int = 12000) -> str:
        raise NotImplementedError


class OpenAIProvider(ProviderBase):
    def __init__(self, model: str, limiter: TokenBucket):
        super().__init__(model, limiter)
        if OpenAI is None:
            raise ProviderError("openai package not installed. `pip install openai`")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=20), retry=retry_if_exception_type(Exception))
    def complete(self, system_prompt: str, user_text: str, max_output_tokens: int = 12000) -> str:
        total_est = estimate_tokens_from_text(system_prompt) + estimate_tokens_from_text(user_text) + max_output_tokens
        self.limiter.acquire(total_est)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_text}
                ],
                temperature=0,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            raise
        try:
            return resp.choices[0].message.content if resp and resp.choices else ""
        except Exception:
            return ""

class AnthropicProvider(ProviderBase):
    def __init__(self, model: str, limiter: TokenBucket):
        super().__init__(model, limiter)
        if anthropic is None:
            raise ProviderError("anthropic package not installed. `pip install anthropic`")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=1, max=20), retry=retry_if_exception_type(Exception))
    def complete(self, system_prompt: str, user_text: str, max_output_tokens: int = 12000) -> str:
        total_est = estimate_tokens_from_text(system_prompt) + estimate_tokens_from_text(user_text) + max_output_tokens
        self.limiter.acquire(total_est)
        try:
            resp = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role":"user","content":user_text}],
                temperature=0,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            raise
        try:
            parts = []
            for block in resp.content or []:
                if getattr(block, "type", "") == "text":
                    parts.append(block.text)
            return "".join(parts).strip()
        except Exception:
            return ""

MODEL_LIMITS = {
    "gpt-5":       {"TPM": 500_000, "RPM": 500},
    "gpt-5-mini":  {"TPM": 500_000, "RPM": 500},
    "gpt-4.1":     {"TPM": 30_000,  "RPM": 500},
    "claude-3-5-sonnet": {"TPM": 500_000, "RPM": 400},
    "claude-3-haiku":    {"TPM": 500_000, "RPM": 400},
}

def pick_limits(model: str):
    return MODEL_LIMITS.get(model, {"TPM": 30_000, "RPM": 60})

SYSTEM_PROMPT = """You are extracting structured directory entries from scanned historical Seventh-day Adventist Yearbooks (1880s–1910s). These contain directories of people, offices, and institutions across conferences, missions, schools, publishing associations, etc.

Output strict JSON Lines, one object per line. Each line represents one person entry (individual name with role information).

Keys (all required; use null for missing):
yearbook_year (int | null)
page (int | null)
name (string)
last_name (string | null)
prefix (string | null)
suffix (string | null)
gender (string | null)
position (string | null)
position_information (string | null)
organization (string | null)
group (string | null)
conference (string) -- REQUIRED, NEVER NULL
institution_name (string | null)
location (string | null)
region (string | null)

Parsing Rules:
Each object = one identifiable person. Exclude section headings, non-person institutional names, tables, or filler text.

Name Parsing:
name: Full name exactly as printed (preserve initials and punctuation).
Split out last_name (best guess from rightmost capitalized surname).
Prefixes (Eld., Mrs., Dr., Miss, Prof., etc.) and suffixes (Jr., Sr., M.D., etc.) go in their fields.
Derive gender heuristically from prefix (Miss, Mrs. → female; Eld., Mr. → male).

Position Parsing:
Capture person's official title or role (e.g., President, Secretary, Treasurer, Director, Committee Member).
If followed by a colon or semicolon, take the text before colon as position; text after colon as position_information (e.g., "Battle Creek, Mich.").

Organizational Hierarchy:
Assign the nearest organization heading (e.g., General Conference Directory, Health Reform Institute Directory, Michigan Conference) to organization.
Assign the next higher heading (e.g., STATE CONFERENCE DIRECTORIES) to group.

Conference Assignment (CRITICAL - ALWAYS REQUIRED):
EVERY person MUST have a conference value. Follow this hierarchy:
1. If under a state/territory conference heading (e.g., "Michigan Conference", "California Conference"), use that state/territory name
2. If under a union conference heading (e.g., "German Union Conference", "Scandinavian Union"), use that union name
3. If under a mission heading (e.g., "South African Mission", "India Mission"), use that mission name
4. If under General Conference with no specific territory, use "General Conference"
5. If unclear but within a regional section, use the regional identifier
6. As last resort for truly ambiguous cases, use "General Conference" as the default

Conference format:
- State conferences: Use state name (e.g., "Michigan", "California", "Tennessee")
- Union conferences: Keep full name (e.g., "German Union", "Scandinavian Union", "North Pacific Union")
- Mission fields: Keep full name (e.g., "South African Mission", "India Mission")
- General body: "General Conference"

Institution and Location:
If a school, sanitarium, or publishing house is named, fill in institution_name.
Extract city, state, or country to location (normalize to "City, State" if both present).
For colonial-era or non-US entities, include broader label (e.g., "South Lancaster, Mass.", "Battle Creek, Mich.", "Christiana, Norway").

Region:
Use broader geopolitical or denominational grouping (e.g., "United States", "Scandinavia", "Africa", "South America", etc.) if discernible from headings.

Contextual Propagation:
When multiple people are listed under one heading, inherit the same conference / organization / location until a new section heading appears.
The conference value from a section heading applies to ALL people in that section until a new conference heading appears.
Reset on new section headings.

Formatting Requirements:
Output one JSON object per line (JSONL).
Strict JSON — double quotes around keys and values.
No trailing commas or markdown formatting.

Examples of valid entries:
{"yearbook_year":1883,"page":7,"name":"Geo. I. Butler","last_name":"Butler","prefix":null,"suffix":null,"gender":"male","position":"President","position_information":null,"organization":"General Conference","group":null,"conference":"General Conference","institution_name":null,"location":"Battle Creek, Mich.","region":"United States"}
{"yearbook_year":1883,"page":12,"name":"S. N. Haskell","last_name":"Haskell","prefix":null,"suffix":null,"gender":"male","position":"President","position_information":null,"organization":"Pacific S.D.A. Publishing Association","group":"Publishing Association Directories","conference":"California","institution_name":null,"location":"South Lancaster, Mass.","region":"United States"}
{"yearbook_year":1904,"page":11,"name":"A. G. Daniells","last_name":"Daniells","prefix":null,"suffix":null,"gender":"male","position":"President","position_information":null,"organization":"General Conference","group":null,"conference":"General Conference","institution_name":null,"location":"Washington, D.C.","region":"United States"}
{"yearbook_year":1904,"page":15,"name":"W. W. Prescott","last_name":"Prescott","prefix":null,"suffix":null,"gender":"male","position":"Second Vice-President","position_information":null,"organization":"General Conference","group":null,"conference":"General Conference","institution_name":null,"location":"Washington, D.C.","region":"United States"}
{"yearbook_year":1904,"page":27,"name":"S. Fulton","last_name":"Fulton","prefix":null,"suffix":null,"gender":"male","position":"President","position_information":null,"organization":"Tennessee Conference","group":"State Conference Directories","conference":"Tennessee","institution_name":null,"location":"Nashville, Tenn.","region":"United States"}"""

def build_user_prompt(page_text: str, page_index: Optional[int], year: Optional[int]) -> str:
    """Build a compact per-page prompt for the model."""
    text = re.sub(r"[ \t]+", " ", page_text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    head = f"YEAR: {year if year is not None else 'unknown'}\nPAGE_INDEX: {page_index if page_index is not None else 'unknown'}"
    return head + "\n--- PAGE TEXT ---\n" + text[:6000]

CACHE_DIR = Path(tempfile.gettempdir()) / "sda_ai_cache"
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(provider: str, model: str, system_prompt: str, user_prompt: str) -> Path:
    h = hashlib.sha256()
    h.update((provider or "").encode())
    h.update((model or "").encode())
    h.update(system_prompt.encode())
    h.update(user_prompt.encode())
    return CACHE_DIR / (h.hexdigest() + ".jsonl")

def cached_complete(provider: str, model: str, call_fn, system_prompt: str, user_prompt: str) -> str:
    key_path = cache_key(provider, model, system_prompt, user_prompt)
    if key_path.exists():
        try:
            return key_path.read_text(encoding="utf-8")
        except Exception:
            pass
    out = call_fn()
    try:
        key_path.write_text(out, encoding="utf-8")
    except Exception:
        pass
    return out

def run(pdf: Path,
        provider_name: str,
        model: str,
        out_csv: Path,
        year: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        max_pages: Optional[int] = None,
        prefiltered_text_path: Optional[Path] = None) -> List[Row]:

    caps = pick_limits(model)
    limiter = TokenBucket(tokens_per_min=caps["TPM"], requests_per_min=caps["RPM"])

    provider_name = (provider_name or "").lower()
    if provider_name == "anthropic":
        provider = AnthropicProvider(model, limiter)
    elif provider_name == "openai":
        provider = OpenAIProvider(model, limiter)
    else:
        raise ProviderError("Unknown provider: choose 'anthropic' or 'openai'")

    # Read either the prefiltered text bundle or the raw PDF pages.
    pages: List[Tuple[int, str]] = []
    if prefiltered_text_path and Path(prefiltered_text_path).exists():
        raw = Path(prefiltered_text_path).read_text(encoding="utf-8", errors="ignore")
        buf = []
        idx = None
        for line in raw.splitlines():
            m = re.match(r"^==== PAGE\s+(\d+)\s+====\s*$", line.strip())
            if m:
                if idx is not None and buf:
                    pages.append((idx, "\n".join(buf).strip()))
                idx = int(m.group(1))
                buf = []
            else:
                buf.append(line)
        if idx is not None and buf:
            pages.append((idx, "\n".join(buf).strip()))
    else:
        if PdfReader is None:
            raise RuntimeError("PyPDF2 not installed and no prefiltered text provided.")
        reader = PdfReader(str(pdf))
        total = len(reader.pages)
        s = start or 0
        e = min(end if end is not None else total, total)
        for i in range(s, e):
            try:
                text = reader.pages[i].extract_text() or ""
            except Exception:
                text = ""
            pages.append((i, text))

    if max_pages is not None:
        pages = pages[:max_pages]

    rows: List[Row] = []

    # Keep a small worker pool so API calls can overlap without overwhelming the limiter.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = int(os.environ.get("AI_MAX_INFLIGHT", "6"))

    def process_one(item: Tuple[int, str]) -> List[Row]:
        i, txt = item
        user_prompt = build_user_prompt(txt, i, year)
        def caller():
            return provider.complete(SYSTEM_PROMPT, user_prompt, max_output_tokens=12000)
        result_text = cached_complete(provider_name, model, caller, SYSTEM_PROMPT, user_prompt)
        out_rows: List[Row] = []
        for line in result_text.splitlines():
            try:
                obj = json.loads(line)
                out_rows.append(Row(**obj))
            except Exception:
                continue
        return out_rows

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_one, item) for item in pages]
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception:
                continue

    fieldnames = [f.name for f in Row.__dataclass_fields__.values()]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--prefiltered-text", type=Path, default=None)
    args = ap.parse_args()

    run(args.pdf, args.provider, args.model, args.out, args.year, args.start, args.end, args.max_pages, args.prefiltered_text)

if __name__ == "__main__":
    main()
