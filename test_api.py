# test_api.py
"""
Claude sanity test with model-ID fallback.
Run:  python test_api.py
"""

import os, json
from anthropic import Anthropic, NotFoundError
from dotenv import load_dotenv

# Load .env
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing. Put it in .env next to this file.")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Try these in order; the first that works will be used.
MODEL_CANDIDATES = [
    # 3.5 series
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    # 3.x series
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
]

SYSTEM = (
    "You are a strict data extractor. "
    "Return ONLY a JSON array of objects with keys: conference, position, name."
)
USER = """
Extract officers from the text below.
Text:
Atlantic Union Conference - Secretary: J. Smith; Treasurer: M. Doe.
Northern New England Conference - President: A. Brown; Secretary: L. Green.
Return a JSON array with conference, position, and name.
"""

def try_model(model_id: str):
    resp = client.messages.create(
        model=model_id,
        max_tokens=800,
        temperature=0,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER}],
    )
    content = "".join(getattr(p, "text", "") for p in resp.content)
    return content

def main():
    last_error = None
    for m in MODEL_CANDIDATES:
        print(f"→ Trying model: {m}")
        try:
            content = try_model(m)
            print(f"\n✅ Using model: {m}\n")
            print("--- RAW CLAUDE OUTPUT ---\n")
            print(content)
            # Parse JSON
            data = json.loads(content)
            print("\n--- PARSED JSON ---\n")
            print(json.dumps(data, indent=2))
            print(f"\n✅ Parsed {len(data)} rows successfully.")
            return
        except NotFoundError as e:
            print(f"  • Not found on this account/SDK: {m}")
            last_error = e
        except json.JSONDecodeError as e:
            print("  • Model replied but not valid JSON. Raw (first 400 chars):")
            print(content[:400])
            raise
        except Exception as e:
            print(f"  • Error with {m}: {e}")
            last_error = e
    # If we get here, none worked.
    raise SystemExit(
        "\n❌ No candidate model IDs worked. "
        "Upgrade the SDK (pip install -U anthropic) and re-run."
    ) from last_error

if __name__ == "__main__":
    main()

