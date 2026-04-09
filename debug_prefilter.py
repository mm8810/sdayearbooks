import os, io, csv, uuid, tempfile
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import re
from pathlib import Path

from ai_prefilter import prefilter_pdf_for_ai
import sda_yearbook_1800s_parser as p1800s
import sda_yearbook_1900s_parser as p1900s
import sda_yearbook_parser_ai as pai

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _pick_uploaded_file(req):
    """
    Accept common field names and fall back to any PDF upload field.
    Returns (FileStorage or None, reason_string).
    """
    preferred = ["file", "pdf", "upload", "document"]
    for key in preferred:
        fs = req.files.get(key)
        if fs and getattr(fs, "filename", "") and allowed_file(fs.filename):
            return fs, f"found in field '{key}'"
    # Fall back to any upload whose filename ends in .pdf.
    for key, fs in req.files.items():
        if fs and getattr(fs, "filename", "") and allowed_file(fs.filename):
            return fs, f"found in field '{key}' (fallback)"
    if req.files:
        any_name = next(iter(req.files.keys()), None)
        return None, f"files present but not PDF (first field: {any_name})"
    return None, "no files in request"

def guess_year_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(18|19)\d{2}", name)
    return int(m.group(0)) if m else None

def pick_rule_parser(year: Optional[int]):
    if year is not None and year <= 1899:
        return p1800s
    return p1900s

def rows_to_csv_bytes_rule(rows) -> bytes:
    """Serialize rule-parser output to CSV bytes."""
    if not rows:
        return b""
    if hasattr(rows[0], "__dict__"):
        keys = list(rows[0].__dict__.keys())
        data = [r.__dict__ for r in rows]
    else:
        keys = list(rows[0].keys())
        data = rows
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    for r in data:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")

def summarize(rows: List[Dict[str, Any]]):
    from collections import Counter
    years = Counter([r.get("yearbook_year") for r in rows])
    conferences = Counter([r.get("conference") for r in rows])

    def sort_key_year(item):
        yr, _ = item
        return (yr is None, yr if isinstance(yr, int) else 0)

    def sort_key_text(item):
        k, _ = item
        return (k is None, ("" if k is None else str(k)).lower())

    return {
        "total_rows": len(rows),
        "by_year": sorted(years.items(), key=sort_key_year),
        "top_conferences": sorted(conferences.items(), key=sort_key_text)[:20],
    }

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html") if os.path.exists("templates/index.html") else (
            "<h3>SDA Yearbook Parser</h3>"
            "<form method='post' action='/analyze' enctype='multipart/form-data'>"
            "PDF: <input type='file' name='file' /><br/>"
            "Engine: <select name='engine'><option>rules</option><option selected>ai</option></select><br/>"
            "Provider: <select name='provider'><option>openai</option><option>anthropic</option></select><br/>"
            "Model: <input name='model' value='gpt-5-mini' /><br/>"
            "Year (optional): <input name='year' /><br/>"
            "<button type='submit'>Analyze</button></form>"
        )

    @app.route("/analyze", methods=["POST"])
    def analyze():
        file, how_found = _pick_uploaded_file(request)
        if not file or file.filename == "" or not allowed_file(file.filename):
            from datetime import datetime
            print(f"[analyze] {datetime.now().isoformat()} - upload issue: {how_found}; fields={list(request.files.keys())}")
            flash("I couldn't find a PDF in your upload. Please choose a .pdf file and try again. "
                  "(Tip: the file input name should be 'file', 'pdf', or 'upload')")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        tmpdir = tempfile.mkdtemp(prefix="sda_yearbook_")
        pdf_path = os.path.join(tmpdir, filename)
        file.save(pdf_path)

        engine = (request.form.get("engine") or "ai").strip().lower()
        provider = (request.form.get("provider") or "openai").strip().lower()
        model = (request.form.get("model") or "gpt-5-mini").strip()
        year = request.form.get("year")
        year = int(year) if year and year.isdigit() else guess_year_from_filename(filename)

        # Use a tokenized filename so repeated runs do not overwrite each other.
        out_dir = os.path.join(tempfile.gettempdir(), "sda_yearbook_results")
        os.makedirs(out_dir, exist_ok=True)
        token = uuid.uuid4().hex
        out_csv = os.path.join(out_dir, f"sda_yearbook_{token}.csv")

        if engine == "rules":
            parser_mod = pick_rule_parser(year)
            rows = parser_mod.extract_from_pdf(pdf_path)
            rows_dicts = [r.__dict__ if hasattr(r, "__dict__") else r for r in rows]
            csv_bytes = rows_to_csv_bytes_rule(rows_dicts)
            with open(out_csv, "wb") as f:
                f.write(csv_bytes)
        else:
            # Log each prefilter stage so it is easy to inspect what the model receives.
            print(f"🔍 DEBUG: Starting AI analysis of {pdf_path}")
            pages = prefilter_pdf_for_ai(pdf_path, context_window=3)
            print(f"🔍 DEBUG: Prefilter found {len(pages)} pages")
            
            max_pages = int(os.environ.get("AI_MAX_PAGES", "40"))
            pages = pages[:max_pages]
            print(f"🔍 DEBUG: Processing {len(pages)} pages (limited by AI_MAX_PAGES)")

            # Print a small sample of the prefiltered pages before calling the model.
            if not pages:
                print("❌ DEBUG: NO PAGES survived prefiltering! This is the problem.")
                print("💡 SUGGESTION: Try running debug_prefilter.py on your PDF")
            else:
                for i, (page_num, content) in enumerate(pages[:3]):
                    print(f"🔍 DEBUG: Page {page_num} has {len(content)} characters")
                    print(f"🔍 DEBUG: First 100 chars: {repr(content[:100])}")

            # Persist the exact prefiltered payload used by the AI runner.
            pre_txt = os.path.join(tmpdir, f"ai_pages_{token}.txt")
            with open(pre_txt, "w", encoding="utf-8") as f:
                for i, txt in pages:
                    f.write(f"\n\n==== PAGE {i} ====\n{(txt or '').strip()}\n")
            
            print(f"🔍 DEBUG: Wrote prefiltered text to {pre_txt}")
            
            # Flag unexpectedly small bundles before the model call.
            if os.path.exists(pre_txt):
                with open(pre_txt, 'r') as f:
                    content = f.read()
                print(f"🔍 DEBUG: Prefiltered file has {len(content)} characters")
                if len(content) < 100:
                    print(f"⚠️  WARNING: Very little content in prefiltered file!")
                    print(f"Content: {repr(content[:200])}")

            pai.run(
                pdf=Path(pdf_path),
                provider_name=provider,
                model=model,
                out_csv=Path(out_csv),
                year=year,
                start=None,
                end=None,
                max_pages=max_pages,
                prefiltered_text_path=Path(pre_txt),
            )

            # Reload the output CSV so the UI summary matches the saved file.
            import pandas as pd
            try:
                df = pd.read_csv(out_csv)
                rows_dicts = df.fillna("").to_dict(orient="records")
                print(f"🔍 DEBUG: Final CSV has {len(rows_dicts)} rows")
            except Exception as e:
                print(f"❌ DEBUG: Error reading final CSV: {e}")
                rows_dicts = []

        info = summarize(rows_dicts if engine != "rules" else rows_dicts)
        dl_url = url_for("download_token", token=token)
        
        # Include the prefilter bundle path in the response for troubleshooting.
        debug_info = ""
        if engine == "ai":
            debug_info = f"<br/>Debug: {len(pages)} pages processed, prefiltered file: {pre_txt}"
        
        html = (
            f"<p><b>Done.</b> Rows: {info['total_rows']}. "
            f"<a href='{dl_url}'>Download CSV</a></p>"
            f"<pre>Top conferences (sample): {info['top_conferences']}</pre>"
            f"{debug_info}"
        )
        return html

    @app.route("/download/<token>", methods=["GET"])
    def download_token(token):
        tmpdir = os.path.join(tempfile.gettempdir(), "sda_yearbook_results")
        csv_path = os.path.join(tmpdir, f"sda_yearbook_{token}.csv")
        if not os.path.exists(csv_path):
            from flask import abort
            abort(404)
        return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name="sda_yearbook_results.csv")

    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=True)
