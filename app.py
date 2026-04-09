import os, io, csv, uuid, tempfile
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import re
from pathlib import Path
import zipfile
from collections import Counter

from ai_prefilter import prefilter_pdf_for_ai
import sda_yearbook_1800s_parser as p1800s
import sda_yearbook_1900s_parser as p1900s
import sda_yearbook_parser_ai as pai

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def pick_uploaded_pdfs(req):
    """
    Returns (List[FileStorage], reason_string)
    Accepts common field names and supports multi-file inputs.
    """
    preferred = ["file", "files", "pdf", "upload", "document"]
    found = []

    # Check the common upload field names first.
    for key in preferred:
        if key in req.files:
            items = req.files.getlist(key)
            for fs in items:
                if fs and getattr(fs, "filename", "") and allowed_file(fs.filename):
                    found.append(fs)

    # Fall back to any PDF-looking upload field.
    if not found:
        for _, fs in req.files.items():
            if fs and getattr(fs, "filename", "") and allowed_file(fs.filename):
                found.append(fs)

    if found:
        return found, f"found {len(found)} pdf(s)"
    if req.files:
        any_name = next(iter(req.files.keys()), None)
        return [], f"files present but not PDF (first field: {any_name})"
    return [], "no files in request"


def make_output_name(original_filename: str, year: Optional[int]) -> str:
    """
    Prefer YB1883.csv naming. If the year is unknown, use the filename stem.
    """
    if year:
        return f"YB{year}.csv"
    stem = Path(original_filename).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "yearbook"
    return f"{stem}.csv"

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
            "Model: <input name='model' value='gpt-4.1' /><br/>"
            "Year (optional): <input name='year' /><br/>"
            "<button type='submit'>Analyze</button></form>"
        )

    @app.route("/analyze", methods=["POST"])
    def analyze():
        pdf_files, how_found = pick_uploaded_pdfs(request)
        if not pdf_files:
            from datetime import datetime
            print(f"[analyze] {datetime.now().isoformat()} - upload issue: {how_found}; fields={list(request.files.keys())}")
            flash("I couldn't find a PDF in your upload. Please choose one or more .pdf files and try again.")
            return redirect(url_for("index"))

        engine = (request.form.get("engine") or "ai").strip().lower()
        provider = (request.form.get("provider") or "openai").strip().lower()
        model = (request.form.get("model") or "gpt-4.1").strip()
        year_override = request.form.get("year")
        year_override = int(year_override) if year_override and year_override.isdigit() else None

        # Keep each request in its own temporary result directory.
        out_root = os.path.join(tempfile.gettempdir(), "sda_yearbook_results")
        os.makedirs(out_root, exist_ok=True)
        token = uuid.uuid4().hex
        run_dir = os.path.join(out_root, token)
        os.makedirs(run_dir, exist_ok=True)

        tmpdir = tempfile.mkdtemp(prefix="sda_yearbook_")

        produced_csvs = []
        combined_rows_for_summary = []

        used_names = Counter()

        for fs in pdf_files:
            filename = secure_filename(fs.filename)
            pdf_path = os.path.join(tmpdir, filename)
            fs.save(pdf_path)

            year = year_override if year_override else guess_year_from_filename(filename)

            # Prefer year-based filenames, with a suffix if multiple uploads collide.
            base_name = make_output_name(filename, year)
            used_names[base_name] += 1
            if used_names[base_name] > 1:
                stem, ext = os.path.splitext(base_name)
                base_name = f"{stem}_{used_names[base_name]}{ext}"

            out_csv = os.path.join(run_dir, base_name)

            if engine == "rules":
                parser_mod = pick_rule_parser(year)
                rows = parser_mod.extract_from_pdf(pdf_path)
                rows_dicts = [r.__dict__ if hasattr(r, "__dict__") else r for r in rows]
                csv_bytes = rows_to_csv_bytes_rule(rows_dicts)
                with open(out_csv, "wb") as f:
                    f.write(csv_bytes)
                combined_rows_for_summary.extend(rows_dicts)

            else:
                # Prefilter before the model call so the prompt only contains
                # directory-like lines and a small amount of nearby context.
                pages = prefilter_pdf_for_ai(pdf_path, context_window=3)
                max_pages = int(os.environ.get("AI_MAX_PAGES", "40"))
                pages = pages[:max_pages]

                pre_txt = os.path.join(tmpdir, f"ai_pages_{uuid.uuid4().hex}.txt")
                with open(pre_txt, "w", encoding="utf-8") as f:
                    for i, txt in pages:
                        f.write(f"\n\n==== PAGE {i} ====\n{(txt or '').strip()}\n")

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

                # Reload the written CSV so the summary path is the same for all engines.
                try:
                    import pandas as pd
                    df = pd.read_csv(out_csv)
                    rows_dicts = df.fillna("").to_dict(orient="records")
                except Exception:
                    rows_dicts = []

                combined_rows_for_summary.extend(rows_dicts)

            produced_csvs.append(out_csv)

        # Bundle multi-file runs into a single archive for download.
        if len(produced_csvs) > 1:
            zip_path = os.path.join(run_dir, f"YB_outputs_{token}.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in produced_csvs:
                    z.write(p, arcname=os.path.basename(p))
            dl_url = url_for("download_token", token=token)
            info = summarize(combined_rows_for_summary)
            return (
                f"<p><b>Done.</b> Files: {len(produced_csvs)} • Rows: {info['total_rows']} • "
                f"<a href='{dl_url}'>Download ZIP</a></p>"
                f"<pre>Top conferences (sample): {info['top_conferences']}</pre>"
            )

        # Return the CSV directly when there is only one output file.
        dl_url = url_for("download_token", token=token)
        info = summarize(combined_rows_for_summary)
        return (
            f"<p><b>Done.</b> Rows: {info['total_rows']}. "
            f"<a href='{dl_url}'>Download CSV</a></p>"
            f"<pre>Top conferences (sample): {info['top_conferences']}</pre>"
        )
    @app.route("/download/<token>", methods=["GET"])
    def download_token(token):
        run_dir = os.path.join(tempfile.gettempdir(), "sda_yearbook_results", token)
        if not os.path.isdir(run_dir):
            from flask import abort
            abort(404)

        # Multi-file runs create a ZIP; single-file runs leave just a CSV.
        zips = [p for p in os.listdir(run_dir) if p.lower().endswith(".zip")]
        if zips:
            zip_path = os.path.join(run_dir, zips[0])
            return send_file(zip_path, mimetype="application/zip", as_attachment=True, download_name=zips[0])

        csvs = [p for p in os.listdir(run_dir) if p.lower().endswith(".csv")]
        if len(csvs) == 1:
            csv_path = os.path.join(run_dir, csvs[0])
            return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name=csvs[0])

        from flask import abort
        abort(404)
    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=True)




