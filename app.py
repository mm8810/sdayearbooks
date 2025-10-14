import os
import io
import csv
from typing import List
from dataclasses import asdict
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import uuid, tempfile

# Use the local parser
import sda_yearbook_parser as parser

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def rows_to_csv_bytes(rows: List[parser.Row]) -> bytes:
    # Build CSV (UTF-8) in-memory using the parser's own field_map (keeps column order consistent)
    output = io.StringIO()
    w = csv.writer(output, lineterminator="\n")
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
    w.writerow(headers)
    for r in rows:
        d = asdict(r)
        w.writerow([d.get(fm[1], "") for fm in field_map])
    return output.getvalue().encode("utf-8")

def summarize(rows: List[parser.Row]):
    # Simple counts and basic data quality indicators
    from collections import Counter
    years = Counter([r.yearbook_year for r in rows])
    conferences = Counter([r.conference for r in rows])
    # Missing-field counts
    missing = {
        "conference": sum(1 for r in rows if not getattr(r, "conference", None)),
        "organization": sum(1 for r in rows if not getattr(r, "organization", None)),
        "group": sum(1 for r in rows if not getattr(r, "group", None)),
    }
    return {
        "total_rows": len(rows),
        "years": sorted(years.items()),
        "conferences": sorted(conferences.items()),
        "missing": missing
    }

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/analyze", methods=["POST"])
    def analyze():
        if "files" not in request.files:
            flash("No files part in request.")
            return redirect(url_for("index"))
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            flash("Please choose at least one PDF.")
            return redirect(url_for("index"))

        uploaded_paths = []
        workdir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(workdir, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(workdir, filename)
                file.save(save_path)
                uploaded_paths.append(save_path)

        if not uploaded_paths:
            flash("No valid PDF files found.")
            return redirect(url_for("index"))

        # Run extraction for each file and collect rows
        all_rows = []
        errors = []
        for path in sorted(set(uploaded_paths)):
            try:
                rows = parser.extract_from_pdf(path)
                all_rows.extend(rows)
            except Exception as e:
                errors.append((os.path.basename(path), str(e)))

        # Prepare CSV bytes
        csv_bytes = rows_to_csv_bytes(all_rows)

# Save to a temp file and pass a token to the template
        tmpdir = os.path.join(tempfile.gettempdir(), "sda_yearbook_results")
        os.makedirs(tmpdir, exist_ok=True)
        token = str(uuid.uuid4())
        csv_name = f"sda_yearbook_{token}.csv"
        csv_path = os.path.join(tmpdir, csv_name)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        # Build summary and render
        stats = summarize(all_rows)
        return render_template(
            "results.html",
            csv_name="sda_yearbook_results.csv",  # the user-facing filename
            token=token,                           # used to fetch the temp file
            stats=stats,
            errors=errors
    )

    @app.route("/download/<token>", methods=["GET"])
    def download_token(token):
        import os, tempfile
        tmpdir = os.path.join(tempfile.gettempdir(), "sda_yearbook_results")
        # Find the file for this token
        # We named it sda_yearbook_{token}.csv above
        csv_path = os.path.join(tmpdir, f"sda_yearbook_{token}.csv")
        return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name="sda_yearbook_results.csv")
    return app

if __name__ == "__main__":
    # For local testing: `python app.py`, then open http://127.0.0.1:5000
    app = create_app()
    app.run(host="0.0.0.0", port=5010, debug=True)
