# SDA Yearbook Analyzer (Historian-Friendly)

A tiny web app (Flask) that lets a historian upload Seventh-day Adventist Yearbook PDFs and download a clean CSV of officers/roles using `sda_yearbook_parser.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ensure your parser module is on the import path (copied next to app.py or installed)

python app.py
```

Open http://127.0.0.1:5000, choose one or more yearbook PDFs, click **Analyze**, then download the CSV.

## How it works

- The app imports your `sda_yearbook_parser` and calls `extract_from_pdf(path)` for each uploaded file.
- It preserves your CSV column order via the parser's `field_map` (conference, region, institution-name, organization, group, position-information, position, prefix, name, lastname, suffix, gender, location, yearbook-year, page, source-pdf).
- The results page shows quick counts by year and conference and offers a one-click CSV download.
