#!/usr/bin/env python3
"""Extract named entities from PDF yearbooks and save them to a CSV file.

The script scans a directory for PDF files, extracts text from each file,
performs named entity recognition using spaCy, and writes the results to a
CSV file with the following columns:

    file, entity, label

Example usage::

    python ner_yearbooks.py path/to/pdfs output.csv

"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import spacy
from PyPDF2 import PdfReader


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text_parts = []
    with pdf_path.open("rb") as fh:
        reader = PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def iter_pdfs(folder: Path) -> Iterable[Path]:
    """Yield PDF file paths from ``folder`` sorted alphabetically."""
    for path in sorted(folder.glob("*.pdf")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform named entity recognition on PDF yearbooks and export results to CSV."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PDF yearbooks")
    parser.add_argument("output_csv", type=Path, help="Path to the CSV file to create")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory {args.input_dir} does not exist or is not a directory")

    nlp = spacy.load("en_core_web_sm")

    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "entity", "label"])

        for pdf_path in iter_pdfs(args.input_dir):
            text = extract_text(pdf_path)
            doc = nlp(text)
            for ent in doc.ents:
                writer.writerow([pdf_path.name, ent.text, ent.label_])


if __name__ == "__main__":
    main()
