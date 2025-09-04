# SDA Yearbooks NER

This repository contains a simple script to extract named entities from
Seventh-day Adventist Church yearbooks in PDF format and export them to a
CSV file.

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Run the script:

   ```bash
   python ner_yearbooks.py path/to/pdfs entities.csv
   ```

The resulting `entities.csv` will contain the columns `file`, `entity`, and
`label` for each entity detected in the PDFs.
