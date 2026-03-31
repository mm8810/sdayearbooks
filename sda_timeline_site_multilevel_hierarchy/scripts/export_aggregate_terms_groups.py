#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def normalize(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_key(text: str) -> str:
    return normalize(text).lower()


def sheet_rows(workbook: Path, sheet_name: str):
    with ZipFile(workbook) as zf:
      shared = []
      if "xl/sharedStrings.xml" in zf.namelist():
          root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
          shared = ["".join((t.text or "") for t in si.iterfind(".//a:t", NS)) for si in root.findall("a:si", NS)]

      root = ET.fromstring(zf.read(f"xl/worksheets/{sheet_name}.xml"))
      for row in root.findall(".//a:sheetData/a:row", NS):
          values = {}
          for cell in row.findall("a:c", NS):
              ref = cell.attrib.get("r", "")
              col = "".join(ch for ch in ref if ch.isalpha())
              cell_type = cell.attrib.get("t")
              value = ""
              if cell_type == "s":
                  v = cell.find("a:v", NS)
                  if v is not None and v.text is not None:
                      value = shared[int(v.text)]
              elif cell_type == "inlineStr":
                  value = "".join((t.text or "") for t in cell.iterfind(".//a:t", NS))
              else:
                  v = cell.find("a:v", NS)
                  if v is not None and v.text is not None:
                      value = v.text
              values[col] = normalize(value)
          yield int(row.attrib["r"]), values


def build_payload(workbook: Path) -> dict:
    org_map = {}
    group_counter = Counter()
    starred_terms = []

    for row_number, values in sheet_rows(workbook, "sheet3"):
        if row_number == 1:
            continue
        original = normalize(values.get("A", ""))
        group = normalize(values.get("B", ""))
        if not original or not group:
            continue
        org_map[normalize_key(original)] = group
        group_counter[group] += 1
        if group == "*":
            starred_terms.append(original)

    ordered_groups = [name for name, _ in group_counter.most_common() if name != "*"]
    if "*" in group_counter:
        ordered_groups.append("*")

    return {
        "source_workbook": workbook.name,
        "organization_group_aliases": org_map,
        "organization_group_order": ordered_groups,
        "starred_terms": starred_terms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export grouped aggregate terms workbook to JSON.")
    parser.add_argument(
        "workbook",
        nargs="?",
        default="/Users/maryma/sdayearbooks/sda_timeline_site_multilevel_hierarchy/SDA%20Aggregate%20Terms_grouped.xlsx",
        help="Path to the grouped workbook.",
    )
    parser.add_argument(
        "--out",
        default="/Users/maryma/sdayearbooks/sda_timeline_site_multilevel_hierarchy/aggregate_terms_groups.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    workbook = Path(args.workbook)
    out = Path(args.out)
    payload = build_payload(workbook)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {out}")
    print(f"aliases={len(payload['organization_group_aliases'])}")
    print(f"groups={len(payload['organization_group_order'])}")
    print(f"stars={len(payload['starred_terms'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
