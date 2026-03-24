#!/usr/bin/env python3
"""Generate candidate normalization clusters from the SDA yearbook CSVs.

This script is meant to support manual maintenance of normalization.json.
It does not overwrite canonical mappings automatically; instead it groups
related raw labels so they can be reviewed and promoted into aliases.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict


CONFERENCE_TYPES = [
    "union conference",
    "union mission",
    "mission field",
    "conference",
    "mission",
    "union",
]

ORGANIZATION_TYPES = [
    "union conference association",
    "conference tract society",
    "conference association",
    "union conference",
    "union mission",
    "tract and missionary society",
    "tract society",
    "publishing department",
    "publishing association",
    "sabbath-school department",
    "sabbath-school association",
    "directory",
    "department",
    "committee",
    "association",
    "society",
    "conference",
    "mission",
    "union",
    "school",
    "college",
    "academy",
    "hospital",
    "sanitarium",
    "office",
    "press",
]


def normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").strip())


def normalize_key(text: str) -> str:
    return normalize_spacing(text).lower()


def strip_type(label: str, candidates: list[str]) -> tuple[str, str]:
    clean = normalize_key(label)
    for candidate in sorted(set(candidates), key=lambda item: (-len(item), item)):
        if clean.endswith(f" {candidate}"):
            return clean[: -len(candidate) - 1].strip(), candidate
    return clean, ""


def load_rows(data_dir: str):
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)


def build_report(data_dir: str, min_cluster_size: int):
    conference_counts: Counter[str] = Counter()
    organization_counts: Counter[str] = Counter()
    conference_clusters: dict[str, list[str]] = defaultdict(list)
    organization_clusters: dict[str, list[str]] = defaultdict(list)

    for row in load_rows(data_dir):
        conference = normalize_spacing(row.get("conference", ""))
        if conference:
            conference_counts[conference] += 1
            family, _ = strip_type(conference, CONFERENCE_TYPES)
            conference_clusters[family or conference].append(conference)

        organization = normalize_spacing(row.get("organization") or row.get("institution_name") or "")
        if organization:
            organization_counts[organization] += 1
            family, _ = strip_type(organization, ORGANIZATION_TYPES)
            organization_clusters[family or organization].append(organization)

    def summarize(cluster_map: dict[str, list[str]], counts: Counter[str]):
        clusters = []
        for family, raw_values in cluster_map.items():
            uniq_values = sorted(set(raw_values))
            if len(uniq_values) < min_cluster_size:
                continue
            clusters.append(
                {
                    "family_key": family,
                    "variants": [
                        {"label": value, "rows": counts[value]}
                        for value in sorted(uniq_values, key=lambda item: (-counts[item], item))
                    ],
                }
            )
        return sorted(clusters, key=lambda item: (-len(item["variants"]), item["family_key"]))

    return {
        "data_dir": data_dir,
        "conference_clusters": summarize(conference_clusters, conference_counts),
        "organization_clusters": summarize(organization_clusters, organization_counts),
        "top_conferences": conference_counts.most_common(50),
        "top_organizations": organization_counts.most_common(50),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing parsed yearbook CSV files.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Only include clusters with at least this many distinct labels.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON report. Defaults to stdout.",
    )
    args = parser.parse_args()

    report = build_report(os.path.abspath(args.data_dir), args.min_cluster_size)
    payload = json.dumps(report, indent=2, ensure_ascii=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        print(f"Wrote normalization report to {args.output}")
        return

    print(payload)


if __name__ == "__main__":
    main()
