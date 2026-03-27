#!/usr/bin/env python3
"""Render a Markdown guide from normalization.json."""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable


SECTION_ORDER = [
    ("Person Abbreviation Aliases", "person_abbreviation_aliases", ("Abbreviation", "Canonical form")),
    ("Person Exact Aliases", "person_exact_aliases", ("Raw name", "Canonical name")),
    ("Conference Exact Aliases", "conference_exact_aliases", ("Raw conference", "Canonical conference")),
    ("Conference Family Aliases", "conference_family_aliases", ("Raw family", "Canonical family")),
    ("Conference Rollup Aliases", "conference_rollup_aliases", ("Conference", "Guide rollup")),
    ("Conference Category Aliases", "conference_category_aliases", ("Conference", "Category")),
    ("Organization Exact Aliases", "organization_exact_aliases", ("Raw organization", "Canonical organization")),
    ("Organization Family Aliases", "organization_family_aliases", ("Raw family", "Canonical family")),
    ("Organization Type Aliases", "organization_type_aliases", ("Raw type", "Canonical type")),
    ("Region Exact Aliases", "region_exact_aliases", ("Raw region", "Canonical region")),
]


def escape_pipes(text: str) -> str:
    return str(text).replace("|", "\\|")


def sorted_items(mapping: dict[str, str]) -> Iterable[tuple[str, str]]:
    return sorted(mapping.items(), key=lambda item: (item[0].lower(), item[1].lower()))


def render_table(rows: Iterable[tuple[str, str]], headers: tuple[str, str]) -> str:
    rows = list(rows)
    if not rows:
        return "_No entries._\n"
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "| --- | --- |",
    ]
    for left, right in rows:
        lines.append(f"| {escape_pipes(left)} | {escape_pipes(right)} |")
    return "\n".join(lines) + "\n"


def build_guide(payload: dict) -> str:
    lines = [
        "# Normalization Guide",
        "",
        "This guide is generated from `normalization.json` and documents the canonical values the app uses for people, conferences, organizations, and regions.",
        "",
        "When a raw value appears in the left column of a table below, the app rewrites it to the canonical value in the right column before building filters, timeline rows, summary counts, or hierarchy views.",
        "",
    ]

    for title, key, headers in SECTION_ORDER:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(render_table(sorted_items(payload.get(key, {})), headers).rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--input",
        default=os.path.join(repo_root, "normalization.json"),
        help="Path to normalization.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(repo_root, "NORMALIZATION_GUIDE.md"),
        help="Path to write the Markdown guide",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)

    guide = build_guide(payload)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(guide)

    print(f"Wrote normalization guide to {args.output}")


if __name__ == "__main__":
    main()
