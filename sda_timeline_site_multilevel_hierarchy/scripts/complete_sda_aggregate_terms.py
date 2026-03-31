#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", NS)
ET.register_namespace("xml", XML_NS)


def qn(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def col_index(col: str) -> int:
    value = 0
    for ch in col:
        value = (value * 26) + (ord(ch.upper()) - 64)
    return value


def cell_parts(ref: str) -> tuple[str, int]:
    col = "".join(ch for ch in ref if ch.isalpha())
    row = int("".join(ch for ch in ref if ch.isdigit()))
    return col, row


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall(qn("si")):
        values.append("".join(node.text or "" for node in si.findall(".//" + qn("t"))))
    return values


def cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "s":
        value = cell.find(qn("v"))
        if value is None or value.text is None:
            return ""
        try:
            return shared_strings[int(value.text)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//" + qn("t")))
    value = cell.find(qn("v"))
    return value.text if value is not None and value.text is not None else ""


def set_inline_text(cell: ET.Element, text: str) -> None:
    cell.attrib.pop("s", None)
    cell.attrib.pop("vm", None)
    cell.attrib.pop("cm", None)
    cell.set("t", "inlineStr")
    for child in list(cell):
        cell.remove(child)
    is_el = ET.SubElement(cell, qn("is"))
    t_el = ET.SubElement(is_el, qn("t"))
    if text.strip() != text or "\n" in text:
        t_el.set(f"{{{XML_NS}}}space", "preserve")
    t_el.text = text


def ensure_cell(row_el: ET.Element, ref: str) -> ET.Element:
    for cell in row_el.findall(qn("c")):
        if cell.get("r") == ref:
            return cell

    new_cell = ET.Element(qn("c"), {"r": ref})
    target_col, _ = cell_parts(ref)
    target_index = col_index(target_col)
    inserted = False
    for idx, cell in enumerate(row_el.findall(qn("c"))):
        col, _ = cell_parts(cell.get("r", "A1"))
        if col_index(col) > target_index:
            row_el.insert(idx, new_cell)
            inserted = True
            break
    if not inserted:
        row_el.append(new_cell)
    return new_cell


def parse_sheet_rows(sheet_root: ET.Element) -> dict[int, ET.Element]:
    sheet_data = sheet_root.find(qn("sheetData"))
    if sheet_data is None:
        raise RuntimeError("Missing sheetData")
    rows = {}
    for row_el in sheet_data.findall(qn("row")):
        rows[int(row_el.get("r"))] = row_el
    return rows


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def lower(text: str) -> str:
    return normalize(text).lower()


def organization_group(name: str) -> str:
    raw = normalize(name)
    s = raw.lower()

    exact = {
        "department of education": "Educational Department",
        "educational department": "Educational Department",
        "educational dept.": "Educational Department",
        "general conference directory": "Workers' Directory",
        "general conference laborers": "Workers' Directory",
        "general laborers holding credentials from the north american division conference": "Workers' Directory",
        "minister directory": "Workers' Directory",
        "ministerial directory": "Workers' Directory",
        "ministers' directory": "Workers' Directory",
        "worker directory": "Workers' Directory",
        "workers directory": "Workers' Directory",
        "workers' directory": "Workers' Directory",
        "workers": "Workers' Directory",
        "workers' record": "Workers' Directory",
        "seventh-day adventist workers": "Workers' Directory",
        "workers in the s.d.a. cause": "Workers' Directory",
        "general agents": "Canvassing Agents",
        "state agents": "Canvassing Agents",
        "transportation agents": "Canvassing Agents",
        "religious liberty association": "Religious Liberty",
        "relief bureau": "*",
        "labor bureau": "*",
        "general organizations": "*",
        "general conference proceedings": "*",
        "licensed ministers": "*",
        "missionary schooner pitcairn": "Institution",
        "james white memorial home": "Institution",
        "educational society": "School",
        "s. d. a. educational society": "School",
        "seventh-day adventist educational society": "School",
        "seventh-day adventist educational society (battle creek college)": "School",
        "seventh-day adventist publishing association": "Publishing",
        "s. d. a. publishing association": "Publishing",
        "s.d.a. publishing association": "Publishing",
        "review and herald pub. assn.": "Publishing",
        "international pub. assn.": "Publishing",
        "pacific publishing association": "Publishing",
        "pacific s. d. a. publishing association": "Publishing",
        "pacific s.d.a. publishing association": "Publishing",
        "pacific seventh-day adventist publishing association": "Publishing",
        "good health publishing company": "Publishing",
        "echo publishing company, limited": "Publishing",
        "publishing department": "Publishing",
        "union book depository": "Publishing",
        "foreign literature depositories": "Publishing",
        "new york branch of review and herald pub. assn.": "Publishing",
        "sabbath-school and young people's department": "*",
        "young people's dept.": "Young People's Dept",
        "young people's missionary volunteer department": "Young People's Dept",
        "north american foreign department": "Foreign Department",
        "foreign departments": "Foreign Department",
        "north american negro department": "Negro Department",
        "home missionary department": "Home Missionary Department",
        "medical department": "Medical Department",
        "medical missionary department": "Medical Department",
        "general conference medical department": "Medical Department",
        "medical council": "Medical Department",
        "medical missionary council": "Medical Department",
        "foreign mission board": "Board",
        "trustees of the foreign mission board": "Board",
        "legal trustees of the foreign mission board": "Board",
        "publication committee": "Committee",
        "general conference executive committee": "Committee",
        "general conference committee": "Committee",
        "missionary campaign committees": "Campaign Committee",
        "sabbath-school association officers": "Sabbath-School",
        "sabbath-school department": "Sabbath-School",
        "sabbath-school association": "Sabbath-School",
        "sabbath-school association, oklahoma and indian territory": "Sabbath-School",
        "general field": "Conference",
        "foreign conferences": "Conference",
        "conference officers": "Conference",
        "laborers engaged in general work and in mission fields": "*",
        "managers circulating departments and branches": "*",
        "north american division conference publishing department": "Publishing",
        "north american conference corporation of seventh-day adventists": "Conference",
        "indiana association of seventh-day adventists": "Conference",
        "maine conference assn. of s. d. a.": "Conference",
        "southern new england conference assn. of s. d. a.": "Conference",
        "seventh-day adventist association of colorado": "Conference",
        "the seventh-day adventist assn. of colorado": "Conference",
    }
    if s in exact:
        return exact[s]

    if "camp-meeting committee" in s:
        return "Camp-Meeting Committee"
    if "campaign committee" in s:
        return "Campaign Committee"
    if "book committee" in s:
        return "Book Committee"
    if "committee" in s:
        return "Committee"
    if "council" in s:
        return "Committee"
    if "board" in s:
        return "Board"
    if "press bureau" in s:
        return "Press Bureau"
    if "religious liberty" in s:
        return "Religious Liberty"
    if "young people" in s or "missionary volunteer" in s:
        return "Young People's Dept"
    if "worker" in s or "directory" in s:
        return "Workers' Directory"
    if "canvassing agents" in s or s.endswith(" agents"):
        return "Canvassing Agents"
    if "health and temperance" in s or "h. and t." in s:
        return "Health and Temperance"
    if "sabbath-school" in s or "s. s." in s:
        return "Sabbath-School"
    if "tract and missionary" in s or "t. and m." in s:
        return "Tract and Missionary"
    if "home and foreign tract society" in s or "tract society" in s:
        return "Tract Society"
    if "city mission" in s:
        return "City Mission"
    if "mission field" in s:
        return "Mission"
    if s == "european missions" or s == "southern missionary society":
        return "Mission"
    if "missionary training school" in s:
        return "School"
    if "medical missionary and benevolent association" in s or "medical missionary and sanitarium association" in s:
        return "Medical Missionary and Benevolent Association"
    if "medical missionary" in s:
        return "Medical Department"
    if "publishing house" in s:
        return "Publishing House"
    if "publishing" in s or "pub. assn." in s or "pub. co." in s or "depository" in s:
        return "Publishing"
    if "educational department" in s or "department of education" in s or s == "educational dept.":
        return "Educational Department"
    if "foreign department" in s:
        return "Foreign Department"
    if "negro department" in s:
        return "Negro Department"
    if "home missionary" in s:
        return "Home Missionary Department"
    if "medical department" in s or "medical council" in s:
        return "Medical Department"
    if "conference legal association" in s or "legal assn." in s:
        return "Legal"
    if "conference association" in s or "conf. assn." in s or "conference corporation" in s or "conference agency" in s:
        return "Conference"
    if "association of seventh-day adventists" in s and "conference" in s:
        return "Conference"
    if s.startswith("conference ") or s == "conference":
        return "Conference"
    if "union conference" in s or " division conference" in s or s.endswith(" conference") or s.endswith(" conference.") or s.endswith(" district"):
        return "Conference"
    if s in {
        "dakota territory conference",
        "delaware and maryland",
        "district of columbia",
        "district of columbia and takoma park",
        "denmark",
        "england",
        "georgia",
        "great britain",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "louisiana",
        "maine",
        "michigan",
        "minnesota",
        "missouri",
        "nebraska",
        "new england",
        "new york",
        "north carolina",
        "north pacific",
        "norway",
        "ohio",
        "oklahoma and indian territory",
        "ontario",
        "pennsylvania",
        "province of quebec",
        "south africa",
        "sweden",
        "switzerland conference",
        "tennessee",
        "texas",
        "upper columbia",
        "vermont",
        "virginia",
        "west virginia",
        "wisconsin",
    }:
        return "Conference"
    if s.endswith(" mission") or s.endswith(" union mission") or " mission " in s:
        return "Mission"
    if "school" in s or "college" in s or "academy" in s or "faculty" in s:
        return "School"
    if "sanitarium" in s or "health reform institute" in s or "retreat" in s:
        return "Sanitarium"
    if "institution" in s or "church of s. d. adventists" in s or s.endswith(" home"):
        return "Institution"
    return "*"


def aggregate_group(term: str) -> str:
    raw = normalize(term)
    s = raw.lower()

    exact = {
        "conference": "Conference",
        "conference association": "Conference Association",
        "conference officers": "Conference",
        "department of education": "Educational Department",
        "educational department": "Educational Department",
        "field educational department": "Educational Department",
        "employees free training school": "Educational Department",
        "institute of music": "Educational Department",
        "normal department": "Educational Department",
        "preparatory department": "Educational Department",
        "primary department": "Educational Department",
        "school of theology faculty": "Educational Department",
        "theological school": "Educational Department",
        "training-school": "Educational Department",
        "training-school faculty": "Educational Department",
        "training-school for nurses": "Educational Department",
        "vocational faculty": "Educational Department",
        "health and temperance association | health and temperance society": "Health and Temperance Association",
        "sabbath-school department | sabbath-school association": "Sabbath-School Department",
        "tract society department | tract society | tract and missionary society": "Tract and Missionary Society",
        "religious liberty bureau | religious liberty association | religious liberty department": "Religious Liberty Department",
        "church and missionary schools | church schools": "Church and Missionary School",
        "medical department": "Medical Department",
        "medical faculty": "Medical Department",
        "medical council": "Medical Department",
        "medical missionary board": "Medical Missionary Department",
        "medical missionary council": "Medical Missionary Department",
        "medical missionary department": "Medical Missionary Department",
        "medical missionary training": "Medical Missionary Department",
        "medical mission board": "Medical Missionary Department",
        "foreign mission board": "Foreign Mission Board",
        "home missionary": "Home Missionary Department",
        "home missionary department": "Home Missionary Department",
        "missionary department": "Missionary Department",
        "missionary volunteer department": "Young People's Department",
        "volunteer department": "Young People's Department",
        "young people's missionary volunteer department": "Young People's Department",
        "young people's department | young people's society": "Young People's Department",
        "sabbath-school and young people's department": "Young People's Department",
        "press bureau": "Press Bureau",
        "publishing department": "Publishing Department",
        "foreign literature depositories": "Publishing Department",
        "union book depository": "Publishing Department",
        "union conference books": "Publishing Department",
        "periodical department": "Publishing Department",
        "book department": "Publishing Department",
        "book society": "Publishing Department",
        "general": "General",
        "general laborers holding credentials from the general conference": "General",
        "general conference committee": "General",
        "general conference association": "General",
        "general conference association for district four": "General",
        "general conference corporation": "General",
        '"indiana association of seventh-day adventists" | "the indiana association of seventh-day adventists"': "Conference Association",
        '"southeastern california association of seventh-day adventists"': "Conference Association",
        "north american foreign department": "Foreign Department",
        "north american negro | north american negro department": "Negro Department",
        "negro mission department": "Negro Department",
        "negro department": "Negro Department",
        "foreign department": "Foreign Department",
        "transportation agents": "Canvassing Agents",
        "sabbatarian association": "Sabbatarian Association",
        "canadian branch": "*",
        "atlanta branch": "*",
        "fort worth branch": "*",
        "international branch": "*",
        "kansas city branch": "*",
        "los angeles branch": "*",
        "new york branch": "*",
        "portland branch": "*",
        "regina branch": "*",
        "south bend branch": "*",
        "st. paul branch": "*",
        "texas branch": "*",
        "washington branch": "*",
        "western branch": "*",
        "book department": "Publishing Department",
        "book society": "Publishing Department",
        "campaign literature department": "Publishing Department",
        "los angeles department": "*",
        "manual arts department": "Educational Department",
        "labor bureau": "*",
        "german work": "German Work",
        "sanitarium work": "Sanitarium Work",
        "scandinavian work": "Scandinavian Work",
        "presidents of union conferences": "Conference",
        "secretaries of departments": "*",
        "conference association": "Conference Association",
        "general conference association": "Conference Association",
    }
    if s in exact:
        return exact[s]

    if "conference association" in s or "conference corporation" in s or "conference agency" in s:
        return "Conference Association"
    if "medical missionary and benevolent association" in s or "medical missionary and sanitarium association" in s:
        return "Medical Missionary and Benevolent Association"
    if "church and missionary school" in s:
        return "Church and Missionary School"
    if "educational" in s or "school" in s or "faculty" in s or "institute of music" in s:
        return "Educational Department"
    if "health and temperance" in s:
        return "Health and Temperance Association"
    if "sabbath-school" in s:
        return "Sabbath-School Department"
    if "tract and missionary" in s or "tract society" in s:
        return "Tract and Missionary Society"
    if "religious liberty" in s:
        return "Religious Liberty Department"
    if "publishing" in s or "book depository" in s or "book" in s or "literature depositories" in s or "press bureau" in s:
        return "Publishing Department"
    if "missionary volunteer" in s or "young people" in s or "volunteer department" in s:
        return "Young People's Department"
    if "mission board" in s:
        return "Foreign Mission Board"
    if "missionary department" in s or "home missionary" in s:
        return "Missionary Department"
    if "medical missionary" in s:
        return "Medical Missionary Department"
    if "medical department" in s or "medical council" in s:
        return "Medical Department"
    if "foreign department" in s:
        return "Foreign Department"
    if "negro department" in s or "negro mission department" in s:
        return "Negro Department"
    if "transportation agents" in s or "agents" in s:
        return "Canvassing Agents"
    if "conference" in s:
        return "Conference"
    return "*"


def update_sheet(
    sheet_root: ET.Element, shared_strings: list[str], source_col: str, target_col: str, mapper, stats_key: str
) -> Counter:
    rows = parse_sheet_rows(sheet_root)
    stats: Counter[str] = Counter()
    for rn, row_el in rows.items():
        if rn == 1:
            continue
        source_text = ""
        target_cell = None
        for cell in row_el.findall(qn("c")):
            col, _ = cell_parts(cell.get("r", ""))
            if col == source_col:
                source_text = normalize(cell_text(cell, shared_strings))
            elif col == target_col:
                target_cell = cell
        if not source_text:
            continue
        current_target = normalize(cell_text(target_cell, shared_strings)) if target_cell is not None else ""
        if current_target:
            continue
        mapped = mapper(source_text)
        target_ref = f"{target_col}{rn}"
        target_cell = ensure_cell(row_el, target_ref)
        set_inline_text(target_cell, mapped)
        stats[f"{stats_key}:filled"] += 1
        stats[f"{stats_key}:mapped:{mapped}"] += 1
    return stats


def rewrite_workbook(path: Path, dry_run: bool = False) -> Counter:
    stats: Counter[str] = Counter()
    with ZipFile(path) as zin:
        shared_strings = load_shared_strings(zin)
        sheet2_root = ET.fromstring(zin.read("xl/worksheets/sheet2.xml"))
        sheet3_root = ET.fromstring(zin.read("xl/worksheets/sheet3.xml"))

        stats.update(update_sheet(sheet2_root, shared_strings, "B", "A", aggregate_group, "sheet2"))
        stats.update(update_sheet(sheet3_root, shared_strings, "A", "B", organization_group, "sheet3"))

        if dry_run:
            return stats

        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp_path = Path(tmp.name)

        with ZipFile(path) as zin, ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "xl/worksheets/sheet2.xml":
                    data = ET.tostring(sheet2_root, encoding="utf-8", xml_declaration=True)
                elif item.filename == "xl/worksheets/sheet3.xml":
                    data = ET.tostring(sheet3_root, encoding="utf-8", xml_declaration=True)
                zout.writestr(item, data)

        shutil.move(tmp_path, path)
        stats["workbook:backup_written"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Complete SDA Aggregate Terms workbook groupings.")
    parser.add_argument(
        "workbook",
        nargs="?",
        default="/Users/maryma/sdayearbooks/sda_timeline_site_multilevel_hierarchy/SDA Aggregate Terms.xlsx",
        help="Path to the workbook to update.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be filled without modifying the workbook.")
    args = parser.parse_args()

    workbook = Path(args.workbook)
    if not workbook.exists():
        print(f"Workbook not found: {workbook}", file=sys.stderr)
        return 1

    stats = rewrite_workbook(workbook, dry_run=args.dry_run)
    for key in sorted(stats):
        print(f"{key}={stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
