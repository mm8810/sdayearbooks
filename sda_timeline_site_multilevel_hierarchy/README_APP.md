## Local run

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Adding more yearbooks

1. Place parsed CSVs in `data/`.
2. Register each file in `manifest.json`.
3. Refresh the page.

Example:

```json
{
  "datasets": [
    { "year": 1883, "file": "data/1883.csv", "label": "1883 Yearbook" },
    { "year": 1884, "file": "data/1884.csv", "label": "1884 Yearbook" }
  ]
}
```

## Timeline model

- The timeline uses one row per guide-based conference rollup.
- Each timeline rollup is derived from the conference normalization guide, so state or subconference names can roll up into headings like `Atlantic Union / Columbia Union`, `Lake Union`, `Southern Union`, `Pacific Union`, and `General`.
- Each conference row shows one or more active spans based on the years where that rollup appears in the loaded data.
- Conference rows are ordered by guide-based conference categories such as `General`, `North American - Lake Union`, and `North American - Pacific Union`.
- `Region` is treated as metadata and filtering context, not as the main timeline lane.
- Clicking a conference span at a given year opens a year-specific leadership view.
- The popup uses the selected rollup and year, then shows `organization -> position -> people` along with the source conferences included in that rollup.
- The popup uses the yearbooks currently loaded from `manifest.json` and the current filters.

## Normalization system

- Canonical mappings live in `normalization.json`.
- The app computes `conference_canonical` and `organization_canonical` for every row before filtering or rendering the timeline.
- The goal is to merge spelling and label variants without collapsing genuinely different entity types.
- The normalization guide PDF is the primary source for explicit aliases, especially for institution renames and equivalent organization labels.
- A small amount of rule-based cleanup also trims obvious legal-name suffixes such as `Conference Association of Seventh-day Adventists` down to the underlying conference association name.

Examples:

- `British` and `Great Britain` normalize to `British`.
- `Central European` normalizes to `Central Europe`.
- `Workers Directory`, `Worker Directory`, and `S.D.A. Workers' Directory` normalize to `Workers' Directory`.

## Maintaining normalization rules

Use the helper script to generate candidate clusters from the raw CSV labels:

```bash
python3 scripts/generate_normalization_report.py --min-cluster-size 2
```

You can also write the report to a file:

```bash
python3 scripts/generate_normalization_report.py --output normalization-report.json
```

Review the suggested clusters, then promote confirmed variants into `normalization.json`.

## Exporting data

Use `Export view` to download the current filtered rows as JSON.
