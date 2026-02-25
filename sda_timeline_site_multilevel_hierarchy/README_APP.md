python3 -m http.server 8000
http://localhost:8000

Adding More Yearbooks
Place parsed CSVs in data/
data/1884.csv
data/1885.csv
Register them in manifest.json:
{
"datasets": [
{ "year": 1883, "file": "data/1883.csv", "label": "1883 Yearbook" },
{ "year": 1884, "file": "data/1884.csv", "label": "1884 Yearbook" }
]
}
Refresh the page. The dataset list updates automatically.
Hierarchy View
Click "Hierarchy view" in the top bar.
Select a conference.
Choose a grouping strategy:
Organization -> Position -> People
Position -> People
Organization -> Group -> Position -> People
Only enabled datasets are included.
Positions are grouped by exact string match.
Exporting Data
Click "Export view" to download the currently filtered rows as JSON.
Useful for network analysis, leadership composition analysis, debugging parsing pipelines, and feeding into Python or R notebooks.