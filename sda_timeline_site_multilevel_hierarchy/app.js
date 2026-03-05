let manifest = null;
let allRows = [];      // raw rows across all enabled datasets
let enabledYears = new Set();
let timeline = null;
let items = null;
let lastFiltered = [];

let hierarchyPinned = null;

const els = {};
function $(id){ return document.getElementById(id); }

function safe(v){
  if (v === null || v === undefined) return "";
  return String(v).trim();
}

function yearToDate(y){
  // place all points within a year at Jan 1; timeline groups/collisions handle density.
  return new Date(Number(y), 0, 1);
}

function buildTitle(row){
  const name = safe(row.name) || "(unknown)";
  const pos = safe(row.position);
  const org = safe(row.organization);
  const loc = safe(row.location);
  const page = safe(row.page);
  const bits = [];
  if (pos) bits.push(pos);
  if (org) bits.push(org);
  if (loc) bits.push(loc);
  const line = bits.join(" • ");
  return `${name}\n${line}${page ? `\nPage: ${page}` : ""}`;
}

function rowToItem(row, idx){
  const year = Number(row.yearbook_year);
  const date = yearToDate(year);

  const content = safe(row.name) || "(unknown)";
  const region = safe(row.region) || "Unknown";
  const group = region; // vis group = region (can change later)

  return {
    id: `${year}-${idx}`,
    content,
    start: date,
    group,
    title: buildTitle(row),
    _row: row
  };
}

function uniqSorted(values){
  const s = new Set(values.filter(v => safe(v)));
  return Array.from(s).sort((a,b)=> a.localeCompare(b));
}

function setStats(){
  const loaded = allRows.length;
  const shown = lastFiltered.length;
  const years = Array.from(enabledYears).sort((a,b)=>a-b);
  const yearLabel = years.length ? `${years[0]}–${years[years.length-1]}` : "–";
  els.statLoaded.textContent = `Loaded: ${loaded.toLocaleString()} rows`;
  els.statShown.textContent = `Shown: ${shown.toLocaleString()} rows`;
  els.statYears.textContent = `Years: ${yearLabel}`;
}

function hydrateFilterOptions(){
  const regions = uniqSorted(allRows.map(r => r.region));
  const confs = uniqSorted(allRows.map(r => r.conference));
  const positions = uniqSorted(allRows.map(r => r.position));
  const genders = uniqSorted(allRows.map(r => r.gender));
  const orgs = uniqSorted(
    allRows
      .map(r => safe(r.organization) || safe(r.institution_name))
      .filter(v => safe(v))
  );

  // Preserve current selection when possible
  const prevRegion = els.regionSelect.value;
  const prevConf = els.confSelect.value;
  const prevPos = els.positionSelect?.value || "";
  const prevGender = els.genderSelect?.value || "";
  const prevOrg = els.orgSelect?.value || "";

  els.regionSelect.innerHTML =
    '<option value="">All regions</option>' +
    regions.map(r => `<option value="${escapeHtml(r)}">${escapeHtml(r)}</option>`).join("");

  els.confSelect.innerHTML =
    '<option value="">All conferences</option>' +
    confs.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");

  if (els.positionSelect){
    els.positionSelect.innerHTML =
      '<option value="">All positions</option>' +
      positions.map(p => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("");
  }

  if (els.genderSelect){
    els.genderSelect.innerHTML =
      '<option value="">All genders</option>' +
      genders.map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join("");
  }

  if (els.orgSelect){
    els.orgSelect.innerHTML =
      '<option value="">All organizations</option>' +
      orgs.map(o => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("");
  }

  if (regions.includes(prevRegion)) els.regionSelect.value = prevRegion;
  if (confs.includes(prevConf)) els.confSelect.value = prevConf;
  if (els.positionSelect && positions.includes(prevPos)) els.positionSelect.value = prevPos;
  if (els.genderSelect && genders.includes(prevGender)) els.genderSelect.value = prevGender;
  if (els.orgSelect && orgs.includes(prevOrg)) els.orgSelect.value = prevOrg;
}

function escapeHtml(str){
  return String(str)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function applyFilters(){
  const q = safe(els.searchInput.value).toLowerCase();
  const region = safe(els.regionSelect.value);
  const conf = safe(els.confSelect.value);
  const pos = safe(els.positionSelect?.value);
  const gender = safe(els.genderSelect?.value);
  const orgSel = safe(els.orgSelect?.value);
  const yMin = Number(els.yearMin.value || 0);
  const yMax = Number(els.yearMax.value || 9999);

  const filtered = allRows.filter(r => {
    const y = Number(r.yearbook_year);
    if (Number.isFinite(yMin) && y < yMin) return false;
    if (Number.isFinite(yMax) && y > yMax) return false;

    if (q){
      const nm = safe(r.name).toLowerCase();
      const ln = safe(r.last_name).toLowerCase();
      if (!nm.includes(q) && !ln.includes(q)) return false;
    }

    if (region && safe(r.region) !== region) return false;
    if (conf && safe(r.conference) !== conf) return false;

    if (pos && safe(r.position) !== pos) return false;
    if (gender && safe(r.gender) !== gender) return false;

    if (orgSel){
      const orgLabel = safe(r.organization) || safe(r.institution_name);
      if (safe(orgLabel) !== orgSel) return false;
    }

    return true;
  });

  lastFiltered = filtered;

  const itemObjs = filtered.map((r, i) => rowToItem(r, i));
  items.clear();
  items.add(itemObjs);

  const groupNames = uniqSorted(itemObjs.map(it => it.group));
  const groups = groupNames.map(g => ({ id: g, content: g }));
  timeline.setGroups(groups);

  setStats();
}

function resetFilters(){
  els.searchInput.value = "";
  els.regionSelect.value = "";
  els.confSelect.value = "";
  if (els.positionSelect) els.positionSelect.value = "";
  if (els.genderSelect) els.genderSelect.value = "";
  if (els.orgSelect) els.orgSelect.value = "";
  els.yearMin.value = "1883";
  els.yearMax.value = "1921";
  applyFilters();
}

function renderDetail(row){
  if (!row){
    els.detailCard.classList.add("muted");
    els.detailCard.textContent = "Click a point on the timeline to see details here.";
    return;
  }
  els.detailCard.classList.remove("muted");

  const name = safe(row.name) || "(unknown)";
  const year = safe(row.yearbook_year);
  const page = safe(row.page);

  const fields = [
    ["Year", year],
    ["Page", page],
    ["Prefix", safe(row.prefix)],
    ["Last name", safe(row.last_name)],
    ["Suffix", safe(row.suffix)],
    ["Gender", safe(row.gender)],
    ["Position", safe(row.position)],
    ["Pos. info", safe(row.position_information)],
    ["Organization", safe(row.organization)],
    ["Group", safe(row.group)],
    ["Conference", safe(row.conference)],
    ["Institution", safe(row.institution_name)],
    ["Location", safe(row.location)],
    ["Region", safe(row.region)],
  ].filter(([k,v]) => safe(v));

  els.detailCard.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(name)}</div>
    <div class="kv">
      ${fields.map(([k,v]) => `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`).join("")}
    </div>
  `;
}


function openHierarchy(){
  els.hierarchyModal.setAttribute("aria-hidden", "false");

  const confs = uniqSorted(allRows.map(r => r.conference));
  const prev = els.hierarchyConferenceSelect.value;

  els.hierarchyConferenceSelect.innerHTML =
    '<option value="">Select a conference</option>' +
    confs.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");

  if (confs.includes(prev)) els.hierarchyConferenceSelect.value = prev;

  if (els.hierarchyConferenceSelect.value){
    renderHierarchyForConference(els.hierarchyConferenceSelect.value);
  } else {
    clearHierarchy();
  }
}

function closeHierarchy(){
  els.hierarchyModal.setAttribute("aria-hidden", "true");
}

function clearHierarchy(){
  els.hierarchyChart.innerHTML = "";
  hierarchyPinned = null;
  els.hierarchyDetail.classList.add("muted");
  els.hierarchyDetail.textContent = "Select a conference to render the hierarchy. Then click a node for details.";
}

function buildHierarchyData(confName){
  const rows = allRows.filter(r => safe(r.conference) === confName);

  const mode = safe(els.hierarchyGroupingSelect?.value) || "position";

  // Helpers
  const posLabel = (r) => safe(r.position) || "(No position)";
  const orgLabel = (r) => safe(r.organization) || safe(r.institution_name) || "(No organization)";
  const groupLabel = (r) => safe(r.group) || "(No group)";

  // Build a nested map according to mode
  // Leaves are people nodes
  function personNode(r){
    return { name: safe(r.name) || "(unknown)", kind: "person", _row: r };
  }

  if (mode === "conference_org_position") {
  const byOrg = new Map();

  for (const r of rows){
    const org = orgLabel(r);
    if (!byOrg.has(org)) byOrg.set(org, []);
    byOrg.get(org).push(r);
  }

  const orgs = Array.from(byOrg.keys()).sort((a,b)=>a.localeCompare(b));

  const children = orgs.map(org => {
    const orgRows = byOrg.get(org);

    const byPos = new Map();
    for (const r of orgRows){
      const pos = posLabel(r);
      if (!byPos.has(pos)) byPos.set(pos, []);
      byPos.get(pos).push(r);
    }

    const positions = Array.from(byPos.keys()).sort((a,b)=>a.localeCompare(b));

    return {
      name: org,
      kind: "organization",
      children: positions.map(pos => ({
        name: pos,
        kind: "position",
        children: byPos
          .get(pos)
          .slice()
          .sort((a,b)=>(safe(a.name)||"").localeCompare(safe(b.name)||""))
          .map(personNode)
      }))
    };
  });

  return {
    name: confName,
    kind: "conference",
    children
  };
}

  if (mode === "position"){
    const byPos = new Map();
    for (const r of rows){
      const pos = posLabel(r);
      if (!byPos.has(pos)) byPos.set(pos, []);
      byPos.get(pos).push(r);
    }
    const positions = Array.from(byPos.keys()).sort((a,b)=>a.localeCompare(b));
    const children = positions.map(pos => ({
      name: pos,
      kind: "position",
      children: byPos.get(pos).slice().sort((a,b)=>(safe(a.name)||"").localeCompare(safe(b.name)||"")).map(personNode)
    }));
    return { name: confName, kind: "conference", children };
  }

  if (mode === "org_position"){
    const byOrg = new Map();
    for (const r of rows){
      const org = orgLabel(r);
      if (!byOrg.has(org)) byOrg.set(org, []);
      byOrg.get(org).push(r);
    }
    const orgs = Array.from(byOrg.keys()).sort((a,b)=>a.localeCompare(b));
    const children = orgs.map(org => {
      const orgRows = byOrg.get(org);
      const byPos = new Map();
      for (const r of orgRows){
        const pos = posLabel(r);
        if (!byPos.has(pos)) byPos.set(pos, []);
        byPos.get(pos).push(r);
      }
      const positions = Array.from(byPos.keys()).sort((a,b)=>a.localeCompare(b));
      return {
        name: org,
        kind: "organization",
        children: positions.map(pos => ({
          name: pos,
          kind: "position",
          children: byPos.get(pos).slice().sort((a,b)=>(safe(a.name)||"").localeCompare(safe(b.name)||"")).map(personNode)
        }))
      };
    });
    return { name: confName, kind: "conference", children };
  }

  // org_group_position
  const byOrg = new Map();
  for (const r of rows){
    const org = orgLabel(r);
    if (!byOrg.has(org)) byOrg.set(org, []);
    byOrg.get(org).push(r);
  }
  const orgs = Array.from(byOrg.keys()).sort((a,b)=>a.localeCompare(b));
  const children = orgs.map(org => {
    const orgRows = byOrg.get(org);

    const byGroup = new Map();
    for (const r of orgRows){
      const g = groupLabel(r);
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g).push(r);
    }
    const groups = Array.from(byGroup.keys()).sort((a,b)=>a.localeCompare(b));

    return {
      name: org,
      kind: "organization",
      children: groups.map(g => {
        const gRows = byGroup.get(g);

        const byPos = new Map();
        for (const r of gRows){
          const pos = posLabel(r);
          if (!byPos.has(pos)) byPos.set(pos, []);
          byPos.get(pos).push(r);
        }
        const positions = Array.from(byPos.keys()).sort((a,b)=>a.localeCompare(b));

        return {
          name: g,
          kind: "group",
          children: positions.map(pos => ({
            name: pos,
            kind: "position",
            children: byPos.get(pos).slice().sort((a,b)=>(safe(a.name)||"").localeCompare(safe(b.name)||"")).map(personNode)
          }))
        };
      })
    };
  });

  return { name: confName, kind: "conference", children };
}

function renderHierarchyRow(row){
  const name = safe(row.name) || "(unknown)";
  const fields = [
    ["Year", safe(row.yearbook_year)],
    ["Page", safe(row.page)],
    ["Position", safe(row.position)],
    ["Organization", safe(row.organization)],
    ["Conference", safe(row.conference)],
    ["Region", safe(row.region)],
    ["Location", safe(row.location)],
    ["Institution", safe(row.institution_name)],
  ].filter(([k,v]) => safe(v));

  els.hierarchyDetail.classList.remove("muted");
  els.hierarchyDetail.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(name)}</div>
    <div class="kv">
      ${fields.map(([k,v]) => `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`).join("")}
    </div>
  `;
}

function renderHierarchyForConference(confName){
  if (!window.d3){
    els.hierarchyDetail.classList.remove("muted");
    els.hierarchyDetail.textContent = "d3 failed to load. Check your network connection.";
    return;
  }

  const data = buildHierarchyData(confName);
  const wrap = els.hierarchyChart;
  wrap.innerHTML = "";

  const width = wrap.clientWidth || 800;
  const height = wrap.clientHeight || 600;

  const margin = { top: 18, right: 18, bottom: 18, left: 18 };

  const root = d3.hierarchy(data);
  const tree = d3.tree().nodeSize([26, 200]);
  tree(root);

  // Bounds
  let x0 = Infinity, x1 = -Infinity;
  root.each(d => { if (d.x < x0) x0 = d.x; if (d.x > x1) x1 = d.x; });
  const innerH = x1 - x0 + margin.top + margin.bottom;
  const innerW = (root.height + 1) * 200 + margin.left + margin.right;

  // Create a fixed-size viewport SVG and enable pan/zoom inside it.
  // (If the SVG grows to innerW/innerH, there's nothing to pan; you just get scrollbars.)
  const viewportW = Math.max(320, width);
  const viewportH = Math.max(240, height);

  const svg = d3.select(wrap).append("svg")
    .attr("width", viewportW)
    .attr("height", viewportH)
    .attr("viewBox", [0, 0, viewportW, viewportH].join(" "));

  // gZoom is the element that d3.zoom() transforms.
  const gZoom = svg.append("g").attr("class", "hzoom");

  // g is the tree content in its own coordinate system.
  const g = gZoom.append("g")
    .attr("transform", `translate(${margin.left},${margin.top - x0})`);

  g.append("g")
    .selectAll("path")
    .data(root.links())
    .join("path")
    .attr("class", "hlink")
    .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

  const node = g.append("g")
    .selectAll("g")
    .data(root.descendants())
    .join("g")
    .attr("class", d => (d.children ? "hnode" : "hnode hnode--leaf"))
    .attr("transform", d => `translate(${d.y},${d.x})`);

  node.append("circle").attr("r", 6);

  node.append("text")
    .attr("dy", "0.32em")
    .attr("x", d => d.children ? -10 : 10)
    .attr("text-anchor", d => d.children ? "end" : "start")
    .text(d => d.data.name);

  function setSelected(sel){
    g.selectAll(".hnode").classed("hnode--selected", d => d === sel);
  }

  node.on("click", (event, d) => {
    event.stopPropagation();
    hierarchyPinned = d;
    setSelected(d);

    if (d.data.kind === "person" && d.data._row){
      renderHierarchyRow(d.data._row);
    } else {
      els.hierarchyDetail.classList.remove("muted");
      els.hierarchyDetail.innerHTML = `
        <div style="font-weight:750; font-size:14px;">${escapeHtml(d.data.name)}</div>
        <div class="hint" style="margin-top:8px;">${d.data.kind === "position" ? "Position group" : "Conference root"}</div>
      `;
    }
  });

  // Prevent pan from starting when interacting with nodes (keeps clicks crisp).
  node.on("mousedown.zoom", (event) => event.stopPropagation());
  node.on("touchstart.zoom", (event) => event.stopPropagation());

  svg.on("click", () => {
    hierarchyPinned = null;
    setSelected(null);
    els.hierarchyDetail.classList.add("muted");
    els.hierarchyDetail.textContent = "Click a node to pin details here.";
  });

  // Pan + zoom: drag to pan; wheel/pinch to zoom.
  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => {
      gZoom.attr("transform", event.transform);
    });

  svg.call(zoom).on("dblclick.zoom", null);

  // Fit-to-view on initial render (cap at 1.0 so text stays readable).
  const fitScale = Math.min(viewportW / innerW, viewportH / innerH, 1);
  const fitX = (viewportW - innerW * fitScale) / 2;
  const fitY = (viewportH - innerH * fitScale) / 2;
  svg.call(zoom.transform, d3.zoomIdentity.translate(fitX, fitY).scale(fitScale));

  // Default summary
  els.hierarchyDetail.classList.remove("muted");
  els.hierarchyDetail.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(confName)}</div>
    <div class="hint" style="margin-top:8px;">${root.leaves().length.toLocaleString()} people • depth ${root.height} • top-level ${(root.children || []).length} nodes.</div>
  `;
}


function exportView(){
  const blob = new Blob([JSON.stringify(lastFiltered, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sda_timeline_view.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function loadManifest(){
  const res = await fetch("manifest.json");
  if (!res.ok) throw new Error("Could not load manifest.json");
  return res.json();
}

function loadCsv(file){
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: (err) => reject(err)
    });
  });
}

function normalizeRow(r, year){
  // Ensure consistent fields even if some datasets vary later
  const out = { ...r };
  if (!out.yearbook_year && year) out.yearbook_year = year;
  // Coerce numeric-like
  if (out.page !== undefined) out.page = safe(out.page) ? Number(out.page) : out.page;
  if (out.yearbook_year !== undefined) out.yearbook_year = safe(out.yearbook_year) ? Number(out.yearbook_year) : out.yearbook_year;
  return out;
}

async function reloadAllEnabled(){
  allRows = [];
  const enabled = manifest.datasets.filter(d => enabledYears.has(d.year));
  const loads = enabled.map(async d => {
    const rows = await loadCsv(d.file);
    return rows.map(r => normalizeRow(r, d.year));
  });

  const chunks = await Promise.all(loads);
  allRows = chunks.flat();

  hydrateFilterOptions();
  applyFilters();

  // If hierarchy modal is open, refresh its conference list
  if (els.hierarchyModal && els.hierarchyModal.getAttribute('aria-hidden') === 'false'){
    const keep = els.hierarchyConferenceSelect.value;
    openHierarchy();
    if (keep) els.hierarchyConferenceSelect.value = keep;
    if (els.hierarchyConferenceSelect.value) renderHierarchyForConference(els.hierarchyConferenceSelect.value);
  }
}

function renderDatasetList(){
  const html = manifest.datasets.map(ds => {
    const on = enabledYears.has(ds.year);
    return `
      <div class="datasetPill" data-year="${ds.year}">
        <div class="datasetPill__left">
          <div class="datasetPill__title">${escapeHtml(ds.label || String(ds.year))}</div>
          <div class="datasetPill__meta">${escapeHtml(ds.file)}</div>
        </div>
        <div class="datasetPill__right">
          <span class="toggle" aria-label="toggle">${on ? "✓" : ""}</span>
        </div>
      </div>
    `;
  }).join("");

  els.datasetList.innerHTML = html;

  els.datasetList.querySelectorAll(".datasetPill").forEach(el => {
    el.addEventListener("click", async () => {
      const y = Number(el.getAttribute("data-year"));
      if (enabledYears.has(y)) enabledYears.delete(y);
      else enabledYears.add(y);

      renderDatasetList();
      await reloadAllEnabled();
    });
  });
}

function initTimeline(){
  const container = els.timeline;

  items = new vis.DataSet([]);
  const groups = new vis.DataSet([]);

  const options = {
    stack: true,
    maxHeight: "640px",
    zoomMin: 1000 * 60 * 60 * 24 * 365 * 0.8,    // ~0.8 year
    zoomMax: 1000 * 60 * 60 * 24 * 365 * 200,    // ~200 years
    horizontalScroll: true,
    verticalScroll: true,
    zoomKey: "ctrlKey",
    margin: { item: 10, axis: 10 },
    tooltip: { followMouse: true },
  };

  timeline = new vis.Timeline(container, items, groups, options);

  timeline.on("select", (props) => {
    const id = props.items && props.items[0];
    if (!id) return renderDetail(null);
    const it = items.get(id);
    renderDetail(it?._row || null);
  });

  // Start centered around earliest year
  const start = new Date(1880, 0, 1);
  const end = new Date(1890, 0, 1);
  timeline.setWindow(start, end, { animation: false });
}

async function main(){
  els.datasetList = $("datasetList");
  els.searchInput = $("searchInput");
  els.regionSelect = $("regionSelect");
  els.confSelect = $("confSelect");
  els.positionSelect = $("positionSelect");
els.genderSelect = $("genderSelect");
els.orgSelect = $("orgSelect");
  els.yearMin = $("yearMin");
  els.yearMax = $("yearMax");
  els.applyBtn = $("applyBtn");
  els.resetBtn = $("resetBtn");
  els.detailCard = $("detailCard");
  els.hierarchyBtn = $("hierarchyBtn");
  els.exportBtn = $("exportBtn");
  els.hierarchyModal = $("hierarchyModal");
  els.hierarchyBackdrop = $("hierarchyBackdrop");
  els.hierarchyCloseBtn = $("hierarchyCloseBtn");
  els.hierarchyConferenceSelect = $("hierarchyConferenceSelect");
  els.hierarchyGroupingSelect = $("hierarchyGroupingSelect");
  els.hierarchyChart = $("hierarchyChart");
  els.hierarchyDetail = $("hierarchyDetail");

  els.statLoaded = $("statLoaded");
  els.statShown = $("statShown");
  els.statYears = $("statYears");
  els.timeline = $("timeline");

  initTimeline();

  manifest = await loadManifest();

  // Enable all datasets by default
  enabledYears = new Set(manifest.datasets.map(d => d.year));

  renderDatasetList();
  await reloadAllEnabled();

  els.applyBtn.addEventListener("click", applyFilters);
  els.resetBtn.addEventListener("click", resetFilters);

  els.hierarchyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    openHierarchy();
  });

  els.hierarchyBackdrop.addEventListener("click", closeHierarchy);
  els.hierarchyCloseBtn.addEventListener("click", closeHierarchy);
  els.hierarchyConferenceSelect.addEventListener("change", () => {
    const c = safe(els.hierarchyConferenceSelect.value);
    if (!c) return clearHierarchy();
    renderHierarchyForConference(c);
  });

  els.hierarchyGroupingSelect.addEventListener("change", () => {
    const c = safe(els.hierarchyConferenceSelect.value);
    if (!c) return;
    renderHierarchyForConference(c);
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && els.hierarchyModal.getAttribute("aria-hidden") === "false") closeHierarchy();
  });

  els.exportBtn.addEventListener("click", (e) => {
    e.preventDefault();
    exportView();
  });

  // Small UX: apply filters on enter in search
  els.searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") applyFilters();
  });
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    alert("Failed to load the app. Open the console for details.\n\n" + err.message);
  });
});

