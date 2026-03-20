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

  attachTabHandlers();
  attachSummaryHandlers();
}

window.addEventListener("DOMContentLoaded", () => {
  main().catch(err => {
    console.error(err);
    alert("Failed to load the app. Open the console for details.\n\n" + err.message);
  });
});




/* ----------------------------
   Summary tab
-----------------------------*/

let summaryRefreshTimer = null;
let summaryChartFrame = 0;

function normGender(g){
  const s = safe(g).toLowerCase();
  if (!s) return "";
  if (s.startsWith("f")) return "female";
  if (s.startsWith("m")) return "male";
  return s;
}

function personId(r){
  const n = safe(r.name);
  if (n) return n.toLowerCase();
  const combo = [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim();
  return combo ? combo.toLowerCase() : "";
}

function personLabel(r){
  return safe(r.name) || [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim() || "(unnamed)";
}

function summaryOrgLabel(r){
  return safe(r.organization) || safe(r.institution_name);
}

function tokenizeQuery(query){
  return safe(query).toLowerCase().split(/\s+/).filter(Boolean);
}

function matchesAllTerms(text, query){
  const terms = Array.isArray(query) ? query : tokenizeQuery(query);
  if (!terms.length) return true;
  const haystack = safe(text).toLowerCase();
  return terms.every(term => haystack.includes(term));
}

function summaryHaystack(r){
  return [
    safe(r.name),
    safe(r.last_name),
    safe(r.prefix),
    safe(r.suffix),
    safe(r.position),
    safe(r.position_information),
    summaryOrgLabel(r),
    safe(r.group),
    safe(r.conference),
    safe(r.location),
    safe(r.region),
    safe(r.institution_name),
  ].filter(Boolean).join(" ");
}

const NON_PERSON_NAME_RE = /\b(association|committee|conference|mission|society|college|school|department|board|church|publishing|union|home|sanitarium|academy|institute|office|company|press|hospital|tract|sabbath|division|corporation|committee of|secretary office)\b/i;

function isLikelyNamedIndividual(r){
  const explicit = safe(r.name);
  const fallback = [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim();
  const candidate = explicit || fallback;
  if (!candidate) return false;
  if (safe(r.last_name)) return true;
  if (NON_PERSON_NAME_RE.test(candidate)) return false;
  return /[a-z]/i.test(candidate);
}

function formatPercent(value, digits = 1){
  return Number.isFinite(value) ? `${value.toFixed(digits)}%` : "–";
}

function setSelectOptions(selectEl, values, {includeAll=true, allLabel="All"} = {}){
  if (!selectEl) return;
  const cur = selectEl.value;
  selectEl.innerHTML = "";
  if (includeAll){
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = allLabel;
    selectEl.appendChild(opt);
  }
  for (const v of values){
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }
  if (cur && Array.from(selectEl.options).some(o => o.value === cur)) {
    selectEl.value = cur;
  } else if (!includeAll && values.length) {
    selectEl.value = values[0];
  } else {
    selectEl.value = "";
  }
}

function getSummaryYearBounds(rows){
  const years = rows.map(r => Number(r.yearbook_year)).filter(Number.isFinite).sort((a,b)=>a-b);
  if (!years.length) return { min: 0, max: 0 };
  return { min: years[0], max: years[years.length - 1] };
}

function hydrateSummaryFilters(rows){
  const regions = uniqSorted(rows.map(r => safe(r.region)).filter(Boolean));
  const confs = uniqSorted(rows.map(r => safe(r.conference)).filter(Boolean));
  const orgs = uniqSorted(rows.map(r => summaryOrgLabel(r)).filter(Boolean));
  const groups = uniqSorted(rows.map(r => safe(r.group)).filter(Boolean));
  const roles = uniqSorted(rows.map(r => safe(r.position)).filter(Boolean));
  const genders = uniqSorted(rows.map(r => normGender(r.gender)).filter(Boolean));
  const bounds = getSummaryYearBounds(rows);

  setSelectOptions($("sumRegion"), regions, { includeAll: true, allLabel: "All regions" });
  setSelectOptions($("sumConference"), confs, { includeAll: true, allLabel: "All conferences" });
  setSelectOptions($("sumOrganization"), orgs, { includeAll: true, allLabel: "All organizations" });
  setSelectOptions($("sumGroup"), groups, { includeAll: true, allLabel: "All groups" });
  setSelectOptions($("sumRole"), roles, { includeAll: true, allLabel: "All roles" });
  setSelectOptions($("sumGender"), genders, { includeAll: true, allLabel: "All genders" });

  const minEl = $("sumYearMin");
  const maxEl = $("sumYearMax");
  const currentMin = Number(minEl?.value);
  const currentMax = Number(maxEl?.value);

  if (minEl){
    minEl.min = String(bounds.min || 0);
    minEl.max = String(bounds.max || 0);
    minEl.value = Number.isFinite(currentMin) && currentMin ? String(Math.max(bounds.min, Math.min(bounds.max, currentMin))) : String(bounds.min || "");
  }
  if (maxEl){
    maxEl.min = String(bounds.min || 0);
    maxEl.max = String(bounds.max || 0);
    maxEl.value = Number.isFinite(currentMax) && currentMax ? String(Math.max(bounds.min, Math.min(bounds.max, currentMax))) : String(bounds.max || "");
  }
}

function readSummaryState(rows = allRows){
  const bounds = getSummaryYearBounds(rows);
  let yearMin = Number($("sumYearMin")?.value);
  let yearMax = Number($("sumYearMax")?.value);
  if (!Number.isFinite(yearMin)) yearMin = bounds.min;
  if (!Number.isFinite(yearMax)) yearMax = bounds.max;
  if (yearMin > yearMax) [yearMin, yearMax] = [yearMax, yearMin];

  return {
    search: safe($("sumSearch")?.value),
    region: safe($("sumRegion")?.value),
    conference: safe($("sumConference")?.value),
    organization: safe($("sumOrganization")?.value),
    group: safe($("sumGroup")?.value),
    role: safe($("sumRole")?.value),
    roleDetail: safe($("sumRoleDetail")?.value),
    gender: safe($("sumGender")?.value),
    yearMin,
    yearMax,
  };
}

function filterSummaryRows(rows, state = readSummaryState(rows)){
  return rows.filter(r => {
    const year = Number(r.yearbook_year);
    if (!Number.isFinite(year) || year < state.yearMin || year > state.yearMax) return false;
    if (state.search && !matchesAllTerms(summaryHaystack(r), state.search)) return false;
    if (state.region && safe(r.region) !== state.region) return false;
    if (state.conference && safe(r.conference) !== state.conference) return false;
    if (state.organization && summaryOrgLabel(r) !== state.organization) return false;
    if (state.group && safe(r.group) !== state.group) return false;
    if (state.role && safe(r.position) !== state.role) return false;
    if (state.roleDetail && !matchesAllTerms(safe(r.position_information), state.roleDetail)) return false;
    if (state.gender && normGender(r.gender) !== state.gender) return false;
    return true;
  });
}

function computeSummaryStats(rows){
  const byYear = new Map();
  for (const r of rows){
    const year = Number(r.yearbook_year);
    if (!Number.isFinite(year)) continue;
    if (!byYear.has(year)) byYear.set(year, []);
    byYear.get(year).push(r);
  }

  const stats = Array.from(byYear.entries()).sort((a,b)=>a[0]-b[0]).map(([year, yearRows]) => {
    const people = new Map();
    for (const r of yearRows){
      if (!isLikelyNamedIndividual(r)) continue;
      const pid = personId(r);
      if (!pid) continue;
      if (!people.has(pid)) {
        people.set(pid, {
          id: pid,
          label: personLabel(r),
          female: false,
          roleCount: 0,
        });
      }
      const p = people.get(pid);
      p.roleCount += 1;
      if (normGender(r.gender) === "female") p.female = true;
      if (!p.label || p.label === "(unnamed)") p.label = personLabel(r);
    }

    const namedIndividuals = people.size;
    const women = Array.from(people.values()).filter(p => p.female).length;
    const gt5 = Array.from(people.values()).filter(p => p.roleCount > 5).length;
    const conferences = new Set(yearRows.map(r => safe(r.conference)).filter(Boolean)).size;

    return {
      year,
      matchingRows: yearRows.length,
      namedIndividuals,
      women,
      womenPct: namedIndividuals ? (women / namedIndividuals) * 100 : NaN,
      gt5,
      gt5Pct: namedIndividuals ? (gt5 / namedIndividuals) * 100 : NaN,
      conferences,
      people,
      rows: yearRows,
    };
  });

  return stats;
}

function computeAggregateMetrics(stats){
  let totalNamed = 0;
  let totalWomen = 0;
  let totalGt5 = 0;
  for (const s of stats){
    totalNamed += s.namedIndividuals;
    totalWomen += s.women;
    totalGt5 += s.gt5;
  }
  return {
    totalNamed,
    totalWomen,
    totalGt5,
    womenPct: totalNamed ? (totalWomen / totalNamed) * 100 : NaN,
    gt5Pct: totalNamed ? (totalGt5 / totalNamed) * 100 : NaN,
  };
}

function setMetricPills(prefix, series, valueKey){
  const latestEl = $(`${prefix}Latest`);
  const peakEl = $(`${prefix}Peak`);
  const averageEl = $(`${prefix}Average`);
  const valid = series.filter(d => Number.isFinite(d[valueKey]));

  if (!valid.length){
    if (latestEl) latestEl.textContent = "Latest: –";
    if (peakEl) peakEl.textContent = "Peak: –";
    if (averageEl) averageEl.textContent = "Average: –";
    return;
  }

  const latest = valid[valid.length - 1];
  let peak = valid[0];
  let total = 0;
  for (const d of valid){
    total += d[valueKey];
    if (d[valueKey] > peak[valueKey]) peak = d;
  }
  const avg = total / valid.length;

  if (latestEl) latestEl.textContent = `Latest: ${formatPercent(latest[valueKey])} (${latest.year})`;
  if (peakEl) peakEl.textContent = `Peak: ${formatPercent(peak[valueKey])} (${peak.year})`;
  if (averageEl) averageEl.textContent = `Average: ${formatPercent(avg)}`;
}

function nicePercentCeiling(maxValue){
  if (!Number.isFinite(maxValue) || maxValue <= 0) return 10;
  if (maxValue <= 5) return 5;
  if (maxValue <= 10) return 10;
  if (maxValue <= 20) return 20;
  if (maxValue <= 25) return 25;
  if (maxValue <= 50) return 50;
  if (maxValue <= 75) return 75;
  return 100;
}

function buildPercentTicks(maxPercent){
  if (maxPercent <= 5) return [0, 1, 2, 3, 4, 5];
  if (maxPercent <= 10) return [0, 2, 4, 6, 8, 10];
  if (maxPercent <= 20) return [0, 5, 10, 15, 20];
  if (maxPercent <= 25) return [0, 5, 10, 15, 20, 25];
  if (maxPercent <= 50) return [0, 10, 20, 30, 40, 50];
  if (maxPercent <= 75) return [0, 15, 30, 45, 60, 75];
  return [0, 20, 40, 60, 80, 100];
}

function pickYearTicks(series){
  if (series.length <= 8) return series.map(d => d.year);
  const target = 7;
  const step = Math.max(1, Math.ceil((series.length - 1) / (target - 1)));
  const ticks = [];
  for (let i = 0; i < series.length; i += step) ticks.push(series[i].year);
  const last = series[series.length - 1].year;
  if (!ticks.includes(last)) ticks.push(last);
  return ticks;
}

function renderPercentChart(elId, series, valueKey, opts = {}){
  const el = $(elId);
  if (!el) return;

  const label = opts.label || "percent";
  const lineCls = opts.alt ? "summaryChart__line summaryChart__line--alt" : "summaryChart__line";
  const areaCls = opts.alt ? "summaryChart__area summaryChart__area--alt" : "summaryChart__area";
  const pointCls = opts.alt ? "summaryChart__point summaryChart__point--alt" : "summaryChart__point";
  const height = opts.height || 420;

  const valid = series.filter(d => Number.isFinite(d[valueKey]));
  if (!valid.length){
    el.innerHTML = `<div class="summaryChart__empty">No yearly values are available for this filtered slice.</div>`;
    return;
  }

  const width = Math.max(el.clientWidth || 0, 320);
  const margin = { top: 18, right: 20, bottom: 42, left: 56 };
  const innerWidth = Math.max(120, width - margin.left - margin.right);
  const innerHeight = Math.max(140, height - margin.top - margin.bottom);

  const years = valid.map(d => d.year);
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);
  const maxPercent = nicePercentCeiling(Math.max(...valid.map(d => d[valueKey]), 0));
  const yTicks = buildPercentTicks(maxPercent);
  const xTicks = pickYearTicks(valid);

  const x = year => maxYear === minYear ? margin.left + innerWidth / 2 : margin.left + ((year - minYear) / (maxYear - minYear)) * innerWidth;
  const y = value => margin.top + innerHeight - (value / maxPercent) * innerHeight;
  const baselineY = y(0);

  const coords = valid.map(d => ({ ...d, cx: x(d.year), cy: y(d[valueKey]) }));
  const linePath = coords.length > 1 ? coords.map((d, i) => `${i === 0 ? "M" : "L"} ${d.cx.toFixed(2)} ${d.cy.toFixed(2)}`).join(" ") : "";
  const areaPath = coords.length > 1 ? [
    `M ${coords[0].cx.toFixed(2)} ${baselineY.toFixed(2)}`,
    `L ${coords[0].cx.toFixed(2)} ${coords[0].cy.toFixed(2)}`,
    ...coords.slice(1).map(d => `L ${d.cx.toFixed(2)} ${d.cy.toFixed(2)}`),
    `L ${coords[coords.length - 1].cx.toFixed(2)} ${baselineY.toFixed(2)}`,
    "Z"
  ].join(" ") : "";

  const yGrid = yTicks.map(value => `<g class="summaryChart__grid"><line x1="${margin.left}" y1="${y(value).toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${y(value).toFixed(2)}"></line></g>`).join("");
  const yAxis = yTicks.map(value => `<g class="summaryChart__axis"><text x="${(margin.left - 10).toFixed(2)}" y="${(y(value) + 4).toFixed(2)}" text-anchor="end">${escapeHtml(formatPercent(value, value % 1 ? 1 : 0))}</text></g>`).join("");
  const xAxis = xTicks.map(year => `<g class="summaryChart__axis"><line x1="${x(year).toFixed(2)}" y1="${margin.top}" x2="${x(year).toFixed(2)}" y2="${baselineY.toFixed(2)}"></line><text x="${x(year).toFixed(2)}" y="${(baselineY + 26).toFixed(2)}" text-anchor="middle">${escapeHtml(String(year))}</text></g>`).join("");
  const points = coords.map(d => {
    const title = opts.title ? opts.title(d) : `${d.year}: ${formatPercent(d[valueKey])}`;
    return `<circle class="${pointCls}" cx="${d.cx.toFixed(2)}" cy="${d.cy.toFixed(2)}" r="4.5"><title>${escapeHtml(title)}</title></circle>`;
  }).join("");

  const latest = coords[coords.length - 1];
  const latestLabel = latest ? `<text class="summaryChart__label" x="${(latest.cx - 8).toFixed(2)}" y="${Math.max(margin.top + 14, latest.cy - 10).toFixed(2)}" text-anchor="end">${escapeHtml(formatPercent(latest[valueKey]))}</text>` : "";

  el.innerHTML = `
    <div class="summaryChart">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)} over time">
        ${yGrid}
        ${xAxis}
        ${yAxis}
        <g class="summaryChart__axis"><line x1="${margin.left}" y1="${baselineY.toFixed(2)}" x2="${(margin.left + innerWidth).toFixed(2)}" y2="${baselineY.toFixed(2)}"></line></g>
        ${areaPath ? `<path class="${areaCls}" d="${areaPath}"></path>` : ""}
        ${linePath ? `<path class="${lineCls}" d="${linePath}"></path>` : ""}
        ${points}
        ${latestLabel}
      </svg>
      <div class="summaryChart__footnote">${escapeHtml(opts.footnote || "Percent of unique named individuals in each year.")}</div>
    </div>
  `;
}

function renderPerYearTable(stats){
  const table = $("perYearTable");
  const tbody = table ? table.querySelector("tbody") : null;
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const s of stats){
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.year}</td>
      <td>${s.matchingRows.toLocaleString()}</td>
      <td>${s.namedIndividuals.toLocaleString()}</td>
      <td>${s.women.toLocaleString()}</td>
      <td>${formatPercent(s.womenPct)}</td>
      <td>${s.gt5.toLocaleString()}</td>
      <td>${formatPercent(s.gt5Pct)}</td>
      <td>${s.conferences.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderSummaryKpis(filteredRows, stats){
  const metrics = computeAggregateMetrics(stats);
  const uniqueNames = new Set();
  for (const r of filteredRows){
    if (!isLikelyNamedIndividual(r)) continue;
    const pid = personId(r);
    if (pid) uniqueNames.add(pid);
  }

  const elPeople = $("sumUniquePeople");
  const elRows = $("sumRows");
  const elOverall = $("sumOverallPct");
  const elOverallGt5 = $("sumOverallGt5Pct");
  if (elPeople) elPeople.textContent = uniqueNames.size.toLocaleString();
  if (elRows) elRows.textContent = filteredRows.length.toLocaleString();
  if (elOverall) elOverall.textContent = formatPercent(metrics.womenPct);
  if (elOverallGt5) elOverallGt5.textContent = formatPercent(metrics.gt5Pct);

  const ul = $("sumNameSample");
  if (ul){
    ul.innerHTML = "";
    const names = Array.from(new Set(filteredRows.filter(isLikelyNamedIndividual).map(personLabel).filter(Boolean))).sort((a,b)=>a.localeCompare(b)).slice(0, 25);
    if (!names.length){
      const li = document.createElement("li");
      li.textContent = "(no names for this filter)";
      ul.appendChild(li);
    } else {
      for (const n of names){
        const li = document.createElement("li");
        li.textContent = n;
        ul.appendChild(li);
      }
    }
  }
}

function renderSummary(){
  setSummaryLoadedStat();
  hydrateSummaryFilters(allRows);
  const state = readSummaryState(allRows);
  const filteredRows = filterSummaryRows(allRows, state);
  const stats = computeSummaryStats(filteredRows);

  setMetricPills("womenPct", stats, "womenPct");
  setMetricPills("gt5Pct", stats, "gt5Pct");

  renderPercentChart("womenPctChart", stats, "womenPct", {
    label: "Percentage of named individuals identified as women over time",
    height: 460,
    title: d => `${d.year}: ${formatPercent(d.womenPct)} (${d.women} women of ${d.namedIndividuals} named individuals)`,
    footnote: "Percentage = unique named individuals identified as women divided by all unique named individuals in each year. One person with many roles is still counted once per year.",
  });

  renderPercentChart("gt5RolesChart", stats, "gt5Pct", {
    label: "Percentage of named individuals with more than five roles over time",
    height: 320,
    alt: true,
    title: d => `${d.year}: ${formatPercent(d.gt5Pct)} (${d.gt5} people with >5 roles of ${d.namedIndividuals} named individuals)`,
    footnote: "For each year, this uses the filtered slice and counts whether each unique named individual has more than five matching role rows that year.",
  });

  renderPerYearTable(stats);
  renderSummaryKpis(filteredRows, stats);
}

function queueSummaryRender(){
  if (summaryChartFrame) cancelAnimationFrame(summaryChartFrame);
  summaryChartFrame = window.requestAnimationFrame(() => {
    summaryChartFrame = 0;
    renderSummary();
  });
}

function attachSummaryHandlers(){
  const ids = ["sumSearch", "sumRegion", "sumConference", "sumOrganization", "sumGroup", "sumRole", "sumRoleDetail", "sumGender", "sumYearMin", "sumYearMax"];
  for (const id of ids){
    const el = $(id);
    if (!el) continue;
    const evt = (el.tagName === "INPUT" && el.type !== "number") ? "input" : "change";
    el.addEventListener(evt, () => {
      if (summaryRefreshTimer) clearTimeout(summaryRefreshTimer);
      summaryRefreshTimer = window.setTimeout(queueSummaryRender, evt === "input" ? 120 : 0);
    });
    if (evt !== "change") {
      el.addEventListener("change", queueSummaryRender);
    }
  }

  const resetBtn = $("sumResetBtn");
  if (resetBtn){
    resetBtn.addEventListener("click", () => {
      if ($("sumSearch")) $("sumSearch").value = "";
      if ($("sumRegion")) $("sumRegion").value = "";
      if ($("sumConference")) $("sumConference").value = "";
      if ($("sumOrganization")) $("sumOrganization").value = "";
      if ($("sumGroup")) $("sumGroup").value = "";
      if ($("sumRole")) $("sumRole").value = "";
      if ($("sumRoleDetail")) $("sumRoleDetail").value = "";
      if ($("sumGender")) $("sumGender").value = "";
      const bounds = getSummaryYearBounds(allRows);
      if ($("sumYearMin")) $("sumYearMin").value = bounds.min ? String(bounds.min) : "";
      if ($("sumYearMax")) $("sumYearMax").value = bounds.max ? String(bounds.max) : "";
      queueSummaryRender();
    });
  }

  window.addEventListener("resize", () => {
    if ($("tabSummary")?.classList.contains("active")) queueSummaryRender();
  });
}

function setSummaryLoadedStat(){
  const el = $("sumStatLoaded");
  if (!el) return;
  const years = Array.from(enabledYears).sort((a,b)=>a-b);
  const label = years.length ? `${years[0]}–${years[years.length - 1]}` : "–";
  el.textContent = `Loaded: ${allRows.length.toLocaleString()} rows (${label})`;
}

function attachTabHandlers(){
  const btnSum = $("tabBtnSummary");
  const btnTime = $("tabBtnTimeline");
  const paneSum = $("tabSummary");
  const paneTime = $("tabTimeline");
  if (!btnSum || !btnTime || !paneSum || !paneTime) return;

  function activate(which){
    const sumOn = which === "summary";
    btnSum.classList.toggle("active", sumOn);
    btnTime.classList.toggle("active", !sumOn);
    paneSum.classList.toggle("active", sumOn);
    paneTime.classList.toggle("active", !sumOn);

    if (sumOn){
      queueSummaryRender();
    } else if (timeline){
      setTimeout(() => { try { timeline.redraw(); } catch(e){} }, 50);
    }
  }

  btnSum.addEventListener("click", (e) => { e.preventDefault(); activate("summary"); });
  btnTime.addEventListener("click", (e) => { e.preventDefault(); activate("timeline"); });
}

function refreshSummary(){
  queueSummaryRender();
}

const __reloadAllEnabled = reloadAllEnabled;
reloadAllEnabled = async function(){
  await __reloadAllEnabled();
  refreshSummary();
};
