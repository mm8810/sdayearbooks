let manifest = null;
let allRows = [];      // raw rows across all enabled datasets
let enabledYears = new Set();
let timeline = null;
let items = null;
let lastFiltered = [];
let lastTimelineEntities = [];
let timelineInitialWindowSet = false;

let hierarchyPinned = null;
let hierarchyYear = null;
let normalizationConfig = null;

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

const CONFERENCE_TYPE_PRIORITY = [
  "Union Conference",
  "Union Mission",
  "Mission Field",
  "Conference",
  "Mission",
  "Union",
];

const ORGANIZATION_TYPE_PRIORITY = [
  "Conference Tract Society",
  "Conference Association",
  "Conference Corporation",
  "Conference Agency",
  "Union Conference Association",
  "Union Conference",
  "Union Mission",
  "Tract and Missionary Society",
  "Tract Society Department",
  "Tract Society",
  "Health and Temperance Association",
  "Health and Temperance Society",
  "Religious Liberty Department",
  "Religious Liberty Association",
  "Religious Liberty Bureau",
  "Medical Missionary and Benevolent Association",
  "Publishing Department",
  "Publishing Association",
  "Sabbath-School Department",
  "Sabbath-School Association",
  "Directory",
  "Department",
  "Committee",
  "Association",
  "Society",
  "Conference",
  "Mission",
  "Union",
  "School",
  "College",
  "Academy",
  "Hospital",
  "Sanitarium",
  "Office",
  "Press",
];

function normalizeSpacing(str){
  return safe(str).replace(/[’`]/g, "'").replace(/\s+/g, " ").trim();
}

function normalizeKey(str){
  return normalizeSpacing(str).toLowerCase();
}

function mergeNormalizationMaps(raw = {}){
  return Object.fromEntries(
    Object.entries(raw || {}).map(([key, value]) => [normalizeKey(key), normalizeSpacing(value)])
  );
}

async function loadNormalizationConfig(){
  normalizationConfig = {
    conferenceExactAliases: {},
    conferenceFamilyAliases: {},
    organizationExactAliases: {},
    organizationFamilyAliases: {},
    organizationTypeAliases: {},
    regionExactAliases: {},
  };

  try {
    const res = await fetch("normalization.json");
    if (!res.ok) return;
    const raw = await res.json();
    normalizationConfig = {
      conferenceExactAliases: mergeNormalizationMaps(raw.conference_exact_aliases),
      conferenceFamilyAliases: mergeNormalizationMaps(raw.conference_family_aliases),
      organizationExactAliases: mergeNormalizationMaps(raw.organization_exact_aliases),
      organizationFamilyAliases: mergeNormalizationMaps(raw.organization_family_aliases),
      organizationTypeAliases: mergeNormalizationMaps(raw.organization_type_aliases),
      regionExactAliases: mergeNormalizationMaps(raw.region_exact_aliases),
    };
  } catch (err) {
    console.warn("normalization.json could not be loaded; using built-in defaults.", err);
  }
}

function escapeRegExp(str){
  return String(str).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function titleCaseWords(str){
  return normalizeSpacing(str).split(" ").filter(Boolean).map(word => {
    const lower = word.toLowerCase();
    if (/^[A-Z0-9'.-]+$/.test(word) && word !== word.toLowerCase()) return word;
    if (lower === "and" || lower === "of" || lower === "the") return lower;
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  }).join(" ");
}

function applyAlias(map, value){
  const normalized = normalizeKey(value);
  return map?.[normalized] || normalizeSpacing(value);
}

function normalizeRegionName(raw){
  const clean = normalizeSpacing(raw);
  if (!clean) return "Unknown";
  return applyAlias(normalizationConfig?.regionExactAliases, clean);
}

function normalizeOrganizationGuideForms(raw){
  let clean = normalizeSpacing(raw);
  if (!clean) return "";
  clean = clean.replace(/\bAssn\.\b/gi, "Association");
  clean = clean.replace(/^The\s+/i, "");

  const corporateSuffixes = [
    /^(.*? Conference Association)(?: of (?:the )?Seventh-day Adventists(?:, Incorporated)?| of the S\.?\s*D\.?\s*A\.?(?:\.|\s*Church)?| of S\.?\s*D\.?\s*A\.?(?:\.|\s*Church)?| of S\.D\.A\.)$/i,
    /^(.*? Conference Corporation)(?: of (?:the )?Seventh-day Adventists)$/i,
    /^(.*? Conference Agency)(?: of (?:the )?Seventh-day Adventists| of S\.?\s*D\.?\s*A\.?(?:\.|\s*Incorporated)?)$/i,
  ];

  for (const pattern of corporateSuffixes){
    const match = clean.match(pattern);
    if (match) return normalizeSpacing(match[1]);
  }

  return clean;
}

function extractType(label, candidates){
  const clean = normalizeSpacing(label);
  const orderedCandidates = Array.from(new Set(candidates))
    .sort((a, b) => b.length - a.length || a.localeCompare(b));
  for (const type of orderedCandidates){
    const re = new RegExp(`\\s+${escapeRegExp(type)}$`, "i");
    if (re.test(clean)) {
      return {
        family: clean.replace(re, "").trim(),
        type,
      };
    }
  }
  return { family: clean, type: "" };
}

function normalizeConferenceEntity(raw){
  let clean = normalizeSpacing(raw);
  if (!clean) return { raw: "", family: "", type: "", canonical: "" };
  clean = applyAlias(normalizationConfig?.conferenceExactAliases, clean);
  const parts = extractType(clean, CONFERENCE_TYPE_PRIORITY);
  const family = applyAlias(normalizationConfig?.conferenceFamilyAliases, parts.family || clean);
  const canonical = normalizeSpacing(`${family}${parts.type ? ` ${parts.type}` : ""}`);
  return {
    raw: clean,
    family,
    type: parts.type,
    canonical: canonical || family,
  };
}

function normalizeOrganizationEntity(raw){
  let clean = normalizeSpacing(raw);
  if (!clean) return { raw: "", family: "", type: "", canonical: "" };
  clean = normalizeOrganizationGuideForms(clean);
  clean = applyAlias(normalizationConfig?.organizationExactAliases, clean);
  const parts = extractType(clean, ORGANIZATION_TYPE_PRIORITY);
  const type = parts.type ? applyAlias(normalizationConfig?.organizationTypeAliases, parts.type) : "";
  let family = parts.type ? parts.family : (parts.family || clean);
  if (family) {
    family = applyAlias(normalizationConfig?.organizationFamilyAliases, family);
    family = titleCaseWords(family);
  }
  const canonical = normalizeSpacing([family, type].filter(Boolean).join(" "));
  return {
    raw: clean,
    family,
    type,
    canonical: canonical || family || type,
  };
}

function pluralize(count, singular, plural = `${singular}s`){
  return `${count.toLocaleString()} ${count === 1 ? singular : plural}`;
}

function renderHierarchyDetailCard(title, fields, hint = ""){
  els.hierarchyDetail.classList.remove("muted");
  els.hierarchyDetail.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(title)}</div>
    <div class="kv">
      ${fields
        .filter(([, value]) => safe(value))
        .map(([label, value]) => `<div class="k">${escapeHtml(label)}</div><div class="v">${escapeHtml(value)}</div>`)
        .join("")}
    </div>
    ${safe(hint) ? `<div class="hint" style="margin-top:8px;">${escapeHtml(hint)}</div>` : ""}
  `;
}

function slugifyToken(str){
  return safe(str).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "default";
}

function summarizeValues(values, limit = 6){
  const cleaned = uniqSorted(values.map(v => safe(v)).filter(Boolean));
  if (!cleaned.length) return "";
  if (cleaned.length <= limit) return cleaned.join(" • ");
  return `${cleaned.slice(0, limit).join(" • ")} • +${cleaned.length - limit} more`;
}

function regionLabelForRow(row){
  return safe(row.region_normalized) || safe(row.region) || "Unknown";
}

function makeEntityKey(scope, label){
  return `${scope}::${label}`;
}

function parseEntityKey(key){
  const [scope = "", ...rest] = safe(key).split("::");
  return {
    scope,
    label: rest.join("::"),
  };
}

function inferConferenceEntityType(row){
  const explicit = safe(row.conference_type);
  if (explicit) return explicit;
  const label = safe(row.conference_canonical) || safe(row.conference);
  if (!label) return "";
  if (label === "General Conference") return "General Conference";
  return "Conference";
}

function timelineFamilyOrder(family){
  if (family === "General Conference") return -1;
  return 0;
}

function conferenceMatchesTimelineFilters(entity){
  const type = safe(els.entityTypeSelect?.value);
  if (type && entity.type !== type) return false;
  return true;
}

function splitIntoYearRanges(years){
  if (!years.length) return [];
  const ranges = [];
  let start = years[0];
  let prev = years[0];
  for (let i = 1; i < years.length; i += 1){
    const year = years[i];
    if (year === prev + 1){
      prev = year;
      continue;
    }
    ranges.push({ start, end: prev });
    start = year;
    prev = year;
  }
  ranges.push({ start, end: prev });
  return ranges;
}

function buildHierarchyEntities(rows){
  const byKey = new Map();
  for (const row of rows){
    const label = safe(row.conference_canonical) || safe(row.conference);
    if (!label) continue;
    const entity = {
      key: makeEntityKey("conference", label),
      label,
      family: safe(row.conference_family) || label,
      type: inferConferenceEntityType(row),
    };
    if (!conferenceMatchesTimelineFilters(entity)) continue;
    if (!byKey.has(entity.key)) {
      byKey.set(entity.key, {
        key: entity.key,
        label: entity.label,
        family: entity.family,
        familyOrder: timelineFamilyOrder(entity.family),
        scope: "conference",
        type: entity.type,
      });
    }
  }
  return Array.from(byKey.values()).sort((a, b) =>
    a.familyOrder - b.familyOrder ||
    a.family.localeCompare(b.family) ||
    a.label.localeCompare(b.label)
  );
}

function buildEntityTimelineItems(rows){
  const byConference = new Map();

  for (const row of rows){
    const year = Number(row.yearbook_year);
    if (!Number.isFinite(year)) continue;

    const conferenceLabel = safe(row.conference_canonical) || safe(row.conference);
    const relatedOrganization = safe(row.organization_canonical) || safe(row.organization) || safe(row.institution_name);
    if (!conferenceLabel) continue;

    const entry = {
      key: makeEntityKey("conference", conferenceLabel),
      label: conferenceLabel,
      scope: "conference",
      type: inferConferenceEntityType(row),
      family: safe(row.conference_family) || conferenceLabel,
    };
    if (!conferenceMatchesTimelineFilters(entry)) continue;

    if (!byConference.has(entry.key)) {
      byConference.set(entry.key, {
        ...entry,
        regions: new Set(),
        relatedOrganizations: new Set(),
        years: new Set(),
        rowsByYear: new Map(),
      });
    }

    const bucket = byConference.get(entry.key);
    bucket.regions.add(regionLabelForRow(row));
    if (relatedOrganization) bucket.relatedOrganizations.add(relatedOrganization);
    bucket.years.add(year);
    if (!bucket.rowsByYear.has(year)) bucket.rowsByYear.set(year, []);
    bucket.rowsByYear.get(year).push(row);
  }

  const itemObjs = [];
  lastTimelineEntities = [];
  for (const entry of byConference.values()){
    lastTimelineEntities.push({
      key: entry.key,
      label: entry.label,
      family: entry.family,
      scope: entry.scope,
      type: entry.type,
    });

    const years = Array.from(entry.years).sort((a, b) => a - b);
    const ranges = splitIntoYearRanges(years);

    for (const range of ranges){
      const segmentRowsByYear = {};
      for (let year = range.start; year <= range.end; year += 1){
        const rowsForYear = entry.rowsByYear.get(year);
        if (rowsForYear?.length) segmentRowsByYear[String(year)] = rowsForYear;
      }

      itemObjs.push({
        id: `${entry.key}-${range.start}-${range.end}`,
        content: entry.label,
        start: yearToDate(range.start),
        end: yearToDate(range.end + 1),
        type: "range",
        group: entry.label,
        className: `timelineItem timelineItem--scope-conference timelineItem--${slugifyToken(entry.type || "default")}`,
        title: `${entry.label}\nFamily: ${entry.family}\nType: ${entry.type || "Unspecified"}\nRegions: ${summarizeValues(Array.from(entry.regions), 8)}\nActive: ${range.start}–${range.end}`,
        _entityKey: entry.key,
        _label: entry.label,
        _scope: entry.scope,
        _entityType: entry.type,
        _family: entry.family,
        _regions: Array.from(entry.regions).sort((a, b) => a.localeCompare(b)),
        _relatedOrganizations: Array.from(entry.relatedOrganizations).sort((a, b) => a.localeCompare(b)),
        _startYear: range.start,
        _endYear: range.end,
        _rowsByYear: segmentRowsByYear,
      });
    }
  }

  lastTimelineEntities.sort((a, b) =>
    timelineFamilyOrder(a.family) - timelineFamilyOrder(b.family) ||
    a.family.localeCompare(b.family) ||
    a.label.localeCompare(b.label)
  );

  return itemObjs.sort((a, b) =>
    timelineFamilyOrder(a._family) - timelineFamilyOrder(b._family) ||
    a._family.localeCompare(b._family) ||
    a.content.localeCompare(b.content) ||
    a._startYear - b._startYear
  );
}

function timelineRowsForYear(item, year){
  if (!item) return [];
  return item._rowsByYear?.[String(year)] || [];
}

function countNamedPeople(rows){
  const people = new Set(
    rows
      .map(r => safe(r.name) || [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim())
      .filter(Boolean)
  );
  return people.size;
}

function renderTimelineDetail(item, year = null){
  if (!item){
    els.detailCard.classList.add("muted");
    els.detailCard.textContent = "Click a conference span on the timeline to inspect that conference at a specific year.";
    return;
  }

  const activeYears = item._startYear === item._endYear ? String(item._startYear) : `${item._startYear}–${item._endYear}`;
  const selectedYear = Number.isFinite(year) ? Math.max(item._startYear, Math.min(item._endYear, year)) : null;
  const yearRows = selectedYear !== null ? timelineRowsForYear(item, selectedYear) : [];
  const yearRegions = uniqSorted(yearRows.map(regionLabelForRow));
  const orgCounts = new Map();
  const confCounts = new Map();
  for (const row of yearRows){
    const organization = safe(row.organization_canonical) || safe(row.organization) || safe(row.institution_name) || "(No organization)";
    const conference = safe(row.conference_canonical) || safe(row.conference) || "(No conference)";
    orgCounts.set(organization, (orgCounts.get(organization) || 0) + 1);
    confCounts.set(conference, (confCounts.get(conference) || 0) + 1);
  }
  const relatedLabel = Array.from(orgCounts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 5)
    .map(([organization, count]) => `${organization} (${count})`);

  const fields = [
    ["Entity", item._label],
    ["Family", item._family],
    ["Scope", "conference"],
    ["Type", item._entityType || "Unspecified"],
    ["Active years", activeYears],
    ["Regions", summarizeValues(item._regions || [])],
    selectedYear !== null ? ["Selected year", String(selectedYear)] : null,
    selectedYear !== null ? ["Regions in year", summarizeValues(yearRegions)] : null,
    selectedYear !== null ? ["Rows in year", yearRows.length.toLocaleString()] : null,
    selectedYear !== null ? ["Named people in year", countNamedPeople(yearRows).toLocaleString()] : null,
    relatedLabel.length ? ["Organizations in year", relatedLabel.join(" • ")] : null,
  ].filter(Boolean);

  els.detailCard.classList.remove("muted");
  els.detailCard.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(item._label)}</div>
    <div class="kv">
      ${fields.map(([k,v]) => `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`).join("")}
    </div>
  `;
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
  const regions = uniqSorted(allRows.map(r => safe(r.region_normalized) || safe(r.region)));
  const confs = uniqSorted(allRows.map(r => safe(r.conference_canonical) || safe(r.conference)));
  const positions = uniqSorted(allRows.map(r => r.position));
  const genders = uniqSorted(allRows.map(r => r.gender));
  const entityTypes = uniqSorted(allRows.map(row => inferConferenceEntityType(row)).filter(Boolean));
  const orgs = uniqSorted(
    allRows
      .map(r => safe(r.organization_canonical) || safe(r.organization) || safe(r.institution_name))
      .filter(v => safe(v))
  );

  // Preserve current selection when possible
  const prevRegion = els.regionSelect.value;
  const prevConf = els.confSelect.value;
  const prevPos = els.positionSelect?.value || "";
  const prevGender = els.genderSelect?.value || "";
  const prevOrg = els.orgSelect?.value || "";
  const prevEntityType = els.entityTypeSelect?.value || "";

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

  if (els.entityTypeSelect){
    els.entityTypeSelect.innerHTML =
      '<option value="">All conference types</option>' +
      entityTypes.map(type => `<option value="${escapeHtml(type)}">${escapeHtml(type)}</option>`).join("");
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
  if (els.entityTypeSelect && entityTypes.includes(prevEntityType)) els.entityTypeSelect.value = prevEntityType;
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

    if (region && (safe(r.region_normalized) || safe(r.region)) !== region) return false;
    if (conf && (safe(r.conference_canonical) || safe(r.conference)) !== conf) return false;

    if (pos && safe(r.position) !== pos) return false;
    if (gender && safe(r.gender) !== gender) return false;

    if (orgSel){
      const orgLabel = safe(r.organization_canonical) || safe(r.organization) || safe(r.institution_name);
      if (safe(orgLabel) !== orgSel) return false;
    }

    return true;
  });

  lastFiltered = filtered;

  const itemObjs = buildEntityTimelineItems(filtered);
  items.clear();
  items.add(itemObjs);

  const presentConferences = uniqSorted(itemObjs.map(it => it.group));
  const groupMeta = new Map();
  itemObjs.forEach(item => {
    if (!groupMeta.has(item.group)) groupMeta.set(item.group, item._family);
  });
  const groups = presentConferences
    .sort((a, b) =>
      timelineFamilyOrder(groupMeta.get(a) || a) - timelineFamilyOrder(groupMeta.get(b) || b) ||
      String(groupMeta.get(a) || a).localeCompare(String(groupMeta.get(b) || b)) ||
      a.localeCompare(b)
    )
    .map((conference, index) => ({
      id: conference,
      content: conference,
      sortOrder: index,
    }));
  timeline.setGroups(groups);

  if (itemObjs.length){
    const minYear = Math.min(...itemObjs.map(item => item._startYear));
    const maxYear = Math.max(...itemObjs.map(item => item._endYear));
    const endYear = Math.min(maxYear + 1, minYear + 12);
    const minDate = new Date(minYear, 0, 1);
    const maxDate = new Date(maxYear + 1, 0, 1);
    timeline.setOptions({
      min: minDate,
      max: maxDate,
    });
    if (!timelineInitialWindowSet){
      timeline.setWindow(minDate, new Date(endYear, 0, 1), { animation: false });
      timelineInitialWindowSet = true;
    } else {
      const current = timeline.getWindow();
      const currentStart = current.start < minDate ? minDate : (current.start > maxDate ? minDate : current.start);
      const currentEnd = current.end > maxDate ? maxDate : (current.end < minDate ? maxDate : current.end);
      timeline.setWindow(currentStart, currentEnd, { animation: false });
    }
  }

  renderTimelineDetail(null);
  setStats();
}

function resetFilters(){
  els.searchInput.value = "";
  els.regionSelect.value = "";
  els.confSelect.value = "";
  if (els.positionSelect) els.positionSelect.value = "";
  if (els.genderSelect) els.genderSelect.value = "";
  if (els.orgSelect) els.orgSelect.value = "";
  if (els.entityTypeSelect) els.entityTypeSelect.value = "";
  els.yearMin.value = "1883";
  els.yearMax.value = "1921";
  applyFilters();
}


function openHierarchy(opts = {}){
  hierarchyYear = Number.isFinite(Number(opts.year)) ? Number(opts.year) : null;
  els.hierarchyModal.setAttribute("aria-hidden", "false");

  const entities = buildHierarchyEntities(lastFiltered);
  const requestedKey = safe(opts.entityKey);
  const prev = els.hierarchyConferenceSelect.value;

  els.hierarchyConferenceSelect.innerHTML =
    '<option value="">Select a conference</option>' +
    entities.map(entity => `<option value="${escapeHtml(entity.key)}">${escapeHtml(entity.label)}</option>`).join("");

  if (requestedKey && entities.some(entity => entity.key === requestedKey)) {
    els.hierarchyConferenceSelect.value = requestedKey;
  } else if (entities.some(entity => entity.key === prev)) {
    els.hierarchyConferenceSelect.value = prev;
  }

  if (els.hierarchyConferenceSelect.value){
    renderHierarchyForConference(els.hierarchyConferenceSelect.value, { year: hierarchyYear });
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
  els.hierarchyDetail.textContent = "Select a conference to render the normalized organizations and leadership for that year. Then click an organization, position, or person for details.";
}

function buildHierarchyData(entityKey, opts = {}){
  const targetYear = Number.isFinite(Number(opts.year)) ? Number(opts.year) : null;
  const sourceRows = Array.isArray(opts.rows) ? opts.rows : lastFiltered;
  const { label } = parseEntityKey(entityKey);
  const conferenceLabel = (r) => safe(r.conference_canonical) || safe(r.conference) || "(No conference)";
  const organizationLabel = (r) => safe(r.organization_canonical) || safe(r.organization) || safe(r.institution_name) || "(No organization)";
  const rows = sourceRows.filter(r =>
    conferenceLabel(r) === label &&
    (targetYear === null || Number(r.yearbook_year) === targetYear)
  );
  const posLabel = (r) => safe(r.position) || "(No position)";
  const personKeyForRow = (r, index) => personId(r) || `${safe(r.name)}__${safe(r.position)}__${safe(r.page)}__${index}`;
  const personLabelForRow = (r) => personLabel(r);

  const byOrg = new Map();
  for (const r of rows){
    const org = organizationLabel(r);
    if (!byOrg.has(org)) byOrg.set(org, []);
    byOrg.get(org).push(r);
  }

  const organizations = Array.from(byOrg.entries())
    .map(([name, orgRows]) => ({ name, rows: orgRows }))
    .sort((a, b) => b.rows.length - a.rows.length || a.name.localeCompare(b.name));

  const children = organizations.map(({ name: org, rows: orgRows }) => {
    const byPos = new Map();
    for (const r of orgRows){
      const pos = posLabel(r);
      if (!byPos.has(pos)) byPos.set(pos, []);
      byPos.get(pos).push(r);
    }
    const positions = Array.from(byPos.entries())
      .map(([name, posRows]) => ({ name, rows: posRows }))
      .sort((a, b) => b.rows.length - a.rows.length || a.name.localeCompare(b.name));

    const positionChildren = positions.map(({ name: pos, rows: posRows }) => {
      const byPerson = new Map();
      posRows.forEach((row, index) => {
        const personKey = personKeyForRow(row, index);
        if (!byPerson.has(personKey)) {
          byPerson.set(personKey, {
            name: personLabelForRow(row),
            kind: "person",
            rowCount: 0,
            _row: row,
          });
        }
        byPerson.get(personKey).rowCount += 1;
      });

      const people = Array.from(byPerson.values())
        .sort((a, b) => a.name.localeCompare(b.name));

      return {
        name: pos,
        kind: "position",
        rowCount: posRows.length,
        personCount: people.length,
        children: people,
      };
    });

    return {
      name: org,
      kind: "organization",
      rowCount: orgRows.length,
      personCount: countNamedPeople(orgRows),
      positionCount: positionChildren.length,
      children: positionChildren,
    };
  });

  const sampleRow = rows[0] || null;
  const entityType = sampleRow ? inferConferenceEntityType(sampleRow) : "";
  const family = sampleRow ? (safe(sampleRow.conference_family) || label) : label;

  return {
    name: label || "(No entity)",
    kind: "conference",
    scope: "conference",
    entityType,
    family,
    regions: uniqSorted(rows.map(regionLabelForRow)),
    year: targetYear,
    rowCount: rows.length,
    personCount: countNamedPeople(rows),
    organizationCount: children.length,
    positionCount: children.reduce((sum, org) => sum + org.positionCount, 0),
    childKind: "organization",
    children,
  };
}

function renderHierarchyRow(row, extraFields = [], hint = ""){
  const name = safe(row.name) || "(unknown)";
  const fields = [
    ["Year", safe(row.yearbook_year)],
    ["Page", safe(row.page)],
    ["Position", safe(row.position)],
    ["Organization", safe(row.organization_canonical) || safe(row.organization) || safe(row.institution_name)],
    ["Entity", safe(row.conference_canonical) || safe(row.conference)],
    ["Region", regionLabelForRow(row)],
    ["Location", safe(row.location)],
    ["Institution", safe(row.institution_name)],
    ...extraFields,
  ];

  renderHierarchyDetailCard(name, fields, hint);
}

function renderHierarchyForConference(entityKey, opts = {}){
  const targetYear = Number.isFinite(Number(opts.year)) ? Number(opts.year) : null;
  const { label: entityLabel } = parseEntityKey(entityKey);
  const data = buildHierarchyData(entityKey, { year: targetYear });
  const wrap = els.hierarchyChart;
  wrap.innerHTML = "";
  const childLabel = data.childKind === "conference" ? "Conferences" : "Organizations";
  const childLabelSingular = data.childKind === "conference" ? "Conference" : "Organization";

  if (!data.children.length){
    wrap.innerHTML = `
      <div class="hierEmptyState">
        <div class="hierEmptyState__title">No matching rows in this slice</div>
        <div class="hint">Try a different year, entity filter, or row filter combination.</div>
      </div>
    `;
    renderHierarchyDetailCard(
      entityLabel || "(No entity)",
      [
        ["Scope", data.scope || "entity"],
        ["Family", data.family || "Unknown"],
        ["Type", data.entityType || "Unspecified"],
        ["Year", targetYear !== null ? String(targetYear) : "All loaded years"],
        ["Matching rows", "0"],
      ],
      "No normalized records matched the current filters."
    );
    return;
  }

  const detailLookup = new Map();
  let activeButton = null;

  function registerDetail(id, payload){
    detailLookup.set(id, payload);
  }

  const orgCardsHtml = data.children.map((org, orgIndex) => {
    const orgDetailId = `org-${orgIndex}`;
    registerDetail(orgDetailId, {
      kind: "group",
      title: org.name,
      fields: [
        ["Entity", data.name],
        ["Scope", data.scope],
        ["Family", data.family],
        ["Type", data.entityType || "Unspecified"],
        ["Regions", summarizeValues(data.regions)],
        ["Year", targetYear !== null ? String(targetYear) : "All loaded years"],
        ["Positions", pluralize(org.positionCount, "position")],
        ["People", pluralize(org.personCount, "person")],
        ["Matching rows", org.rowCount.toLocaleString()],
      ],
      hint: `Normalized ${data.childKind} within the selected entity and year.`,
    });

    const positionsHtml = org.children.map((position, positionIndex) => {
      const positionDetailId = `pos-${orgIndex}-${positionIndex}`;
      registerDetail(positionDetailId, {
        kind: "group",
        title: position.name,
        fields: [
          [childLabelSingular, org.name],
          ["Entity", data.name],
          ["Scope", data.scope],
          ["Family", data.family],
          ["Regions", summarizeValues(data.regions)],
          ["Year", targetYear !== null ? String(targetYear) : "All loaded years"],
          ["People", pluralize(position.personCount, "person")],
          ["Matching rows", position.rowCount.toLocaleString()],
        ],
        hint: "Position group inside the selected organization.",
      });

      const peopleHtml = position.children.map((person, personIndex) => {
        const personDetailId = `person-${orgIndex}-${positionIndex}-${personIndex}`;
        registerDetail(personDetailId, {
          kind: "person",
          row: person._row,
          extraFields: person.rowCount > 1 ? [["Matching rows", person.rowCount.toLocaleString()]] : [],
          hint: person.rowCount > 1 ? "This person appears in multiple matching rows for this position in the selected slice." : "",
        });

        return `
          <button type="button" class="hierPersonChip" data-detail-id="${escapeHtml(personDetailId)}">
            ${escapeHtml(person.name)}
          </button>
        `;
      }).join("");

      return `
        <section class="hierPositionCard">
          <button type="button" class="hierPositionHeader" data-detail-id="${escapeHtml(positionDetailId)}">
            <span>${escapeHtml(position.name)}</span>
            <span class="hierMeta">${escapeHtml(`${pluralize(position.personCount, "person")} • ${position.rowCount.toLocaleString()} rows`)}</span>
          </button>
          <div class="hierPeopleRow">
            ${peopleHtml || '<span class="hint">No named people</span>'}
          </div>
        </section>
      `;
    }).join("");

    return `
      <section class="hierOrgCard">
        <button type="button" class="hierOrgHeader" data-detail-id="${escapeHtml(orgDetailId)}">
          <span>${escapeHtml(org.name)}</span>
          <span class="hierMeta">${escapeHtml(`${pluralize(org.personCount, "person")} • ${pluralize(org.positionCount, "position")}`)}</span>
        </button>
        <div class="hierPositionList">
          ${positionsHtml}
        </div>
      </section>
    `;
  }).join("");

  wrap.innerHTML = `
    <div class="hierSummaryBar">
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Entity</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.name)}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Scope</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.scope || "entity")}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Type</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.entityType || "Unspecified")}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Family</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.family || "Unknown")}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Regions</span>
        <span class="hierSummaryStat__value">${escapeHtml(summarizeValues(data.regions, 4) || "Unknown")}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Year</span>
        <span class="hierSummaryStat__value">${escapeHtml(targetYear !== null ? String(targetYear) : "All loaded years")}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">${escapeHtml(childLabel)}</span>
        <span class="hierSummaryStat__value">${data.organizationCount.toLocaleString()}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">People</span>
        <span class="hierSummaryStat__value">${data.personCount.toLocaleString()}</span>
      </div>
    </div>
    <div class="hierOrgList">${orgCardsHtml}</div>
  `;

  function setActiveButton(button){
    if (activeButton) activeButton.classList.remove("is-active");
    activeButton = button || null;
    if (activeButton) activeButton.classList.add("is-active");
  }

  wrap.querySelectorAll("[data-detail-id]").forEach(button => {
    button.addEventListener("click", () => {
      const detailId = button.getAttribute("data-detail-id");
      const detail = detailLookup.get(detailId);
      if (!detail) return;
      hierarchyPinned = detail;
      setActiveButton(button);
      if (detail.kind === "person" && detail.row){
        renderHierarchyRow(detail.row, detail.extraFields || [], detail.hint || "");
        return;
      }
      renderHierarchyDetailCard(detail.title, detail.fields || [], detail.hint || "");
    });
  });

  hierarchyPinned = null;
  renderHierarchyDetailCard(
    entityLabel || "(No entity)",
    [
      ["Scope", data.scope || "entity"],
      ["Type", data.entityType || "Unspecified"],
      ["Family", data.family || "Unknown"],
      ["Regions", summarizeValues(data.regions)],
      ["Year", targetYear !== null ? String(targetYear) : "All loaded years"],
      [childLabel, pluralize(data.organizationCount, data.childKind)],
      ["Positions", pluralize(data.positionCount, "position")],
      ["People", pluralize(data.personCount, "person")],
      ["Matching rows", data.rowCount.toLocaleString()],
    ],
    "Use the organization and position sections on the left to inspect the leadership structure for this entity and year."
  );
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
  out.region = normalizeSpacing(out.region);
  out.region_normalized = normalizeRegionName(out.region);

  const conferenceNorm = normalizeConferenceEntity(out.conference);
  out.conference = normalizeSpacing(out.conference);
  out.conference_family = conferenceNorm.family;
  out.conference_type = conferenceNorm.type;
  out.conference_canonical = conferenceNorm.canonical;

  const organizationSource = safe(out.organization) || safe(out.institution_name);
  const organizationNorm = normalizeOrganizationEntity(organizationSource);
  out.organization = normalizeSpacing(out.organization);
  out.institution_name = normalizeSpacing(out.institution_name);
  out.organization_family = organizationNorm.family;
  out.organization_type = organizationNorm.type;
  out.organization_canonical = organizationNorm.canonical;
  return out;
}

async function reloadAllEnabled(){
  allRows = [];
  enabledYears = new Set(manifest.datasets.map(d => d.year));
  const enabled = manifest.datasets;
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
    openHierarchy({ year: hierarchyYear });
    if (keep) els.hierarchyConferenceSelect.value = keep;
    if (els.hierarchyConferenceSelect.value) renderHierarchyForConference(els.hierarchyConferenceSelect.value, { year: hierarchyYear });
  }
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
    groupOrder: (a, b) => (a.sortOrder ?? 999) - (b.sortOrder ?? 999),
  };

  timeline = new vis.Timeline(container, items, groups, options);

  timeline.on("select", (props) => {
    const id = props.items && props.items[0];
    if (!id) return renderTimelineDetail(null);
    const it = items.get(id);
    renderTimelineDetail(it || null);
  });

  timeline.on("click", (props) => {
    if (!props.item) return;
    const it = items.get(props.item);
    if (!it) return;
    const clickedDate = props.time instanceof Date ? props.time : new Date(props.time);
    const clickedYear = Number.isFinite(clickedDate.getFullYear()) ? clickedDate.getFullYear() : it._startYear;
    const targetYear = Math.max(it._startYear, Math.min(it._endYear, clickedYear));
    renderTimelineDetail(it, targetYear);
    openHierarchy({ entityKey: it._entityKey, year: targetYear });
  });
}

async function main(){
  els.searchInput = $("searchInput");
  els.regionSelect = $("regionSelect");
  els.confSelect = $("confSelect");
  els.positionSelect = $("positionSelect");
  els.genderSelect = $("genderSelect");
  els.orgSelect = $("orgSelect");
  els.entityTypeSelect = $("entityTypeSelect");
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
  els.hierarchyChart = $("hierarchyChart");
  els.hierarchyDetail = $("hierarchyDetail");

  els.statLoaded = $("statLoaded");
  els.statShown = $("statShown");
  els.statYears = $("statYears");
  els.timeline = $("timeline");

  await loadNormalizationConfig();
  initTimeline();

  manifest = await loadManifest();

  enabledYears = new Set(manifest.datasets.map(d => d.year));
  await reloadAllEnabled();

  els.applyBtn.addEventListener("click", applyFilters);
  els.resetBtn.addEventListener("click", resetFilters);

  els.hierarchyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    openHierarchy({ year: null });
  });

  els.hierarchyBackdrop.addEventListener("click", closeHierarchy);
  els.hierarchyCloseBtn.addEventListener("click", closeHierarchy);
  els.hierarchyConferenceSelect.addEventListener("change", () => {
    const entityKey = safe(els.hierarchyConferenceSelect.value);
    if (!entityKey) return clearHierarchy();
    renderHierarchyForConference(entityKey, { year: hierarchyYear });
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

  [els.regionSelect, els.confSelect, els.positionSelect, els.genderSelect, els.orgSelect, els.entityTypeSelect]
    .filter(Boolean)
    .forEach(select => select.addEventListener("change", applyFilters));

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

  const width = Math.max(Math.floor(opts.width || el.clientWidth || 0), 320);
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

function renderTopPeopleByYear(stats){
  const host = $("topPeopleByYear");
  if (!host) return;
  host.innerHTML = "";

  if (!stats.length){
    host.innerHTML = `<div class="summaryEmptyState">No yearly values are available for this filtered slice.</div>`;
    return;
  }

  for (const s of stats){
    const leaders = Array.from(s.people.values())
      .sort((a, b) => (b.roleCount - a.roleCount) || a.label.localeCompare(b.label))
      .slice(0, 10);

    const article = document.createElement("article");
    article.className = "yearLeaderCard";

    const items = leaders.length
      ? leaders.map((person, idx) => `
          <li class="yearLeaderItem">
            <span class="yearLeaderRank">${idx + 1}</span>
            <span class="yearLeaderName">${escapeHtml(person.label)}</span>
            <span class="yearLeaderCount">${person.roleCount.toLocaleString()} role${person.roleCount === 1 ? "" : "s"}</span>
          </li>
        `).join("")
      : `<li class="yearLeaderItem yearLeaderItem--empty">No named individuals in this filtered slice.</li>`;

    article.innerHTML = `
      <div class="yearLeaderCard__header">
        <div class="yearLeaderCard__year">${s.year}</div>
        <div class="yearLeaderCard__meta">${s.namedIndividuals.toLocaleString()} named individual${s.namedIndividuals === 1 ? "" : "s"}</div>
      </div>
      <ol class="yearLeaderList">${items}</ol>
    `;
    host.appendChild(article);
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
  const summaryChartHeight = 380;
  const summaryChartBoxHeight = 460;
  const summaryChartWidths = ["womenPctChart", "gt5RolesChart"]
    .map(id => $(id)?.parentElement?.clientWidth || 0)
    .filter(width => width > 0);
  const summaryChartWidth = Math.max(summaryChartWidths.length ? Math.min(...summaryChartWidths) : 320, 320);
  for (const id of ["womenPctChart", "gt5RolesChart"]){
    const chartEl = $(id);
    if (!chartEl) continue;
    chartEl.style.width = `${summaryChartWidth}px`;
    chartEl.style.maxWidth = "100%";
    chartEl.style.height = `${summaryChartBoxHeight}px`;
    chartEl.style.marginLeft = "auto";
    chartEl.style.marginRight = "auto";
  }

  setMetricPills("womenPct", stats, "womenPct");
  setMetricPills("gt5Pct", stats, "gt5Pct");

  renderPercentChart("womenPctChart", stats, "womenPct", {
    label: "Percentage of named individuals identified as women over time",
    height: summaryChartHeight,
    width: summaryChartWidth,
    title: d => `${d.year}: ${formatPercent(d.womenPct)} (${d.women} women of ${d.namedIndividuals} named individuals)`,
    footnote: "Percentage = unique named individuals identified as women divided by all unique named individuals in each year. One person with many roles is still counted once per year.",
  });

  renderPercentChart("gt5RolesChart", stats, "gt5Pct", {
    label: "Percentage of named individuals with more than five roles over time",
    height: summaryChartHeight,
    width: summaryChartWidth,
    alt: true,
    title: d => `${d.year}: ${formatPercent(d.gt5Pct)} (${d.gt5} people with >5 roles of ${d.namedIndividuals} named individuals)`,
    footnote: "For each year, this uses the filtered slice and counts whether each unique named individual has more than five matching role rows that year.",
  });

  renderPerYearTable(stats);
  renderTopPeopleByYear(stats);
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
