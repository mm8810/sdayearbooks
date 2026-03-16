let manifest = null;
let allRows = [];      // raw rows across all enabled datasets
let enabledYears = new Set();
let timeline = null;
let items = null;
let groupsData = null;
let lastFiltered = [];
let lastConferenceView = { conferenceCount: 0, segmentCount: 0, minYear: null, maxYear: null };
let selectedConferenceItem = null;
let selectedConferenceContext = null;
let selectedOrganizationContext = null;
let datasetStatus = new Map();
let datasetLoadIssues = [];

let hierarchyPinned = null;

const els = {};
function $(id){ return document.getElementById(id); }

function safe(v){
  if (v === null || v === undefined) return "";
  return String(v).trim();
}

function escapeHtml(str){
  return String(str)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function yearToDate(y){
  return new Date(Number(y), 0, 1);
}

function yearToEndDate(y){
  return new Date(Number(y) + 1, 0, 1);
}

function personDisplayName(row){
  const direct = safe(row.name);
  if (direct) return direct;
  const combo = [safe(row.prefix), safe(row.last_name), safe(row.suffix)].filter(Boolean).join(" ").trim();
  return combo || "(unknown)";
}

function organizationLabel(row){
  return safe(row.organization) || safe(row.institution_name) || "(No organization)";
}

function uniqSorted(values){
  const s = new Set(values.filter(v => safe(v)));
  return Array.from(s).sort((a,b)=> a.localeCompare(b));
}

function uniqueFiniteYears(rows){
  return Array.from(
    new Set(
      rows
        .map(r => Number(r.yearbook_year))
        .filter(Number.isFinite)
    )
  ).sort((a,b)=>a-b);
}

function formatYearRange(startYear, endYear){
  if (!Number.isFinite(startYear) && !Number.isFinite(endYear)) return "–";
  if (!Number.isFinite(endYear) || startYear === endYear) return String(startYear);
  return `${startYear}–${endYear}`;
}

function summarizeRowCollection(rows){
  const years = uniqueFiniteYears(rows);
  const regions = uniqSorted(rows.map(r => safe(r.region)));
  const organizations = uniqSorted(rows.map(organizationLabel));
  const positions = uniqSorted(rows.map(r => safe(r.position)));
  const uniquePeople = new Set(rows.map(personId));

  return {
    rows,
    years,
    startYear: years.length ? years[0] : null,
    endYear: years.length ? years[years.length - 1] : null,
    entryCount: rows.length,
    uniquePeopleCount: uniquePeople.size,
    organizationsCount: organizations.length,
    positionsCount: positions.length,
    regions,
    regionCount: regions.length
  };
}

function buildConferenceSegments(years){
  const sorted = Array.from(new Set(years.filter(Number.isFinite))).sort((a,b)=>a-b);
  if (!sorted.length) return [];

  const segments = [];
  let start = sorted[0];
  let prev = sorted[0];
  let currentYears = [sorted[0]];

  for (let i = 1; i < sorted.length; i += 1){
    const year = sorted[i];
    if (year === prev + 1){
      prev = year;
      currentYears.push(year);
      continue;
    }

    segments.push({ startYear: start, endYear: prev, years: currentYears.slice() });
    start = year;
    prev = year;
    currentYears = [year];
  }

  segments.push({ startYear: start, endYear: prev, years: currentYears.slice() });
  return segments;
}

function buildConferenceGroupLabel(summary){
  const yearsLabel = formatYearRange(summary.startYear, summary.endYear);
  const metaBits = [yearsLabel];
  if (summary.organizationsCount) metaBits.push(`${summary.organizationsCount} orgs`);
  if (summary.uniquePeopleCount) metaBits.push(`${summary.uniquePeopleCount} people`);

  return `
    <div class="timelineGroup">
      <div class="timelineGroup__title">${escapeHtml(summary.conference)}</div>
      <div class="timelineGroup__meta">${escapeHtml(metaBits.join(" • "))}</div>
    </div>
  `;
}

function buildConferenceSegmentContent(segmentSummary){
  const yearsLabel = formatYearRange(segmentSummary.startYear, segmentSummary.endYear);
  const badge = segmentSummary.state === "return" ? "Returns" : "Starts";
  return `
    <div class="confSegment">
      <span class="confSegment__range">${escapeHtml(yearsLabel)}</span>
      <span class="confSegment__badge">${escapeHtml(badge)}</span>
    </div>
  `;
}

function buildConferenceTooltip(summary, segmentSummary){
  const lines = [
    summary.conference,
    `Span: ${formatYearRange(segmentSummary.startYear, segmentSummary.endYear)}`,
    `${segmentSummary.organizationsCount} organizations • ${segmentSummary.uniquePeopleCount} people`,
    `${segmentSummary.positionsCount} positions • ${segmentSummary.entryCount} rows`,
  ];

  if (segmentSummary.regions.length){
    lines.push(`Regions: ${segmentSummary.regions.join(", ")}`);
  }

  lines.push("Click to drill into organizations and individuals.");
  return lines.join("\n");
}

function buildConferenceTimeline(rows){
  const byConference = new Map();

  for (const row of rows){
    const conference = safe(row.conference) || "(No conference)";
    if (!byConference.has(conference)) byConference.set(conference, []);
    byConference.get(conference).push(row);
  }

  const summaries = Array.from(byConference.entries())
    .map(([conference, conferenceRows]) => {
      const base = summarizeRowCollection(conferenceRows);
      const segments = buildConferenceSegments(base.years).map((segment, segmentIndex) => {
        const segmentRows = conferenceRows.filter(row => {
          const year = Number(row.yearbook_year);
          return Number.isFinite(year) && year >= segment.startYear && year <= segment.endYear;
        });
        return {
          ...summarizeRowCollection(segmentRows),
          ...segment,
          rows: segmentRows,
          state: segmentIndex === 0 ? "start" : "return",
          segmentIndex,
        };
      });

      return {
        conference,
        ...base,
        rows: conferenceRows,
        segments,
      };
    })
    .sort((a,b) => {
      const ay = Number.isFinite(a.startYear) ? a.startYear : 9999;
      const by = Number.isFinite(b.startYear) ? b.startYear : 9999;
      return ay - by || a.conference.localeCompare(b.conference);
    });

  const groups = summaries.map(summary => ({
    id: summary.conference,
    content: buildConferenceGroupLabel(summary),
    title: `${summary.conference}\n${formatYearRange(summary.startYear, summary.endYear)}\n${summary.organizationsCount} organizations • ${summary.uniquePeopleCount} people`
  }));

  const timelineItems = [];
  for (const summary of summaries){
    for (const segment of summary.segments){
      timelineItems.push({
        id: `conference:${summary.conference}:${segment.startYear}-${segment.endYear}:${segment.segmentIndex}`,
        content: buildConferenceSegmentContent(segment),
        start: yearToDate(segment.startYear),
        end: yearToEndDate(segment.endYear),
        type: "range",
        group: summary.conference,
        title: buildConferenceTooltip(summary, segment),
        className: `conference-segment conference-segment--${segment.state} ${segment.startYear === segment.endYear ? "conference-segment--single" : "conference-segment--multi"}`,
        _conferenceSummary: summary,
        _segmentSummary: segment,
      });
    }
  }

  const allYears = uniqueFiniteYears(rows);
  return {
    groups,
    items: timelineItems,
    conferenceCount: summaries.length,
    segmentCount: timelineItems.length,
    minYear: allYears.length ? allYears[0] : null,
    maxYear: allYears.length ? allYears[allYears.length - 1] : null,
  };
}

function setStats(view = lastConferenceView){
  const loaded = allRows.length;
  const shownConferences = view?.conferenceCount || 0;
  const shownSegments = view?.segmentCount || 0;
  const minYear = view?.minYear;
  const maxYear = view?.maxYear;
  const fallbackYears = uniqueFiniteYears(allRows);
  const yearLabel = Number.isFinite(minYear) && Number.isFinite(maxYear)
    ? formatYearRange(minYear, maxYear)
    : fallbackYears.length
      ? formatYearRange(fallbackYears[0], fallbackYears[fallbackYears.length - 1])
      : "–";

  const loadSuffix = datasetLoadIssues.length ? ` • skipped ${datasetLoadIssues.length}` : "";
  els.statLoaded.textContent = `Loaded: ${loaded.toLocaleString()} rows${loadSuffix}`;
  els.statShown.textContent = `Shown: ${shownConferences.toLocaleString()} conferences • ${shownSegments.toLocaleString()} spans`;
  els.statYears.textContent = `Years: ${yearLabel}`;
}

function hydrateFilterOptions(){
  const regions = uniqSorted(allRows.map(r => r.region));
  const confs = uniqSorted(allRows.map(r => r.conference));
  const positions = uniqSorted(allRows.map(r => r.position));
  const genders = uniqSorted(allRows.map(r => r.gender));
  const orgs = uniqSorted(allRows.map(organizationLabel));

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

function applyFilters(){
  closeOrganizationModal();
  closeConferenceModal();

  const q = safe(els.searchInput.value).toLowerCase();
  const region = safe(els.regionSelect.value);
  const conf = safe(els.confSelect.value);
  const pos = safe(els.positionSelect?.value);
  const gender = safe(els.genderSelect?.value);
  const orgSel = safe(els.orgSelect?.value);
  const yMin = Number(els.yearMin.value || 0);
  const yMax = Number(els.yearMax.value || 9999);

  const filtered = allRows.filter(row => {
    const year = Number(row.yearbook_year);
    if (Number.isFinite(yMin) && year < yMin) return false;
    if (Number.isFinite(yMax) && year > yMax) return false;

    if (q){
      const haystack = [
        personDisplayName(row),
        safe(row.last_name),
        safe(row.conference),
        organizationLabel(row),
        safe(row.position),
        safe(row.region),
        safe(row.location),
        safe(row.institution_name),
      ].join(" ").toLowerCase();
      if (!haystack.includes(q)) return false;
    }

    if (region && safe(row.region) !== region) return false;
    if (conf && safe(row.conference) !== conf) return false;
    if (pos && safe(row.position) !== pos) return false;
    if (gender && safe(row.gender) !== gender) return false;
    if (orgSel && organizationLabel(row) !== orgSel) return false;

    return true;
  });

  lastFiltered = filtered;
  selectedConferenceItem = null;

  const conferenceView = buildConferenceTimeline(filtered);
  lastConferenceView = conferenceView;

  items.clear();
  items.add(conferenceView.items);

  groupsData = new vis.DataSet(conferenceView.groups);
  timeline.setGroups(groupsData);

  setStats(conferenceView);
  renderDetail(null);

  if (conferenceView.minYear !== null && conferenceView.maxYear !== null){
    const start = new Date(conferenceView.minYear - 1, 0, 1);
    const end = new Date(conferenceView.maxYear + 1, 0, 1);
    timeline.setWindow(start, end, { animation: false });
  }
}

function resetFilters(){
  const manifestYears = Array.isArray(manifest?.datasets)
    ? manifest.datasets.map(d => Number(d.year)).filter(Number.isFinite).sort((a,b)=>a-b)
    : [];

  els.searchInput.value = "";
  els.regionSelect.value = "";
  els.confSelect.value = "";
  if (els.positionSelect) els.positionSelect.value = "";
  if (els.genderSelect) els.genderSelect.value = "";
  if (els.orgSelect) els.orgSelect.value = "";
  els.yearMin.value = manifestYears.length ? String(manifestYears[0]) : "1883";
  els.yearMax.value = manifestYears.length ? String(manifestYears[manifestYears.length - 1]) : "1921";
  applyFilters();
}

function renderDetail(item){
  if (!item){
    els.detailCard.classList.add("muted");
    els.detailCard.textContent = "Click a conference span on the timeline to open organizations and individuals.";
    return;
  }

  const summary = item._conferenceSummary;
  const segment = item._segmentSummary;
  if (!summary || !segment){
    els.detailCard.classList.add("muted");
    els.detailCard.textContent = "Click a conference span on the timeline to open organizations and individuals.";
    return;
  }

  els.detailCard.classList.remove("muted");

  const fields = [
    ["Conference", summary.conference],
    ["Span", formatYearRange(segment.startYear, segment.endYear)],
    ["Active years", `${segment.years.length}`],
    ["Organizations", `${segment.organizationsCount}`],
    ["Unique people", `${segment.uniquePeopleCount}`],
    ["Positions", `${segment.positionsCount}`],
    ["Regions", segment.regions.join(", ")],
    ["Rows", `${segment.entryCount}`],
  ].filter(([,value]) => safe(value));

  els.detailCard.innerHTML = `
    <div style="font-weight:750; font-size:14px;">${escapeHtml(summary.conference)}</div>
    <div class="kv">
      ${fields.map(([k,v]) => `<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div>`).join("")}
    </div>
    <div style="margin-top:12px; display:flex; justify-content:flex-end;">
      <button class="btn btn--primary" type="button" data-open-conference>Open drill-down</button>
    </div>
  `;

  const btn = els.detailCard.querySelector('[data-open-conference]');
  if (btn){
    btn.addEventListener('click', () => openConferenceModal(item));
  }
}

function buildStatCard(label, value, hint = ""){
  return `
    <div class="miniStat">
      <div class="miniStat__label">${escapeHtml(label)}</div>
      <div class="miniStat__value">${escapeHtml(String(value))}</div>
      ${hint ? `<div class="miniStat__hint">${escapeHtml(hint)}</div>` : ""}
    </div>
  `;
}

function summarizeOrganizationRows(rows, conferenceName, orgName){
  const years = uniqueFiniteYears(rows);
  const positions = uniqSorted(rows.map(r => safe(r.position)));
  const locations = uniqSorted(rows.map(r => safe(r.location)));
  const peopleSeen = new Set();
  const byYear = new Map();

  for (const row of rows){
    const year = Number(row.yearbook_year);
    const yearKey = Number.isFinite(year) ? year : "Unknown";
    if (!byYear.has(yearKey)) byYear.set(yearKey, new Map());

    const personKey = personId(row);
    peopleSeen.add(personKey);

    const yearMap = byYear.get(yearKey);
    if (!yearMap.has(personKey)){
      yearMap.set(personKey, {
        id: personKey,
        name: personDisplayName(row),
        positions: new Set(),
        locations: new Set(),
        pages: new Set(),
        genders: new Set(),
        notes: new Set(),
      });
    }

    const person = yearMap.get(personKey);
    if (safe(row.position)) person.positions.add(safe(row.position));
    if (safe(row.location)) person.locations.add(safe(row.location));
    if (safe(row.page)) person.pages.add(String(row.page));
    if (safe(row.gender)) person.genders.add(safe(row.gender));
    if (safe(row.position_information)) person.notes.add(safe(row.position_information));
  }

  const peopleByYear = Array.from(byYear.entries())
    .sort((a,b) => {
      const ay = typeof a[0] === 'number' ? a[0] : 9999;
      const by = typeof b[0] === 'number' ? b[0] : 9999;
      return ay - by || String(a[0]).localeCompare(String(b[0]));
    })
    .map(([year, peopleMap]) => ({
      year,
      people: Array.from(peopleMap.values())
        .map(person => ({
          ...person,
          positions: Array.from(person.positions).sort((a,b)=>a.localeCompare(b)),
          locations: Array.from(person.locations).sort((a,b)=>a.localeCompare(b)),
          pages: Array.from(person.pages).sort((a,b)=>Number(a) - Number(b)),
          genders: Array.from(person.genders).sort((a,b)=>a.localeCompare(b)),
          notes: Array.from(person.notes).sort((a,b)=>a.localeCompare(b)),
        }))
        .sort((a,b)=>a.name.localeCompare(b.name))
    }));

  return {
    conference: conferenceName,
    organization: orgName,
    rows,
    years,
    startYear: years.length ? years[0] : null,
    endYear: years.length ? years[years.length - 1] : null,
    entryCount: rows.length,
    uniquePeopleCount: peopleSeen.size,
    positionsCount: positions.length,
    locationCount: locations.length,
    positions,
    peopleByYear,
  };
}

function buildOrganizationSummaries(rows, conferenceName){
  const byOrg = new Map();
  for (const row of rows){
    const org = organizationLabel(row);
    if (!byOrg.has(org)) byOrg.set(org, []);
    byOrg.get(org).push(row);
  }

  return Array.from(byOrg.entries())
    .map(([orgName, orgRows]) => summarizeOrganizationRows(orgRows, conferenceName, orgName))
    .sort((a,b) => a.organization.localeCompare(b.organization));
}

function buildConferenceSelectionContext(item){
  const conferenceSummary = item?._conferenceSummary;
  const segmentSummary = item?._segmentSummary;
  if (!conferenceSummary || !segmentSummary) return null;

  const organizations = buildOrganizationSummaries(segmentSummary.rows, conferenceSummary.conference);
  return {
    item,
    conference: conferenceSummary.conference,
    conferenceSummary,
    segmentSummary,
    rows: segmentSummary.rows,
    years: segmentSummary.years,
    organizations,
  };
}

function renderConferenceModal(context){
  if (!context || !els.conferenceModal) return;

  const { conference, segmentSummary, years, organizations } = context;
  els.conferenceModalTitle.textContent = conference;
  els.conferenceModalSubtitle.textContent = `${formatYearRange(segmentSummary.startYear, segmentSummary.endYear)} • ${organizations.length} organizations • ${segmentSummary.uniquePeopleCount} people`;

  els.conferenceStatGrid.innerHTML = [
    buildStatCard('Active span', formatYearRange(segmentSummary.startYear, segmentSummary.endYear), segmentSummary.state === 'return' ? 'Conference returns after a gap' : 'Conference first span in current view'),
    buildStatCard('Active years', segmentSummary.years.length),
    buildStatCard('Organizations', segmentSummary.organizationsCount),
    buildStatCard('Unique people', segmentSummary.uniquePeopleCount),
    buildStatCard('Positions', segmentSummary.positionsCount),
    buildStatCard('Regions', segmentSummary.regionCount || '—', segmentSummary.regions.join(', ') || 'No region labels')
  ].join('');

  els.conferenceYears.innerHTML = years.length
    ? years.map(year => `<span class="chip">${escapeHtml(String(year))}</span>`).join('')
    : '<div class="emptyState">No active years in this conference span.</div>';

  els.conferenceOrgList.innerHTML = organizations.length
    ? organizations.map((org, index) => `
        <button class="orgCard" type="button" data-org-index="${index}">
          <div class="orgCard__name">${escapeHtml(org.organization)}</div>
          <div class="orgCard__meta">
            <span>${escapeHtml(formatYearRange(org.startYear, org.endYear))}</span>
            <span>${org.uniquePeopleCount.toLocaleString()} people</span>
            <span>${org.positionsCount.toLocaleString()} positions</span>
          </div>
        </button>
      `).join('')
    : '<div class="emptyState">No organizations matched the current filters for this conference span.</div>';
}

function openConferenceModal(item){
  if (!els.conferenceModal || !item) return;
  selectedConferenceItem = item;
  selectedConferenceContext = buildConferenceSelectionContext(item);
  selectedOrganizationContext = null;
  renderConferenceModal(selectedConferenceContext);
  els.conferenceModal.setAttribute('aria-hidden', 'false');
}

function closeConferenceModal(){
  if (!els.conferenceModal) return;
  closeOrganizationModal();
  els.conferenceModal.setAttribute('aria-hidden', 'true');
  selectedConferenceContext = null;
  selectedOrganizationContext = null;
}

function renderOrganizationModal(orgSummary){
  if (!orgSummary || !els.organizationModal) return;

  els.organizationModalTitle.textContent = orgSummary.organization;
  els.organizationModalSubtitle.textContent = `${orgSummary.conference} • ${formatYearRange(orgSummary.startYear, orgSummary.endYear)} • ${orgSummary.uniquePeopleCount} people`;

  els.organizationStatGrid.innerHTML = [
    buildStatCard('Active span', formatYearRange(orgSummary.startYear, orgSummary.endYear)),
    buildStatCard('Unique people', orgSummary.uniquePeopleCount),
    buildStatCard('Positions', orgSummary.positionsCount),
    buildStatCard('Locations', orgSummary.locationCount || '—'),
    buildStatCard('Rows', orgSummary.entryCount),
    buildStatCard('Years', orgSummary.years.length || '—')
  ].join('');

  els.organizationPeopleList.innerHTML = orgSummary.peopleByYear.length
    ? orgSummary.peopleByYear.map(yearBlock => `
        <section class="yearBlock">
          <div class="yearBlock__header">
            <span>${escapeHtml(String(yearBlock.year))}</span>
            <span>${yearBlock.people.length.toLocaleString()} people</span>
          </div>
          <div class="personGrid">
            ${yearBlock.people.map(person => {
              const note = person.notes.find(Boolean) || '';
              const trimmedNote = note.length > 140 ? `${note.slice(0, 137)}…` : note;
              return `
                <article class="personCard">
                  <div class="personCard__name">${escapeHtml(person.name)}</div>
                  <div class="personCard__meta">${escapeHtml(person.positions.join(' • ') || 'No position listed')}</div>
                  ${person.locations.length ? `<div class="personCard__sub">${escapeHtml(person.locations.join(' • '))}</div>` : ''}
                  ${person.pages.length ? `<div class="personCard__sub">Page ${escapeHtml(person.pages.join(', '))}</div>` : ''}
                  ${trimmedNote ? `<div class="personCard__note">${escapeHtml(trimmedNote)}</div>` : ''}
                </article>
              `;
            }).join('')}
          </div>
        </section>
      `).join('')
    : '<div class="emptyState">No individuals matched the current filters for this organization.</div>';
}

function openOrganizationModalByIndex(index){
  if (!selectedConferenceContext || !els.organizationModal) return;
  const orgSummary = selectedConferenceContext.organizations[index];
  if (!orgSummary) return;
  selectedOrganizationContext = orgSummary;
  renderOrganizationModal(orgSummary);
  els.organizationModal.setAttribute('aria-hidden', 'false');
}

function closeOrganizationModal(){
  if (!els.organizationModal) return;
  els.organizationModal.setAttribute('aria-hidden', 'true');
  selectedOrganizationContext = null;
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

async function loadCsv(file){
  const res = await fetch(file);
  if (!res.ok) throw new Error(`Could not load ${file}`);
  const csvText = await res.text();
  const parsed = Papa.parse(csvText, {
    header: true,
    skipEmptyLines: true,
  });
  return parsed.data || [];
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
  datasetLoadIssues = [];
  datasetStatus = new Map();

  const enabled = manifest.datasets.filter(d => enabledYears.has(d.year));
  const results = await Promise.all(enabled.map(async dataset => {
    try {
      const rows = await loadCsv(dataset.file);
      const normalized = rows.map(row => normalizeRow(row, dataset.year));
      datasetStatus.set(dataset.year, { status: 'loaded', rows: normalized.length });
      return normalized;
    } catch (err) {
      console.warn(`Skipping dataset ${dataset.year}:`, err);
      datasetLoadIssues.push({ year: dataset.year, file: dataset.file, message: err?.message || String(err) });
      datasetStatus.set(dataset.year, { status: 'error', message: err?.message || String(err) });
      return [];
    }
  }));

  allRows = results.flat();

  renderDatasetList();
  hydrateFilterOptions();
  applyFilters();

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
    const status = datasetStatus.get(ds.year);
    const statusLabel = !on
      ? 'disabled'
      : status?.status === 'loaded'
        ? `${status.rows.toLocaleString()} rows`
        : status?.status === 'error'
          ? 'unavailable'
          : 'pending';
    const extraClass = status?.status === 'error' ? ' datasetPill--error' : '';

    return `
      <div class="datasetPill${extraClass}" data-year="${ds.year}">
        <div class="datasetPill__left">
          <div class="datasetPill__title">${escapeHtml(ds.label || String(ds.year))}</div>
          <div class="datasetPill__meta">${escapeHtml(ds.file)} • ${escapeHtml(statusLabel)}</div>
        </div>
        <div class="datasetPill__right">
          <span class="toggle" aria-label="toggle">${on ? "✓" : ""}</span>
        </div>
      </div>
    `;
  }).join("");

  els.datasetList.innerHTML = html;

  els.datasetList.querySelectorAll('.datasetPill').forEach(el => {
    el.addEventListener('click', async () => {
      const year = Number(el.getAttribute('data-year'));
      if (enabledYears.has(year)) enabledYears.delete(year);
      else enabledYears.add(year);

      renderDatasetList();
      await reloadAllEnabled();
    });
  });
}

function initTimeline(){
  const container = els.timeline;

  items = new vis.DataSet([]);
  groupsData = new vis.DataSet([]);

  const options = {
    stack: false,
    maxHeight: '640px',
    zoomMin: 1000 * 60 * 60 * 24 * 365 * 0.8,
    zoomMax: 1000 * 60 * 60 * 24 * 365 * 200,
    horizontalScroll: true,
    verticalScroll: true,
    zoomKey: 'ctrlKey',
    multiselect: false,
    showCurrentTime: false,
    margin: { item: { horizontal: 10, vertical: 8 }, axis: 10 },
    tooltip: { followMouse: true },
  };

  timeline = new vis.Timeline(container, items, groupsData, options);

  timeline.on('select', props => {
    const id = props.items && props.items[0];
    if (!id){
      selectedConferenceItem = null;
      return renderDetail(null);
    }

    const item = items.get(id);
    selectedConferenceItem = item || null;
    renderDetail(item || null);
    if (item) openConferenceModal(item);
  });

  const start = new Date(1880, 0, 1);
  const end = new Date(1890, 0, 1);
  timeline.setWindow(start, end, { animation: false });
}

async function main(){
  els.datasetList = $('datasetList');
  els.searchInput = $('searchInput');
  els.regionSelect = $('regionSelect');
  els.confSelect = $('confSelect');
  els.positionSelect = $('positionSelect');
  els.genderSelect = $('genderSelect');
  els.orgSelect = $('orgSelect');
  els.yearMin = $('yearMin');
  els.yearMax = $('yearMax');
  els.applyBtn = $('applyBtn');
  els.resetBtn = $('resetBtn');
  els.detailCard = $('detailCard');
  els.hierarchyBtn = $('hierarchyBtn');
  els.exportBtn = $('exportBtn');
  els.hierarchyModal = $('hierarchyModal');
  els.hierarchyBackdrop = $('hierarchyBackdrop');
  els.hierarchyCloseBtn = $('hierarchyCloseBtn');
  els.hierarchyConferenceSelect = $('hierarchyConferenceSelect');
  els.hierarchyGroupingSelect = $('hierarchyGroupingSelect');
  els.hierarchyChart = $('hierarchyChart');
  els.hierarchyDetail = $('hierarchyDetail');
  els.conferenceModal = $('conferenceModal');
  els.conferenceBackdrop = $('conferenceBackdrop');
  els.conferenceCloseBtn = $('conferenceCloseBtn');
  els.conferenceModalTitle = $('conferenceModalTitle');
  els.conferenceModalSubtitle = $('conferenceModalSubtitle');
  els.conferenceStatGrid = $('conferenceStatGrid');
  els.conferenceYears = $('conferenceYears');
  els.conferenceOrgList = $('conferenceOrgList');
  els.organizationModal = $('organizationModal');
  els.organizationBackdrop = $('organizationBackdrop');
  els.organizationBackBtn = $('organizationBackBtn');
  els.organizationCloseBtn = $('organizationCloseBtn');
  els.organizationModalTitle = $('organizationModalTitle');
  els.organizationModalSubtitle = $('organizationModalSubtitle');
  els.organizationStatGrid = $('organizationStatGrid');
  els.organizationPeopleList = $('organizationPeopleList');

  els.statLoaded = $('statLoaded');
  els.statShown = $('statShown');
  els.statYears = $('statYears');
  els.timeline = $('timeline');

  initTimeline();

  manifest = await loadManifest();

  const manifestYears = Array.isArray(manifest?.datasets)
    ? manifest.datasets.map(d => Number(d.year)).filter(Number.isFinite).sort((a,b)=>a-b)
    : [];
  if (manifestYears.length){
    els.yearMin.value = String(manifestYears[0]);
    els.yearMax.value = String(manifestYears[manifestYears.length - 1]);
    els.yearMin.min = String(manifestYears[0]);
    els.yearMin.max = String(manifestYears[manifestYears.length - 1]);
    els.yearMax.min = String(manifestYears[0]);
    els.yearMax.max = String(manifestYears[manifestYears.length - 1]);
  }

  enabledYears = new Set(manifest.datasets.map(d => d.year));

  renderDatasetList();
  await reloadAllEnabled();

  els.applyBtn.addEventListener('click', applyFilters);
  els.resetBtn.addEventListener('click', resetFilters);

  els.hierarchyBtn.addEventListener('click', e => {
    e.preventDefault();
    openHierarchy();
  });

  els.hierarchyBackdrop.addEventListener('click', closeHierarchy);
  els.hierarchyCloseBtn.addEventListener('click', closeHierarchy);
  els.hierarchyConferenceSelect.addEventListener('change', () => {
    const conference = safe(els.hierarchyConferenceSelect.value);
    if (!conference) return clearHierarchy();
    renderHierarchyForConference(conference);
  });

  els.hierarchyGroupingSelect.addEventListener('change', () => {
    const conference = safe(els.hierarchyConferenceSelect.value);
    if (!conference) return;
    renderHierarchyForConference(conference);
  });

  els.exportBtn.addEventListener('click', e => {
    e.preventDefault();
    exportView();
  });

  if (els.conferenceBackdrop) els.conferenceBackdrop.addEventListener('click', closeConferenceModal);
  if (els.conferenceCloseBtn) els.conferenceCloseBtn.addEventListener('click', closeConferenceModal);
  if (els.conferenceOrgList){
    els.conferenceOrgList.addEventListener('click', event => {
      const card = event.target.closest('[data-org-index]');
      if (!card) return;
      openOrganizationModalByIndex(Number(card.getAttribute('data-org-index')));
    });
  }

  if (els.organizationBackdrop) els.organizationBackdrop.addEventListener('click', closeOrganizationModal);
  if (els.organizationBackBtn) els.organizationBackBtn.addEventListener('click', closeOrganizationModal);
  if (els.organizationCloseBtn) els.organizationCloseBtn.addEventListener('click', closeOrganizationModal);

  window.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (els.organizationModal?.getAttribute('aria-hidden') === 'false') return closeOrganizationModal();
    if (els.conferenceModal?.getAttribute('aria-hidden') === 'false') return closeConferenceModal();
    if (els.hierarchyModal?.getAttribute('aria-hidden') === 'false') return closeHierarchy();
  });

  els.searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') applyFilters();
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
  return combo ? combo.toLowerCase() : "(unknown)";
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
  if (cur && Array.from(selectEl.options).some(o => o.value === cur)){
    selectEl.value = cur;
  } else {
    selectEl.value = "";
  }
}

function computePerYear(rows){
  const byYear = new Map();
  for (const r of rows){
    const y = Number(r.yearbook_year);
    if (!Number.isFinite(y)) continue;
    if (!byYear.has(y)) byYear.set(y, {people:new Set(), conf:new Set(), women:new Set()});
    const b = byYear.get(y);
    const pid = personId(r);
    b.people.add(pid);

    const conf = safe(r.conference);
    if (conf) b.conf.add(conf);

    if (normGender(r.gender) === "female"){
      b.women.add(pid);
    }
  }
  const years = Array.from(byYear.keys()).sort((a,b)=>a-b);
  return years.map(y => ({
    year:y,
    unique_people: byYear.get(y).people.size,
    conferences: byYear.get(y).conf.size,
    women: byYear.get(y).women.size
  }));
}

function renderPerYearTable(rows){
  const table = $("perYearTable");
  const tbody = table ? table.querySelector("tbody") : null;
  if (!tbody) return;
  tbody.innerHTML = "";
  const stats = computePerYear(rows);
  for (const s of stats){
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.year}</td>
      <td>${s.unique_people.toLocaleString()}</td>
      <td>${s.conferences.toLocaleString()}</td>
      <td>${s.women.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  }
}

function hydrateSummaryFilters(rows){
  const years = Array.from(new Set(rows.map(r => Number(r.yearbook_year)).filter(Number.isFinite))).sort((a,b)=>a-b).map(String);
  const confs = uniqSorted(rows.map(r => safe(r.conference)));
  const orgs = uniqSorted(rows.map(r => safe(r.organization)));
  const roles = uniqSorted(rows.map(r => safe(r.position)));
  const genders = uniqSorted(rows.map(r => normGender(r.gender))).filter(Boolean);

  setSelectOptions($("sumYear"), years, {includeAll:true, allLabel:"All years"});
  setSelectOptions($("sumConference"), confs, {includeAll:true, allLabel:"All conferences"});
  setSelectOptions($("sumOrganization"), orgs, {includeAll:true, allLabel:"All organizations"});
  setSelectOptions($("sumRole"), roles, {includeAll:true, allLabel:"All roles"});
  setSelectOptions($("sumGender"), genders, {includeAll:true, allLabel:"All genders"});
}

function applySummaryFilter(rows){
  const year = safe($("sumYear")?.value);
  const conf = safe($("sumConference")?.value);
  const org  = safe($("sumOrganization")?.value);
  const role = safe($("sumRole")?.value);
  const gen  = safe($("sumGender")?.value);

  return rows.filter(r => {
    if (year && String(r.yearbook_year) !== year) return false;
    if (conf && safe(r.conference) !== conf) return false;
    if (org  && safe(r.organization) !== org) return false;
    if (role && safe(r.position) !== role) return false;
    if (gen  && normGender(r.gender) !== gen) return false;
    return true;
  });
}

function renderSummaryKpis(rows){
  const filtered = applySummaryFilter(rows);
  const people = new Set(filtered.map(personId));
  const elPeople = $("sumUniquePeople");
  const elRows = $("sumRows");
  if (elPeople) elPeople.textContent = people.size.toLocaleString();
  if (elRows) elRows.textContent = filtered.length.toLocaleString();

  const ul = $("sumNameSample");
  if (ul){
    ul.innerHTML = "";
    const names = Array.from(new Set(filtered.map(r => safe(r.name)).filter(Boolean)))
      .sort((a,b)=>a.localeCompare(b))
      .slice(0,25);
    for (const n of names){
      const li = document.createElement("li");
      li.textContent = n;
      ul.appendChild(li);
    }
    if (!names.length){
      const li = document.createElement("li");
      li.textContent = "(no names for this filter)";
      ul.appendChild(li);
    }
  }
}

function attachSummaryHandlers(){
  const ids = ["sumYear","sumConference","sumOrganization","sumRole","sumGender"];
  for (const id of ids){
    const el = $(id);
    if (el){
      el.addEventListener("change", () => renderSummaryKpis(allRows));
    }
  }
}

function setSummaryLoadedStat(){
  const el = $("sumStatLoaded");
  if (!el) return;
  const years = Array.from(enabledYears).sort((a,b)=>a-b);
  const label = years.length ? `${years[0]}–${years[years.length-1]}` : "–";
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

    if (!sumOn && timeline){
      setTimeout(() => { try { timeline.redraw(); } catch(e){} }, 50);
    }
  }

  btnSum.addEventListener("click", (e) => { e.preventDefault(); activate("summary"); });
  btnTime.addEventListener("click", (e) => { e.preventDefault(); activate("timeline"); });
}

function refreshSummary(){
  setSummaryLoadedStat();
  renderPerYearTable(allRows);
  hydrateSummaryFilters(allRows);
  renderSummaryKpis(allRows);
}

// Wrap reloadAllEnabled so Summary stays in sync with toggled datasets
const __reloadAllEnabled = reloadAllEnabled;
reloadAllEnabled = async function(){
  await __reloadAllEnabled();
  refreshSummary();
};
