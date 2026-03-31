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
let aggregateTermsConfig = null;

const CONFERENCE_CATEGORY_ORDER = [
  "General",
  "North American - Atlantic / Columbia Union",
  "North American - Lake Union",
  "North American - Atlantic / Lake overlap",
  "North American - Southern Union",
  "North American - Pacific Union",
  "North American - Other",
  "International",
  "Uncategorized",
];

const NORTH_AMERICAN_CONFERENCE_HINTS = new Set([
  "alabama",
  "arkansas",
  "arizona",
  "atlantic",
  "atlantic union",
  "canada",
  "canadian union",
  "carolina",
  "central new england",
  "central union",
  "chesapeake",
  "columbia union",
  "colorado",
  "cumberland",
  "dakota",
  "dakota territory",
  "district of columbia",
  "east kansas",
  "east michigan",
  "eastern canadian union",
  "eastern new york",
  "eastern pennsylvania",
  "florida",
  "georgia",
  "greater new york",
  "idaho",
  "illinois",
  "indiana",
  "iowa",
  "kansas",
  "kentucky",
  "lake union",
  "louisiana",
  "maine",
  "manitoba",
  "maritime",
  "maritime provinces of canada",
  "maryland",
  "massachusetts",
  "michigan",
  "minnesota",
  "mississippi",
  "missouri",
  "montana",
  "montana mission",
  "nebraska",
  "new england",
  "new jersey",
  "new york",
  "newfoundland",
  "north american division",
  "north carolina",
  "north dakota",
  "north michigan",
  "north missouri",
  "north pacific",
  "north wisconsin",
  "northern illinois",
  "northern new england",
  "northern union",
  "nova scotia",
  "ohio",
  "oklahoma and indian territory",
  "ontario",
  "oregon",
  "pennsylvania",
  "pacific union",
  "quebec",
  "south dakota",
  "south missouri",
  "south wisconsin",
  "south carolina",
  "southern",
  "southern district",
  "southern illinois",
  "southern mission",
  "southern missouri",
  "southern new england",
  "southern union",
  "st. louis mission field",
  "superior mission",
  "tennessee",
  "tennessee river",
  "texas",
  "upper columbia",
  "utah",
  "vermont",
  "virginia",
  "washington",
  "west kansas",
  "west michigan",
  "west pennsylvania",
  "west virginia",
  "western colorado",
  "western missouri",
  "western nebraska",
  "western new york",
  "western pennsylvania",
  "wisconsin",
  "wyoming",
]);

const HIERARCHY_ROLE_TIERS = [
  { label: "Presiding leadership", color: "#f4b266" },
  { label: "Vice leadership", color: "#e2c86e" },
  { label: "Secretarial leadership", color: "#7bc7ff" },
  { label: "Financial leadership", color: "#5bc7b1" },
  { label: "Department leadership", color: "#8bb4ff" },
  { label: "Supporting leadership", color: "#b7a6f7" },
  { label: "Workers and members", color: "#8ea0b7" },
];

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
    conferenceRollupAliases: {},
    conferenceCategoryAliases: {},
    organizationExactAliases: {},
    organizationFamilyAliases: {},
    organizationTypeAliases: {},
    personAbbreviationAliases: {},
    personExactAliases: {},
    regionExactAliases: {},
  };

  try {
    const res = await fetch("normalization.json");
    if (!res.ok) return;
    const raw = await res.json();
    normalizationConfig = {
      conferenceExactAliases: mergeNormalizationMaps(raw.conference_exact_aliases),
      conferenceFamilyAliases: mergeNormalizationMaps(raw.conference_family_aliases),
      conferenceRollupAliases: mergeNormalizationMaps(raw.conference_rollup_aliases),
      conferenceCategoryAliases: mergeNormalizationMaps(raw.conference_category_aliases),
      organizationExactAliases: mergeNormalizationMaps(raw.organization_exact_aliases),
      organizationFamilyAliases: mergeNormalizationMaps(raw.organization_family_aliases),
      organizationTypeAliases: mergeNormalizationMaps(raw.organization_type_aliases),
      personAbbreviationAliases: mergeNormalizationMaps(raw.person_abbreviation_aliases),
      personExactAliases: mergeNormalizationMaps(raw.person_exact_aliases),
      regionExactAliases: mergeNormalizationMaps(raw.region_exact_aliases),
    };
  } catch (err) {
    console.warn("normalization.json could not be loaded; using built-in defaults.", err);
  }
}

async function loadAggregateTermsConfig(){
  aggregateTermsConfig = {
    organizationGroupAliases: {},
    organizationGroupOrder: [],
  };

  try {
    const res = await fetch("aggregate_terms_groups.json");
    if (!res.ok) return;
    const raw = await res.json();
    aggregateTermsConfig = {
      organizationGroupAliases: mergeNormalizationMaps(raw.organization_group_aliases),
      organizationGroupOrder: Array.isArray(raw.organization_group_order)
        ? raw.organization_group_order.map(value => normalizeSpacing(value)).filter(Boolean)
        : [],
    };
  } catch (err) {
    console.warn("aggregate_terms_groups.json could not be loaded; using fallback group inference.", err);
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

function normalizePersonToken(token){
  const clean = normalizeSpacing(token);
  if (!clean) return "";
  const direct = normalizationConfig?.personAbbreviationAliases?.[normalizeKey(clean)];
  if (direct) return direct;
  const withoutPeriods = clean.replace(/\.+$/g, "");
  return normalizationConfig?.personAbbreviationAliases?.[normalizeKey(withoutPeriods)] || clean;
}

function normalizePersonName(raw){
  let clean = normalizeSpacing(raw);
  if (!clean) return "";
  clean = applyAlias(normalizationConfig?.personExactAliases, clean);

  const expanded = clean
    .split(" ")
    .map(token => normalizePersonToken(token))
    .join(" ");

  clean = normalizeSpacing(expanded);
  clean = applyAlias(normalizationConfig?.personExactAliases, clean);
  return clean;
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

function fallbackOrganizationGroup(row){
  const type = safe(row.organization_type);
  if (type) {
    if (type === "Conference Tract Society" || type === "Tract Society Department") return "Tract Society";
    if (type === "Tract and Missionary Society") return "Tract and Missionary";
    if (type === "Health and Temperance Association" || type === "Health and Temperance Society") return "Health and Temperance";
    if (type === "Sabbath-School Department" || type === "Sabbath-School Association") return "Sabbath-School";
    if (type === "Publishing Association" || type === "Publishing Department") return "Publishing";
    if (type === "Directory") return "Workers' Directory";
    if (type === "College" || type === "Academy") return "School";
    return type;
  }

  const raw = normalizeKey(
    safe(row.organization_canonical) ||
    safe(row.organization) ||
    safe(row.institution_name)
  );
  if (!raw) return "";
  if (raw.includes("tract and missionary")) return "Tract and Missionary";
  if (raw.includes("tract society")) return "Tract Society";
  if (raw.includes("health and temperance")) return "Health and Temperance";
  if (raw.includes("sabbath-school")) return "Sabbath-School";
  if (raw.includes("publishing")) return "Publishing";
  if (raw.includes("city mission")) return "City Mission";
  if (raw.includes("conference")) return "Conference";
  if (raw.includes("mission")) return "Mission";
  if (raw.includes("school") || raw.includes("college") || raw.includes("academy")) return "School";
  return "";
}

function resolveOrganizationGroup(row){
  const candidates = [
    safe(row.organization),
    safe(row.institution_name),
    safe(row.organization_canonical),
    safe(row.organization_raw),
  ]
    .map(value => normalizeKey(value))
    .filter(Boolean);

  for (const key of candidates){
    const hit = aggregateTermsConfig?.organizationGroupAliases?.[key];
    if (hit) return normalizeSpacing(hit);
  }

  return fallbackOrganizationGroup(row) || safe(row.group_raw) || "";
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

function resolveConferenceCategory(...values){
  let fallback = "";
  for (const value of values){
    const clean = normalizeSpacing(value);
    if (!clean) continue;
    if (!fallback) fallback = clean;
    const match = normalizationConfig?.conferenceCategoryAliases?.[normalizeKey(clean)];
    if (match) return match;
  }
  if (!fallback) return "";
  if (NORTH_AMERICAN_CONFERENCE_HINTS.has(normalizeKey(fallback))) return "North American - Other";
  return "International";
}

function resolveConferenceRollup(...values){
  for (const value of values){
    const clean = normalizeSpacing(value);
    if (!clean) continue;
    const match = normalizationConfig?.conferenceRollupAliases?.[normalizeKey(clean)];
    if (match) return match;
  }
  return normalizeSpacing(values.find(value => normalizeSpacing(value)) || "");
}

function conferenceCategoryRank(category){
  const idx = CONFERENCE_CATEGORY_ORDER.indexOf(category || "Uncategorized");
  return idx === -1 ? CONFERENCE_CATEGORY_ORDER.length : idx;
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
    const label = safe(row.conference_rollup) || safe(row.conference_canonical) || safe(row.conference);
    if (!label) continue;
    const entity = {
      key: makeEntityKey("conference", label),
      label,
      family: safe(row.conference_family) || label,
      category: safe(row.conference_category) || "Uncategorized",
      type: inferConferenceEntityType(row),
    };
    if (!conferenceMatchesTimelineFilters(entity)) continue;
    if (!byKey.has(entity.key)) {
      byKey.set(entity.key, {
        key: entity.key,
        label: entity.label,
        family: entity.family,
        category: entity.category,
        familyOrder: timelineFamilyOrder(entity.family),
        scope: "conference",
        type: entity.type,
      });
    }
  }
  return Array.from(byKey.values()).sort((a, b) =>
    conferenceCategoryRank(a.category) - conferenceCategoryRank(b.category) ||
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

    const conferenceLabel = safe(row.conference_rollup) || safe(row.conference_canonical) || safe(row.conference);
    const sourceConference = safe(row.conference_canonical) || safe(row.conference);
    const relatedOrganization = safe(row.organization_canonical) || safe(row.organization) || safe(row.institution_name);
    if (!conferenceLabel) continue;

    const entry = {
      key: makeEntityKey("conference", conferenceLabel),
      label: conferenceLabel,
      scope: "conference",
      type: inferConferenceEntityType(row),
      family: safe(row.conference_family) || conferenceLabel,
      category: safe(row.conference_category) || "Uncategorized",
    };
    if (!conferenceMatchesTimelineFilters(entry)) continue;

    if (!byConference.has(entry.key)) {
      byConference.set(entry.key, {
        ...entry,
        regions: new Set(),
        relatedConferences: new Set(),
        relatedOrganizations: new Set(),
        years: new Set(),
        rowsByYear: new Map(),
      });
    }

    const bucket = byConference.get(entry.key);
    bucket.regions.add(regionLabelForRow(row));
    if (sourceConference) bucket.relatedConferences.add(sourceConference);
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
      category: entry.category,
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
        id: `${entry.key}-connector-${range.start}-${range.end}`,
        content: "",
        start: yearToDate(range.start),
        end: yearToDate(range.end + 1),
        type: "range",
        group: entry.label,
        className: `timelineItem timelineConnector timelineItem--scope-conference timelineItem--${slugifyToken(entry.type || "default")}`,
        title: `${entry.label}\nActive years: ${range.start}–${range.end}\nHover a point for the exact year, then click that point to open the organization view.`,
        _entityKey: entry.key,
        _label: entry.label,
        _scope: entry.scope,
        _entityType: entry.type,
        _family: entry.family,
        _category: entry.category,
        _regions: Array.from(entry.regions).sort((a, b) => a.localeCompare(b)),
        _relatedConferences: Array.from(entry.relatedConferences).sort((a, b) => a.localeCompare(b)),
        _relatedOrganizations: Array.from(entry.relatedOrganizations).sort((a, b) => a.localeCompare(b)),
        _startYear: range.start,
        _endYear: range.end,
        _rowsByYear: segmentRowsByYear,
        _itemKind: "connector",
      });
    }

    years.forEach(year => {
      const yearRows = entry.rowsByYear.get(year) || [];
      itemObjs.push({
        id: `${entry.key}-point-${year}`,
        content: `<span class="timelinePoint__dot" data-year="${escapeHtml(String(year))}" aria-hidden="true"></span>`,
        start: yearToDate(year),
        type: "point",
        group: entry.label,
        className: `timelineItem timelinePoint timelineItem--scope-conference timelineItem--${slugifyToken(entry.type || "default")}`,
        title: `${entry.label}\nYear: ${year}\nClick to open the organization view for this exact year.`,
        _entityKey: entry.key,
        _label: entry.label,
        _scope: entry.scope,
        _entityType: entry.type,
        _family: entry.family,
        _category: entry.category,
        _regions: Array.from(entry.regions).sort((a, b) => a.localeCompare(b)),
        _relatedConferences: Array.from(entry.relatedConferences).sort((a, b) => a.localeCompare(b)),
        _relatedOrganizations: Array.from(entry.relatedOrganizations).sort((a, b) => a.localeCompare(b)),
        _startYear: year,
        _endYear: year,
        _rowsByYear: { [String(year)]: yearRows },
        _clickYear: year,
        _itemKind: "point",
      });
    });
  }

  lastTimelineEntities.sort((a, b) =>
    conferenceCategoryRank(a.category) - conferenceCategoryRank(b.category) ||
    timelineFamilyOrder(a.family) - timelineFamilyOrder(b.family) ||
    a.family.localeCompare(b.family) ||
    a.label.localeCompare(b.label)
  );

  return itemObjs.sort((a, b) =>
    conferenceCategoryRank(a._category) - conferenceCategoryRank(b._category) ||
    timelineFamilyOrder(a._family) - timelineFamilyOrder(b._family) ||
    a._family.localeCompare(b._family) ||
    String(a._itemKind || "").localeCompare(String(b._itemKind || "")) ||
    a._startYear - b._startYear
  );
}

function timelineRowsForYear(item, year){
  if (!item) return [];
  return item._rowsByYear?.[String(year)] || [];
}

function resolveTimelineClickYear(item, clickedDate){
  if (!item) return null;
  if (Number.isFinite(item._clickYear)) return item._clickYear;
  const clickedYear = clickedDate instanceof Date && Number.isFinite(clickedDate.getFullYear())
    ? clickedDate.getFullYear()
    : item._startYear;
  const activeYears = Object.keys(item._rowsByYear || {})
    .map(year => Number(year))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!activeYears.length) {
    return Number.isFinite(clickedYear)
      ? Math.max(item._startYear, Math.min(item._endYear, clickedYear))
      : item._startYear;
  }
  return activeYears.reduce((best, year) => {
    if (best === null) return year;
    return Math.abs(year - clickedYear) < Math.abs(best - clickedYear) ? year : best;
  }, null);
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
    els.detailCard.textContent = "Hover or click a point on the timeline to inspect a conference at a specific year.";
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
  const sourceConferenceLabel = Array.from(confCounts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 6)
    .map(([conference, count]) => `${conference} (${count})`);

  const fields = [
    ["Entity", item._label],
    ["Category", item._category || "Uncategorized"],
    ["Family", item._family],
    ["Scope", "conference"],
    ["Type", item._entityType || "Unspecified"],
    ["Active years", activeYears],
    ["Regions", summarizeValues(item._regions || [])],
    selectedYear !== null ? ["Selected year", String(selectedYear)] : null,
    selectedYear !== null ? ["Regions in year", summarizeValues(yearRegions)] : null,
    selectedYear !== null ? ["Rows in year", yearRows.length.toLocaleString()] : null,
    selectedYear !== null ? ["Named people in year", countNamedPeople(yearRows).toLocaleString()] : null,
    selectedYear !== null ? ["Organization groups in year", summarizeValues(yearRows.map(groupLabelForRow))] : null,
    sourceConferenceLabel.length ? ["Source conferences in year", sourceConferenceLabel.join(" • ")] : null,
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

function majorPlayerTone(key, gender){
  const baseTone = genderTone(gender);
  let hash = 0;
  for (const ch of String(key || "")){
    hash = ((hash << 5) - hash) + ch.charCodeAt(0);
    hash |= 0;
  }
  const shift = Math.abs(hash) % 12;
  return {
    bg: baseTone.bg,
    border: baseTone.border,
    glow: baseTone.glow,
    accent: `translateX(${shift - 6}px)`,
    label: genderBucketLabel(gender),
  };
}

function renderTimelineMajorPlayers(rows, opts = {}){
  const host = els.timelineMajorPlayers;
  if (!host) return;

  const yearMin = Number.isFinite(Number(opts.yearMin)) ? Number(opts.yearMin) : null;
  const yearMax = Number.isFinite(Number(opts.yearMax)) ? Number(opts.yearMax) : null;

  if (!rows.length || !Number.isFinite(yearMin) || !Number.isFinite(yearMax) || yearMin > yearMax){
    host.innerHTML = `<div class="majorPlayersEmpty">No rows match the current timeline filters.</div>`;
    return;
  }

  const rowsByYear = new Map();
  for (const row of rows){
    const year = Number(row.yearbook_year);
    if (!Number.isFinite(year) || year < yearMin || year > yearMax) continue;
    if (!rowsByYear.has(year)) rowsByYear.set(year, []);
    rowsByYear.get(year).push(row);
  }

  const activeYears = Array.from(rowsByYear.keys()).sort((a, b) => a - b);
  if (!activeYears.length){
    host.innerHTML = `<div class="majorPlayersEmpty">No rows match the current timeline filters.</div>`;
    return;
  }

  const yearlyLeaders = [];
  let maxRoleCount = 1;
  for (const year of activeYears){
    const yearRows = rowsByYear.get(year) || [];
    const people = new Map();
    for (const row of yearRows){
      if (!isLikelyNamedIndividual(row)) continue;
      const pid = personId(row);
      if (!pid) continue;
      if (!people.has(pid)) {
        people.set(pid, {
          id: pid,
          label: personLabel(row),
          roleCount: 0,
          gender: "",
        });
      }
      const person = people.get(pid);
      person.roleCount += 1;
      const normalizedGender = normGender(row.gender);
      if (!person.gender && normalizedGender) person.gender = normalizedGender;
      else if (normalizedGender && person.gender && person.gender !== normalizedGender) person.gender = "";
    }

    const leaders = Array.from(people.values())
      .sort((a, b) => b.roleCount - a.roleCount || a.label.localeCompare(b.label))
      .slice(0, 4);

    leaders.forEach(person => {
      if (person.roleCount > maxRoleCount) maxRoleCount = person.roleCount;
    });

    yearlyLeaders.push({
      year,
      rowCount: yearRows.length,
      peopleCount: people.size,
      leaders,
    });
  }

  host.innerHTML = `
    <div class="majorPlayersGrid">
      ${yearlyLeaders.map(entry => {
        const chips = entry.leaders.length
          ? entry.leaders.map(person => {
              const tone = majorPlayerTone(person.id, person.gender);
              const width = Math.max(24, Math.round((person.roleCount / maxRoleCount) * 100));
              return `
                <button
                  type="button"
                  class="majorPlayerChip"
                  data-player-name="${escapeHtml(person.label)}"
                  title="${escapeHtml(`${person.label} • ${person.roleCount} matching roles in ${entry.year} • ${tone.label}`)}"
                  style="--major-bg:${tone.bg}; --major-border:${tone.border}; --major-glow:${tone.glow};"
                >
                  <span class="majorPlayerChip__gender">${escapeHtml(tone.label)}</span>
                  <span class="majorPlayerChip__name">${escapeHtml(person.label)}</span>
                  <span class="majorPlayerChip__count">${person.roleCount.toLocaleString()} role${person.roleCount === 1 ? "" : "s"}</span>
                  <span class="majorPlayerChip__bar" style="width:${width}%"></span>
                </button>
              `;
            }).join("")
          : `<div class="majorPlayersEmpty">No named individuals</div>`;

        return `
          <article class="majorYearCard">
            <div class="majorYearCard__year">${entry.year}</div>
            <div class="majorYearCard__meta">${entry.peopleCount.toLocaleString()} named individual${entry.peopleCount === 1 ? "" : "s"}</div>
            <div class="majorYearCard__list">${chips}</div>
          </article>
        `;
      }).join("")}
    </div>
  `;

  host.querySelectorAll("[data-player-name]").forEach(button => {
    button.addEventListener("click", () => {
      const playerName = safe(button.getAttribute("data-player-name"));
      if (!playerName || !els.searchInput) return;
      els.searchInput.value = playerName;
      applyFilters();
    });
  });
}

function hydrateFilterOptions(){
  const regions = uniqSorted(allRows.map(r => safe(r.region_normalized) || safe(r.region)));
  const confs = uniqSorted(allRows.map(r => safe(r.conference_rollup) || safe(r.conference_canonical) || safe(r.conference)));
  const positions = uniqSorted(allRows.map(r => r.position));
  const genders = uniqSorted(allRows.map(r => r.gender));
  const entityTypes = uniqSorted(allRows.map(row => inferConferenceEntityType(row)).filter(Boolean));
  const groups = orderedOrganizationGroups(allRows);
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
  const prevGroup = els.groupSelect?.value || "";
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

  if (els.groupSelect){
    els.groupSelect.innerHTML =
      '<option value="">All organization groups</option>' +
      groups.map(group => `<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`).join("");
  }

  if (regions.includes(prevRegion)) els.regionSelect.value = prevRegion;
  if (confs.includes(prevConf)) els.confSelect.value = prevConf;
  if (els.positionSelect && positions.includes(prevPos)) els.positionSelect.value = prevPos;
  if (els.genderSelect && genders.includes(prevGender)) els.genderSelect.value = prevGender;
  if (els.entityTypeSelect && entityTypes.includes(prevEntityType)) els.entityTypeSelect.value = prevEntityType;
  if (els.orgSelect && orgs.includes(prevOrg)) els.orgSelect.value = prevOrg;
  if (els.groupSelect && groups.includes(prevGroup)) els.groupSelect.value = prevGroup;
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
  const groupSel = safe(els.groupSelect?.value);
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
    if (conf && (safe(r.conference_rollup) || safe(r.conference_canonical) || safe(r.conference)) !== conf) return false;

    if (pos && safe(r.position) !== pos) return false;
    if (gender && safe(r.gender) !== gender) return false;

    if (orgSel){
      const orgLabel = safe(r.organization_canonical) || safe(r.organization) || safe(r.institution_name);
      if (safe(orgLabel) !== orgSel) return false;
    }

    if (groupSel && groupLabelForRow(r) !== groupSel) return false;

    return true;
  });

  lastFiltered = filtered;

  const itemObjs = buildEntityTimelineItems(filtered);
  items.clear();
  items.add(itemObjs);

  const presentConferences = uniqSorted(itemObjs.map(it => it.group));
  const groups = presentConferences
    .sort((a, b) => a.localeCompare(b))
    .map((conference, index) => ({
      id: conference,
      content: `
        <div class="timelineGroupLabel">
          <div class="timelineGroupLabel__title">${escapeHtml(conference)}</div>
        </div>
      `,
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
  renderOrganizationGroupMix("timelineGroupPulse", filtered, {
    topN: 6,
    compact: true,
    emptyMessage: "No organization-group activity is available for the current timeline filters.",
  });
  renderTimelineMajorPlayers(filtered, { yearMin: yMin, yearMax: yMax });

  if (els.hierarchyModal && els.hierarchyModal.getAttribute("aria-hidden") === "false"){
    refreshHierarchyControls();
  }
}

function resetFilters(){
  els.searchInput.value = "";
  els.regionSelect.value = "";
  els.confSelect.value = "";
  if (els.positionSelect) els.positionSelect.value = "";
  if (els.genderSelect) els.genderSelect.value = "";
  if (els.orgSelect) els.orgSelect.value = "";
  if (els.groupSelect) els.groupSelect.value = "";
  if (els.entityTypeSelect) els.entityTypeSelect.value = "";
  els.yearMin.value = "1883";
  els.yearMax.value = "1921";
  applyFilters();
}


function hierarchyConferenceLabel(row){
  return safe(row.conference_rollup) || safe(row.conference_canonical) || safe(row.conference) || "(No conference)";
}

function hierarchyOrganizationLabel(row){
  return safe(row.organization_canonical) || safe(row.organization) || safe(row.institution_name) || "(No organization)";
}

function parseOptionalNumber(value){
  const clean = safe(value);
  if (!clean) return null;
  const num = Number(clean);
  return Number.isFinite(num) ? num : null;
}

function hierarchySelectState(){
  return {
    year: parseOptionalNumber(els.hierarchyYearSelect?.value),
    conference: safe(els.hierarchyConferenceSelect?.value),
    organization: safe(els.hierarchyOrganizationSelect?.value),
  };
}

function setHierarchySelectOptions(selectEl, options, placeholder, preferredValue = ""){
  if (!selectEl) return "";
  const current = safe(preferredValue) || safe(selectEl.value);
  selectEl.innerHTML =
    `<option value="">${escapeHtml(placeholder)}</option>` +
    options.map(option => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`).join("");
  const resolved = options.some(option => option.value === current) ? current : "";
  selectEl.value = resolved;
  return resolved;
}

function buildHierarchyYearOptions(rows){
  const years = Array.from(new Set(
    rows.map(row => Number(row.yearbook_year)).filter(Number.isFinite)
  )).sort((a, b) => a - b);
  return years.map(year => ({ value: String(year), label: String(year) }));
}

function buildHierarchyConferenceOptions(rows, year){
  if (!Number.isFinite(year)) return [];
  const byLabel = new Map();
  rows
    .filter(row => Number(row.yearbook_year) === year)
    .forEach(row => {
      const label = hierarchyConferenceLabel(row);
      if (!label) return;
      const current = byLabel.get(label) || { label, count: 0, category: safe(row.conference_category) || "Uncategorized" };
      current.count += 1;
      byLabel.set(label, current);
    });

  return Array.from(byLabel.values())
    .sort((a, b) =>
      conferenceCategoryRank(a.category) - conferenceCategoryRank(b.category) ||
      b.count - a.count ||
      a.label.localeCompare(b.label)
    )
    .map(item => ({
      value: item.label,
      label: `${item.label} (${item.count})`,
    }));
}

function buildHierarchyOrganizationOptions(rows, year, conference){
  if (!Number.isFinite(year) || !safe(conference)) return [];
  const byLabel = new Map();
  rows
    .filter(row => Number(row.yearbook_year) === year && hierarchyConferenceLabel(row) === conference)
    .forEach(row => {
      const label = hierarchyOrganizationLabel(row);
      if (!label) return;
      const current = byLabel.get(label) || { label, count: 0 };
      current.count += 1;
      byLabel.set(label, current);
    });

  return Array.from(byLabel.values())
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
    .map(item => ({
      value: item.label,
      label: `${item.label} (${item.count})`,
    }));
}

function hierarchyPromptMessage(state = hierarchySelectState()){
  const missing = [];
  if (!Number.isFinite(state.year)) missing.push("a single year");
  if (!state.conference) missing.push("a single conference rollup");
  if (!state.organization) missing.push("a single organization");
  if (!missing.length) return "";
  if (missing.length === 1) return `Choose ${missing[0]} to render the people network.`;
  if (missing.length === 2) return `Choose ${missing[0]} and ${missing[1]} to render the people network.`;
  return `Choose ${missing[0]}, ${missing[1]}, and ${missing[2]} to render the people network.`;
}

function clearHierarchy(message = ""){
  const text = safe(message) || "Choose a single year, conference rollup, and organization to render the people network.";
  els.hierarchyChart.innerHTML = `
    <div class="hierEmptyState">
      <div>
        <div class="hierEmptyState__title">Hierarchy graph ready when the slice is fully specified</div>
        <div class="hint">${escapeHtml(text)}</div>
      </div>
    </div>
  `;
  hierarchyPinned = null;
  els.hierarchyDetail.classList.add("muted");
  els.hierarchyDetail.textContent = text;
}

function normalizePositionKey(position){
  return safe(position).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function classifyHierarchyRole(position){
  const value = normalizePositionKey(position);
  if (!value) return { rank: 6, tierLabel: HIERARCHY_ROLE_TIERS[6].label };
  const hasAny = (...tokens) => tokens.some(token => value.includes(token));
  const hasAll = (...tokens) => tokens.every(token => value.includes(token));
  const hasSupportQualifier = hasAny("assistant", "associate", "deputy");

  if (hasAny("vice president", "vice-president", "vice chairman", "vice-chairman", "vice chair", "vice-chair")) {
    return { rank: 1, tierLabel: HIERARCHY_ROLE_TIERS[1].label };
  }
  if (hasSupportQualifier && hasAny("president", "chairman", "superintendent", "principal", "secretary", "treasurer", "auditor", "director", "manager", "editor", "dean", "leader")) {
    return { rank: 5, tierLabel: HIERARCHY_ROLE_TIERS[5].label };
  }
  if (hasAny("president", "chairman", "chair woman", "chairwoman", "chair person", "chairperson", "superintendent", "principal", "chief")) {
    return { rank: 0, tierLabel: HIERARCHY_ROLE_TIERS[0].label };
  }
  if (hasAny("secretary", "clerk") || hasAll("corresponding", "secretary") || hasAll("recording", "secretary")) {
    return { rank: 2, tierLabel: HIERARCHY_ROLE_TIERS[2].label };
  }
  if (hasAny("treasurer", "auditor", "cashier", "comptroller", "bursar")) {
    return { rank: 3, tierLabel: HIERARCHY_ROLE_TIERS[3].label };
  }
  if (hasAny("director", "manager", "editor", "dean", "head", "leader")) {
    return { rank: 4, tierLabel: HIERARCHY_ROLE_TIERS[4].label };
  }
  if (hasAny("assistant", "associate", "vice", "deputy", "agent")) {
    return { rank: 5, tierLabel: HIERARCHY_ROLE_TIERS[5].label };
  }
  return { rank: 6, tierLabel: HIERARCHY_ROLE_TIERS[6].label };
}

function truncateLabel(text, maxChars = 26){
  const clean = safe(text);
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, Math.max(0, maxChars - 1)).trim()}…`;
}

function splitHierarchyLabel(label, maxChars = 18, maxLines = 2){
  const words = safe(label).split(/\s+/).filter(Boolean);
  if (!words.length) return ["(unnamed)"];
  const lines = [];
  let current = "";
  for (const word of words){
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxChars || !current){
      current = candidate;
      continue;
    }
    lines.push(current);
    current = word;
    if (lines.length === maxLines - 1) break;
  }
  if (current){
    const consumedWords = lines.join(" ").split(/\s+/).filter(Boolean).length;
    const remainder = words.slice(consumedWords);
    const finalLine = remainder.length ? remainder.join(" ") : current;
    lines.push(truncateLabel(finalLine, maxChars));
  }
  return lines.slice(0, maxLines);
}

function buildOrganizationNetworkData(opts = {}){
  const year = Number(opts.year);
  const conference = safe(opts.conference);
  const organization = safe(opts.organization);
  const sourceRows = Array.isArray(opts.rows) ? opts.rows : lastFiltered;

  const rows = sourceRows.filter(row =>
    Number(row.yearbook_year) === year &&
    hierarchyConferenceLabel(row) === conference &&
    hierarchyOrganizationLabel(row) === organization
  );

  const people = new Map();
  for (const row of rows){
    if (!isLikelyNamedIndividual(row)) continue;
    const label = personLabel(row);
    if (!label || label === "(unnamed)") continue;
    const personKey = personId(row) || `${label.toLowerCase()}__${safe(row.page)}__${safe(row.position)}`;
    if (!people.has(personKey)) {
      people.set(personKey, {
        id: personKey,
        label,
        positions: new Set(),
        sourceConferences: new Set(),
        regions: new Set(),
        locations: new Set(),
        institutions: new Set(),
        pages: new Set(),
        genders: new Set(),
        rows: [],
        bestRank: 6,
        tierLabel: HIERARCHY_ROLE_TIERS[6].label,
        bestPosition: "",
      });
    }
    const person = people.get(personKey);
    const roleInfo = classifyHierarchyRole(row.position);
    person.positions.add(safe(row.position) || "(No position)");
    person.sourceConferences.add(safe(row.conference_canonical) || safe(row.conference) || "(No conference)");
    person.regions.add(regionLabelForRow(row));
    if (safe(row.location)) person.locations.add(safe(row.location));
    if (safe(row.institution_name)) person.institutions.add(safe(row.institution_name));
    if (safe(row.page)) person.pages.add(String(row.page));
    if (safe(row.gender)) person.genders.add(safe(row.gender));
    person.rows.push(row);
    if (roleInfo.rank < person.bestRank || (!person.bestPosition && safe(row.position))) {
      person.bestRank = roleInfo.rank;
      person.tierLabel = roleInfo.tierLabel;
      person.bestPosition = safe(row.position) || "(No position)";
    }
  }

  const nodes = Array.from(people.values())
    .map(person => {
      const positions = Array.from(person.positions).sort((a, b) => a.localeCompare(b));
      return {
        id: person.id,
        label: person.label,
        bestRank: person.bestRank,
        tierLabel: person.tierLabel,
        bestPosition: person.bestPosition || positions[0] || "(No position)",
        positions,
        sourceConferences: Array.from(person.sourceConferences).sort((a, b) => a.localeCompare(b)),
        regions: Array.from(person.regions).sort((a, b) => a.localeCompare(b)),
        locations: Array.from(person.locations).sort((a, b) => a.localeCompare(b)),
        institutions: Array.from(person.institutions).sort((a, b) => a.localeCompare(b)),
        pages: Array.from(person.pages).sort((a, b) => Number(a) - Number(b)),
        genders: Array.from(person.genders).sort((a, b) => a.localeCompare(b)),
        rowCount: person.rows.length,
        rows: person.rows,
      };
    })
    .sort((a, b) =>
      a.bestRank - b.bestRank ||
      b.rowCount - a.rowCount ||
      a.label.localeCompare(b.label)
    );

  const rankGroups = [];
  for (let rank = 0; rank < HIERARCHY_ROLE_TIERS.length; rank += 1){
    const members = nodes.filter(node => node.bestRank === rank);
    if (!members.length) continue;
    rankGroups.push({
      rank,
      label: HIERARCHY_ROLE_TIERS[rank].label,
      color: HIERARCHY_ROLE_TIERS[rank].color,
      nodes: members,
    });
  }

  const links = [];
  for (let index = 1; index < rankGroups.length; index += 1){
    const parentGroup = rankGroups[index - 1];
    const childGroup = rankGroups[index];
    childGroup.nodes.forEach((node, nodeIndex) => {
      const parentIndex = parentGroup.nodes.length === 1 || childGroup.nodes.length === 1
        ? Math.min(nodeIndex, parentGroup.nodes.length - 1)
        : Math.round((nodeIndex / (childGroup.nodes.length - 1)) * (parentGroup.nodes.length - 1));
      const parent = parentGroup.nodes[Math.max(0, parentIndex)];
      if (parent) {
        links.push({
          source: parent.id,
          target: node.id,
          sourceLabel: parent.label,
          targetLabel: node.label,
        });
      }
    });
  }

  const sampleRow = rows[0] || null;
  return {
    year,
    conference,
    organization,
    rows,
    rowCount: rows.length,
    personCount: nodes.length,
    positionCount: uniqSorted(rows.map(row => safe(row.position) || "(No position)")).length,
    regions: uniqSorted(rows.map(regionLabelForRow)),
    sourceConferences: uniqSorted(rows.map(row => safe(row.conference_canonical) || safe(row.conference) || "(No conference)")),
    family: sampleRow ? (safe(sampleRow.conference_family) || conference) : conference,
    category: sampleRow ? (safe(sampleRow.conference_category) || "Uncategorized") : "Uncategorized",
    entityType: sampleRow ? inferConferenceEntityType(sampleRow) : "",
    nodes,
    rankGroups,
    links,
  };
}

function renderHierarchyRow(person, data, hint = ""){
  const fields = [
    ["Best-ranked role", person.bestPosition],
    ["All positions", summarizeValues(person.positions, 8)],
    ["Organization", data.organization],
    ["Conference rollup", data.conference],
    ["Source conferences", summarizeValues(person.sourceConferences, 8)],
    ["Year", String(data.year)],
    ["Regions", summarizeValues(person.regions, 8)],
    ["Locations", summarizeValues(person.locations, 6)],
    ["Institutions", summarizeValues(person.institutions, 6)],
    ["Pages", summarizeValues(person.pages, 8)],
    ["Matching rows", person.rowCount.toLocaleString()],
  ];
  renderHierarchyDetailCard(person.label, fields, hint);
}

function renderHierarchyNetwork(data){
  const wrap = els.hierarchyChart;
  wrap.innerHTML = "";

  if (!data.rows.length || !data.nodes.length){
    clearHierarchy("No named people matched this exact year, conference rollup, and organization slice.");
    renderHierarchyDetailCard(
      data.organization || "(No organization)",
      [
        ["Conference rollup", data.conference || "(No conference)"],
        ["Year", Number.isFinite(data.year) ? String(data.year) : ""],
        ["Matching rows", data.rowCount.toLocaleString()],
      ],
      "Try a different organization or relax the current filters if this slice should contain people."
    );
    return;
  }

  wrap.innerHTML = `
    <div class="hierSummaryBar">
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Organization</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.organization)}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Conference Rollup</span>
        <span class="hierSummaryStat__value">${escapeHtml(data.conference)}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Year</span>
        <span class="hierSummaryStat__value">${escapeHtml(String(data.year))}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">People</span>
        <span class="hierSummaryStat__value">${data.personCount.toLocaleString()}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Positions</span>
        <span class="hierSummaryStat__value">${data.positionCount.toLocaleString()}</span>
      </div>
      <div class="hierSummaryStat">
        <span class="hierSummaryStat__label">Source Conferences</span>
        <span class="hierSummaryStat__value">${escapeHtml(summarizeValues(data.sourceConferences, 3) || "Unknown")}</span>
      </div>
    </div>
    <div class="hierGraphSurface" id="hierarchyGraphSurface"></div>
  `;

  const graphSurface = $("hierarchyGraphSurface");
  const maxNodesInTier = Math.max(...data.rankGroups.map(group => group.nodes.length), 1);
  const width = Math.max(graphSurface.clientWidth || 760, maxNodesInTier * 150 + 220);
  const laneHeight = 132;
  const topPadding = 72;
  const bottomPadding = 72;
  const sidePadding = 132;
  const innerWidth = Math.max(220, width - sidePadding * 2);
  const height = Math.max(380, topPadding + bottomPadding + Math.max(1, data.rankGroups.length - 1) * laneHeight + 60);

  data.rankGroups.forEach((group, groupIndex) => {
    const y = topPadding + groupIndex * laneHeight;
    const nodeCount = group.nodes.length;
    group.nodes.forEach((node, nodeIndex) => {
      node.x = nodeCount === 1 ? width / 2 : sidePadding + (innerWidth * nodeIndex) / Math.max(1, nodeCount - 1);
      node.y = y;
      node.radius = 17 + Math.min(7, node.rowCount - 1) + Math.min(4, node.positions.length - 1);
    });
  });

  const nodeById = new Map(data.nodes.map(node => [node.id, node]));
  const links = data.links
    .map(link => ({
      source: nodeById.get(link.source),
      target: nodeById.get(link.target),
    }))
    .filter(link => link.source && link.target);

  const svg = d3.select(graphSurface)
    .append("svg")
    .attr("class", "hierNetworkSvg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  const laneGroup = svg.append("g").attr("class", "hierLanes");
  laneGroup.selectAll("rect")
    .data(data.rankGroups)
    .join("rect")
    .attr("class", "hierLane")
    .attr("x", 30)
    .attr("y", d => d.nodes[0].y - 48)
    .attr("width", width - 60)
    .attr("height", 96)
    .attr("rx", 18)
    .attr("fill", d => `${d.color}18`)
    .attr("stroke", d => `${d.color}66`);

  laneGroup.selectAll("text")
    .data(data.rankGroups)
    .join("text")
    .attr("class", "hierLaneLabel")
    .attr("x", 46)
    .attr("y", d => d.nodes[0].y - 22)
    .text(d => d.label);

  svg.append("g")
    .attr("class", "hierLinks")
    .selectAll("path")
    .data(links)
    .join("path")
    .attr("class", "hlink")
    .attr("d", d => {
      const midY = (d.source.y + d.target.y) / 2;
      return `M ${d.source.x} ${d.source.y + d.source.radius} C ${d.source.x} ${midY}, ${d.target.x} ${midY}, ${d.target.x} ${d.target.y - d.target.radius}`;
    });

  const nodeSelection = svg.append("g")
    .attr("class", "hierNodes")
    .selectAll("g")
    .data(data.nodes)
    .join("g")
    .attr("class", "hierNode")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .attr("tabindex", 0)
    .on("click", (_, node) => selectHierarchyNode(node))
    .on("keydown", (event, node) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectHierarchyNode(node);
      }
    });

  nodeSelection.append("circle")
    .attr("r", d => d.radius)
    .attr("fill", d => HIERARCHY_ROLE_TIERS[d.bestRank]?.color || "#8ea0b7")
    .attr("fill-opacity", 0.2)
    .attr("stroke", d => HIERARCHY_ROLE_TIERS[d.bestRank]?.color || "#8ea0b7")
    .attr("stroke-width", 2);

  nodeSelection.append("title")
    .text(d => `${d.label}\n${d.positions.join(" • ")}\n${d.rowCount} matching row${d.rowCount === 1 ? "" : "s"}`);

  nodeSelection.each(function(node){
    const group = d3.select(this);
    const labelLines = splitHierarchyLabel(node.label, 18, 2);
    const nameText = group.append("text")
      .attr("class", "hierNodeLabel")
      .attr("text-anchor", "middle")
      .attr("y", node.radius + 16);
    labelLines.forEach((line, index) => {
      nameText.append("tspan")
        .attr("x", 0)
        .attr("dy", index === 0 ? 0 : 14)
        .text(line);
    });

    group.append("text")
      .attr("class", "hierNodeRole")
      .attr("text-anchor", "middle")
      .attr("y", node.radius + 16 + labelLines.length * 14 + 14)
      .text(truncateLabel(node.bestPosition, 24));
  });

  function setNodeSelection(selectedId){
    nodeSelection.classed("is-selected", node => node.id === selectedId);
  }

  function selectHierarchyNode(node){
    hierarchyPinned = node.id;
    setNodeSelection(node.id);
    renderHierarchyRow(
      node,
      data,
      "Role rank and links are inferred from position titles in this slice, so this is a guided leadership view rather than an explicit org chart from the source."
    );
  }

  const defaultNode = hierarchyPinned && nodeById.has(hierarchyPinned)
    ? nodeById.get(hierarchyPinned)
    : data.nodes[0];
  hierarchyPinned = defaultNode?.id || null;
  if (defaultNode) {
    setNodeSelection(defaultNode.id);
    renderHierarchyRow(
      defaultNode,
      data,
      "Role rank and links are inferred from position titles in this slice, so this is a guided leadership view rather than an explicit org chart from the source."
    );
  }
}

function refreshHierarchyControls(opts = {}){
  const current = hierarchySelectState();
  const preferredYear = Number.isFinite(Number(opts.year)) ? String(Number(opts.year)) : (Number.isFinite(current.year) ? String(current.year) : "");
  const parsedKey = safe(opts.entityKey) ? parseEntityKey(opts.entityKey).label : "";
  const preferredConference = safe(opts.conference) || parsedKey || current.conference;
  const preferredOrganization = Object.prototype.hasOwnProperty.call(opts, "organization")
    ? safe(opts.organization)
    : current.organization;

  const yearOptions = buildHierarchyYearOptions(lastFiltered);
  const selectedYear = parseOptionalNumber(setHierarchySelectOptions(
    els.hierarchyYearSelect,
    yearOptions,
    "Select a year",
    preferredYear
  ));

  const resolvedYear = Number.isFinite(selectedYear) ? selectedYear : null;
  const conferenceOptions = buildHierarchyConferenceOptions(lastFiltered, resolvedYear);
  const selectedConference = setHierarchySelectOptions(
    els.hierarchyConferenceSelect,
    conferenceOptions,
    "Select a conference rollup",
    preferredConference
  );

  const organizationOptions = buildHierarchyOrganizationOptions(lastFiltered, resolvedYear, selectedConference);
  const selectedOrganization = setHierarchySelectOptions(
    els.hierarchyOrganizationSelect,
    organizationOptions,
    "Select an organization",
    preferredOrganization
  );

  hierarchyYear = resolvedYear;
  const state = {
    year: resolvedYear,
    conference: selectedConference,
    organization: selectedOrganization,
  };

  if (!Number.isFinite(state.year) || !state.conference || !state.organization) {
    hierarchyPinned = null;
    clearHierarchy(hierarchyPromptMessage(state));
    return;
  }

  renderHierarchyNetwork(buildOrganizationNetworkData(state));
}

function openHierarchy(opts = {}){
  hierarchyPinned = null;
  els.hierarchyModal.setAttribute("aria-hidden", "false");
  refreshHierarchyControls(opts);
}

function closeHierarchy(){
  els.hierarchyModal.setAttribute("aria-hidden", "true");
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
  out.conference_rollup = resolveConferenceRollup(conferenceNorm.canonical, conferenceNorm.family, out.conference);
  out.conference_category = resolveConferenceCategory(out.conference_rollup, conferenceNorm.canonical, conferenceNorm.family, out.conference);

  const organizationSource = safe(out.organization) || safe(out.institution_name);
  const organizationNorm = normalizeOrganizationEntity(organizationSource);
  out.organization_raw = normalizeSpacing(out.organization);
  out.organization = normalizeSpacing(out.organization);
  out.institution_name = normalizeSpacing(out.institution_name);
  out.organization_family = organizationNorm.family;
  out.organization_type = organizationNorm.type;
  out.organization_canonical = organizationNorm.canonical;
  out.group_raw = normalizeSpacing(out.group);
  out.organization_group = resolveOrganizationGroup(out);
  out.group = out.organization_group || out.group_raw;

  out.name_raw = normalizeSpacing(out.name);
  out.name_canonical = normalizePersonName(out.name_raw);
  if (out.name_canonical) out.name = out.name_canonical;
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

  // If hierarchy modal is open, refresh its selector state against the reloaded rows.
  if (els.hierarchyModal && els.hierarchyModal.getAttribute('aria-hidden') === 'false'){
    refreshHierarchyControls();
  }
}

function initTimeline(){
  const container = els.timeline;

  items = new vis.DataSet([]);
  const groups = new vis.DataSet([]);

  const options = {
    stack: false,
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
    renderTimelineDetail(it || null, resolveTimelineClickYear(it, props.time instanceof Date ? props.time : null));
  });

  timeline.on("click", (props) => {
    if (!props.item) return;
    const it = items.get(props.item);
    if (!it) return;
    const clickedDate = props.time instanceof Date ? props.time : new Date(props.time);
    const targetYear = resolveTimelineClickYear(it, clickedDate);
    if (!Number.isFinite(targetYear)) return;
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
  els.groupSelect = $("groupSelect");
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
  els.hierarchyYearSelect = $("hierarchyYearSelect");
  els.hierarchyConferenceSelect = $("hierarchyConferenceSelect");
  els.hierarchyOrganizationSelect = $("hierarchyOrganizationSelect");
  els.hierarchyChart = $("hierarchyChart");
  els.hierarchyDetail = $("hierarchyDetail");

  els.statLoaded = $("statLoaded");
  els.statShown = $("statShown");
  els.statYears = $("statYears");
  els.timeline = $("timeline");
  els.timelineGroupPulse = $("timelineGroupPulse");
  els.timelineMajorPlayers = $("timelineMajorPlayers");

  await loadNormalizationConfig();
  await loadAggregateTermsConfig();
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
  [els.hierarchyYearSelect, els.hierarchyConferenceSelect, els.hierarchyOrganizationSelect]
    .filter(Boolean)
    .forEach(select => select.addEventListener("change", () => refreshHierarchyControls()));

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

  [els.regionSelect, els.confSelect, els.positionSelect, els.genderSelect, els.orgSelect, els.groupSelect, els.entityTypeSelect]
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

function genderBucketLabel(gender){
  const normalized = normGender(gender);
  if (normalized === "female") return "Women";
  if (normalized === "male") return "Men";
  if (normalized) return titleCaseWords(normalized);
  return "Unspecified";
}

function genderTone(gender){
  const normalized = normGender(gender);
  if (normalized === "female") {
    return {
      bg: "rgba(255, 131, 170, 0.20)",
      border: "rgba(255, 171, 201, 0.48)",
      glow: "rgba(255, 108, 165, 0.26)",
      base: "#ff89b1",
      soft: "#ffd0e4",
    };
  }
  if (normalized === "male") {
    return {
      bg: "rgba(102, 164, 255, 0.18)",
      border: "rgba(150, 195, 255, 0.46)",
      glow: "rgba(102, 164, 255, 0.22)",
      base: "#77b3ff",
      soft: "#d3e7ff",
    };
  }
  return {
    bg: "rgba(170, 183, 214, 0.16)",
    border: "rgba(196, 205, 228, 0.38)",
    glow: "rgba(170, 183, 214, 0.18)",
    base: "#b2bdd6",
    soft: "#e0e7f8",
  };
}

function personId(r){
  const n = safe(r.name_canonical) || safe(r.name);
  if (n) return n.toLowerCase();
  const combo = [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim();
  return combo ? combo.toLowerCase() : "";
}

function personLabel(r){
  return safe(r.name_canonical) || safe(r.name) || [safe(r.prefix), safe(r.last_name), safe(r.suffix)].filter(Boolean).join(" ").trim() || "(unnamed)";
}

function summaryOrgLabel(r){
  return safe(r.organization_canonical) || safe(r.organization) || safe(r.institution_name);
}

function groupLabelForRow(r){
  const value = safe(r.organization_group) || safe(r.group);
  if (!value) return "";
  if (value.toLowerCase() === "remove") return "";
  return value;
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
    groupLabelForRow(r),
    safe(r.organization_type),
    inferConferenceEntityType(r),
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
  const confs = uniqSorted(rows.map(r => safe(r.conference_rollup) || safe(r.conference_canonical) || safe(r.conference)).filter(Boolean));
  const conferenceTypes = uniqSorted(rows.map(inferConferenceEntityType).filter(Boolean));
  const orgs = uniqSorted(rows.map(r => summaryOrgLabel(r)).filter(Boolean));
  const organizationTypes = uniqSorted(rows.map(r => safe(r.organization_type)).filter(Boolean));
  const groups = orderedOrganizationGroups(rows);
  const roles = uniqSorted(rows.map(r => safe(r.position)).filter(Boolean));
  const genders = uniqSorted(rows.map(r => normGender(r.gender)).filter(Boolean));
  const bounds = getSummaryYearBounds(rows);

  setSelectOptions($("sumRegion"), regions, { includeAll: true, allLabel: "All regions" });
  setSelectOptions($("sumConference"), confs, { includeAll: true, allLabel: "All conferences" });
  setSelectOptions($("sumConferenceType"), conferenceTypes, { includeAll: true, allLabel: "All conference types" });
  setSelectOptions($("sumOrganization"), orgs, { includeAll: true, allLabel: "All organizations" });
  setSelectOptions($("sumOrganizationType"), organizationTypes, { includeAll: true, allLabel: "All organization types" });
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
    conferenceType: safe($("sumConferenceType")?.value),
    organization: safe($("sumOrganization")?.value),
    organizationType: safe($("sumOrganizationType")?.value),
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
    if (state.conference && (safe(r.conference_rollup) || safe(r.conference_canonical) || safe(r.conference)) !== state.conference) return false;
    if (state.conferenceType && inferConferenceEntityType(r) !== state.conferenceType) return false;
    if (state.organization && summaryOrgLabel(r) !== state.organization) return false;
    if (state.organizationType && safe(r.organization_type) !== state.organizationType) return false;
    if (state.group && groupLabelForRow(r) !== state.group) return false;
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

const GROUP_TONE_PRESETS = {
  "Conference": { base: "#72b1ff", soft: "#a0f0ff" },
  "Mission": { base: "#4ed3bf", soft: "#9ef4ca" },
  "Tract Society": { base: "#ffb35f", soft: "#ffd590" },
  "Tract and Missionary": { base: "#ff8a66", soft: "#ffc28f" },
  "Sabbath-School": { base: "#a48cff", soft: "#d7b7ff" },
  "Health and Temperance": { base: "#77d97a", soft: "#bbf0a1" },
  "Publishing": { base: "#ff7dac", soft: "#ffbbd6" },
  "School": { base: "#6fd2ff", soft: "#b8f2ff" },
  "City Mission": { base: "#ff9b5e", soft: "#ffd0a8" },
  "Board": { base: "#8fd2d0", soft: "#c6f4ec" },
  "Committee": { base: "#a0aec6", soft: "#d9e2ff" },
  "Workers' Directory": { base: "#c68bff", soft: "#e1c3ff" },
  "Young People's Dept": { base: "#ff8f9e", soft: "#ffd0d6" },
  "Religious Liberty": { base: "#e8be68", soft: "#ffe3a6" },
  "Canvassing Agents": { base: "#69c1a7", soft: "#a8edd1" },
  "*": { base: "#8c96b2", soft: "#c7d0e8" },
};

function groupDisplayLabel(group){
  return safe(group) === "*" ? "Uncertain *" : (safe(group) || "Ungrouped");
}

function groupSortIndex(group){
  const order = aggregateTermsConfig?.organizationGroupOrder || [];
  const idx = order.indexOf(safe(group));
  return idx === -1 ? Number.MAX_SAFE_INTEGER : idx;
}

function groupTone(group){
  const label = safe(group) || "*";
  if (GROUP_TONE_PRESETS[label]) return GROUP_TONE_PRESETS[label];
  let hash = 0;
  for (const ch of label){
    hash = ((hash << 5) - hash) + ch.charCodeAt(0);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  return {
    base: `hsl(${hue} 76% 65%)`,
    soft: `hsl(${(hue + 32) % 360} 88% 82%)`,
  };
}

function compareGroups(a, b){
  return groupSortIndex(a) - groupSortIndex(b) || groupDisplayLabel(a).localeCompare(groupDisplayLabel(b));
}

function orderedOrganizationGroups(rows){
  return Array.from(new Set(rows.map(groupLabelForRow).filter(Boolean))).sort(compareGroups);
}

function buildOrganizationGroupMetrics(rows){
  const byGroup = new Map();
  for (const row of rows){
    const group = groupLabelForRow(row);
    if (!group) continue;
    if (!byGroup.has(group)){
      byGroup.set(group, {
        group,
        rowCount: 0,
        years: new Set(),
        people: new Map(),
        organizations: new Map(),
      });
    }
    const bucket = byGroup.get(group);
    bucket.rowCount += 1;
    const year = Number(row.yearbook_year);
    if (Number.isFinite(year)) bucket.years.add(year);

    const org = summaryOrgLabel(row);
    if (org) bucket.organizations.set(org, (bucket.organizations.get(org) || 0) + 1);

    if (!isLikelyNamedIndividual(row)) continue;
    const pid = personId(row);
    if (!pid) continue;
    if (!bucket.people.has(pid)){
      bucket.people.set(pid, {
        label: personLabel(row),
        female: false,
        roleCount: 0,
      });
    }
    const person = bucket.people.get(pid);
    person.roleCount += 1;
    if (normGender(row.gender) === "female") person.female = true;
  }

  return Array.from(byGroup.values())
    .map(bucket => {
      const people = Array.from(bucket.people.values());
      const women = people.filter(person => person.female).length;
      const gt5 = people.filter(person => person.roleCount > 5).length;
      const years = Array.from(bucket.years).sort((a, b) => a - b);
      const topOrganizations = Array.from(bucket.organizations.entries())
        .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
        .slice(0, 3)
        .map(([label, count]) => `${label} (${count})`);

      return {
        group: bucket.group,
        rowCount: bucket.rowCount,
        namedIndividuals: people.length,
        women,
        womenPct: people.length ? (women / people.length) * 100 : NaN,
        gt5,
        gt5Pct: people.length ? (gt5 / people.length) * 100 : NaN,
        years,
        topOrganizations,
      };
    })
    .sort((a, b) =>
      (b.rowCount - a.rowCount) ||
      (b.namedIndividuals - a.namedIndividuals) ||
      compareGroups(a.group, b.group)
    );
}

function buildOrganizationGroupYearMix(rows, topN = 6){
  const yearBuckets = new Map();
  const totals = new Map();

  for (const row of rows){
    const year = Number(row.yearbook_year);
    if (!Number.isFinite(year)) continue;
    const group = groupLabelForRow(row);
    if (!group) continue;
    if (!yearBuckets.has(year)) yearBuckets.set(year, new Map());
    const bucket = yearBuckets.get(year);
    bucket.set(group, (bucket.get(group) || 0) + 1);
    totals.set(group, (totals.get(group) || 0) + 1);
  }

  const topGroups = Array.from(totals.entries())
    .sort((a, b) => (b[1] - a[1]) || compareGroups(a[0], b[0]))
    .slice(0, topN)
    .map(([group]) => group);

  const years = Array.from(yearBuckets.keys()).sort((a, b) => a - b);
  const series = years.map(year => {
    const source = yearBuckets.get(year) || new Map();
    const entries = [];
    let otherCount = 0;
    let total = 0;
    for (const [group, count] of source.entries()){
      total += count;
      if (topGroups.includes(group)) entries.push({ group, count });
      else otherCount += count;
    }
    entries.sort((a, b) => compareGroups(a.group, b.group));
    if (otherCount) entries.push({ group: "Other", count: otherCount });
    const dominant = entries.reduce((best, entry) => !best || entry.count > best.count ? entry : best, null);
    return {
      year,
      total,
      dominant: dominant?.group || "",
      entries: entries.map(entry => ({
        ...entry,
        pct: total ? (entry.count / total) * 100 : 0,
      })),
    };
  });

  const legend = [...topGroups];
  if (series.some(item => item.entries.some(entry => entry.group === "Other"))) legend.push("Other");
  return { series, legend };
}

function renderOrganizationGroupMix(elId, rows, opts = {}){
  const host = $(elId);
  if (!host) return;
  const { series, legend } = buildOrganizationGroupYearMix(rows, opts.topN || 6);
  if (!series.length){
    host.innerHTML = `<div class="summaryEmptyState">${escapeHtml(opts.emptyMessage || "No organization-group activity is available for this slice.")}</div>`;
    return;
  }

  const legendHtml = legend.map(group => {
    const tone = groupTone(group);
    return `
      <div class="groupMixLegend__item">
        <span class="groupMixLegend__swatch" style="--group-base:${tone.base}; --group-soft:${tone.soft};"></span>
        <span>${escapeHtml(groupDisplayLabel(group))}</span>
      </div>
    `;
  }).join("");

  const barsHtml = series.map(item => {
    const segments = item.entries.map(entry => {
      const tone = groupTone(entry.group);
      const pct = Math.max(entry.pct, entry.count > 0 ? 2 : 0);
      return `
        <div
          class="groupMixYear__segment"
          style="height:${pct.toFixed(2)}%; --group-base:${tone.base}; --group-soft:${tone.soft};"
          title="${escapeHtml(`${item.year} • ${groupDisplayLabel(entry.group)} • ${entry.count} rows (${formatPercent(entry.pct)})`)}"
        ></div>
      `;
    }).join("");

    return `
      <article class="groupMixYear">
        <div class="groupMixYear__bar">${segments}</div>
        <div class="groupMixYear__year">${item.year}</div>
        <div class="groupMixYear__meta">${pluralize(item.total, "row")}</div>
        <div class="groupMixYear__dominant">${escapeHtml(groupDisplayLabel(item.dominant || "*"))}</div>
      </article>
    `;
  }).join("");

  host.className = `${host.className.split(" ").filter(Boolean).filter(cls => !cls.startsWith("groupMixChart--")).join(" ")} ${opts.compact ? "groupMixChart--compact" : "groupMixChart--full"}`.trim();
  host.innerHTML = `
    <div class="groupMixLegend">${legendHtml}</div>
    <div class="groupMixChart__scroller">
      <div class="groupMixChart__bars">${barsHtml}</div>
    </div>
  `;
}

function renderOrganizationGroupCards(elId, rows){
  const host = $(elId);
  if (!host) return;
  const metrics = buildOrganizationGroupMetrics(rows).slice(0, 12);
  if (!metrics.length){
    host.innerHTML = `<div class="summaryEmptyState">No organization-group summary is available for this slice.</div>`;
    return;
  }

  host.innerHTML = metrics.map(metric => {
    const tone = groupTone(metric.group);
    const yearsLabel = metric.years.length
      ? (metric.years[0] === metric.years[metric.years.length - 1]
        ? String(metric.years[0])
        : `${metric.years[0]}–${metric.years[metric.years.length - 1]}`)
      : "–";
    return `
      <article class="groupStatCard" style="--group-base:${tone.base}; --group-soft:${tone.soft};">
        <div class="groupStatCard__header">
          <div>
            <div class="groupStatCard__title">${escapeHtml(groupDisplayLabel(metric.group))}</div>
            <div class="groupStatCard__years">Active years: ${escapeHtml(yearsLabel)}</div>
          </div>
          <div class="groupStatCard__rows">${metric.rowCount.toLocaleString()}</div>
        </div>
        <div class="groupStatCard__stats">
          <div><span>Named people</span><strong>${metric.namedIndividuals.toLocaleString()}</strong></div>
          <div><span>% women</span><strong>${escapeHtml(formatPercent(metric.womenPct))}</strong></div>
          <div><span>% > 5 roles</span><strong>${escapeHtml(formatPercent(metric.gt5Pct))}</strong></div>
        </div>
        <div class="groupStatCard__footer">${escapeHtml(metric.topOrganizations.join(" • ") || "No dominant organizations in this slice.")}</div>
      </article>
    `;
  }).join("");
}

function buildTopRolesByGender(rows, topN = 5){
  const buckets = new Map();
  for (const row of rows){
    if (!isLikelyNamedIndividual(row)) continue;
    const role = safe(row.position);
    if (!role) continue;
    const gender = normGender(row.gender) || "";
    if (!buckets.has(gender)){
      buckets.set(gender, {
        gender,
        totalRows: 0,
        roles: new Map(),
      });
    }
    const bucket = buckets.get(gender);
    bucket.totalRows += 1;
    bucket.roles.set(role, (bucket.roles.get(role) || 0) + 1);
  }

  const order = ["female", "male", ""];
  return Array.from(buckets.values())
    .sort((a, b) => {
      const idxA = order.indexOf(a.gender);
      const idxB = order.indexOf(b.gender);
      return (idxA === -1 ? 99 : idxA) - (idxB === -1 ? 99 : idxB) ||
        genderBucketLabel(a.gender).localeCompare(genderBucketLabel(b.gender));
    })
    .map(bucket => ({
      gender: bucket.gender,
      totalRows: bucket.totalRows,
      roles: Array.from(bucket.roles.entries())
        .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
        .slice(0, topN)
        .map(([role, count]) => ({
          role,
          count,
          pct: bucket.totalRows ? (count / bucket.totalRows) * 100 : 0,
        })),
    }));
}

function renderTopRolesByGender(rows){
  const host = $("topRolesByGender");
  if (!host) return;
  const buckets = buildTopRolesByGender(rows, 5);
  if (!buckets.length){
    host.innerHTML = `<div class="summaryEmptyState">No named rows with role titles match the current summary filters.</div>`;
    return;
  }

  host.innerHTML = buckets.map(bucket => {
    const tone = genderTone(bucket.gender);
    const items = bucket.roles.length
      ? bucket.roles.map((entry, index) => `
          <li class="genderRoleItem">
            <span class="genderRoleItem__rank">${index + 1}</span>
            <div class="genderRoleItem__main">
              <div class="genderRoleItem__label">${escapeHtml(entry.role)}</div>
              <div class="genderRoleItem__meta">${entry.count.toLocaleString()} row${entry.count === 1 ? "" : "s"} • ${escapeHtml(formatPercent(entry.pct))} of this gender slice</div>
            </div>
            <div class="genderRoleItem__barWrap">
              <span class="genderRoleItem__bar" style="width:${Math.max(entry.pct, 8).toFixed(2)}%; --gender-base:${tone.base}; --gender-soft:${tone.soft};"></span>
            </div>
          </li>
        `).join("")
      : `<li class="genderRoleItem genderRoleItem--empty">No roles in this gender bucket for the current slice.</li>`;

    return `
      <article class="genderRoleCard" style="--gender-base:${tone.base}; --gender-soft:${tone.soft};">
        <div class="genderRoleCard__header">
          <div>
            <div class="genderRoleCard__title">${escapeHtml(genderBucketLabel(bucket.gender))}</div>
            <div class="genderRoleCard__meta">${bucket.totalRows.toLocaleString()} matching role row${bucket.totalRows === 1 ? "" : "s"}</div>
          </div>
        </div>
        <ol class="genderRoleList">${items}</ol>
      </article>
    `;
  }).join("");
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

function splitYearSeriesIntoSegments(series){
  const segments = [];
  let current = [];
  for (const item of series){
    if (!current.length){
      current.push(item);
      continue;
    }
    const prev = current[current.length - 1];
    if (item.year === prev.year + 1){
      current.push(item);
      continue;
    }
    segments.push(current);
    current = [item];
  }
  if (current.length) segments.push(current);
  return segments;
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
  const segments = splitYearSeriesIntoSegments(coords);
  const linePaths = segments
    .filter(segment => segment.length > 1)
    .map(segment => segment.map((d, i) => `${i === 0 ? "M" : "L"} ${d.cx.toFixed(2)} ${d.cy.toFixed(2)}`).join(" "));
  const areaPaths = segments
    .filter(segment => segment.length > 1)
    .map(segment => [
      `M ${segment[0].cx.toFixed(2)} ${baselineY.toFixed(2)}`,
      `L ${segment[0].cx.toFixed(2)} ${segment[0].cy.toFixed(2)}`,
      ...segment.slice(1).map(d => `L ${d.cx.toFixed(2)} ${d.cy.toFixed(2)}`),
      `L ${segment[segment.length - 1].cx.toFixed(2)} ${baselineY.toFixed(2)}`,
      "Z"
    ].join(" "));

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
        ${areaPaths.map(path => `<path class="${areaCls}" d="${path}"></path>`).join("")}
        ${linePaths.map(path => `<path class="${lineCls}" d="${path}"></path>`).join("")}
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

  renderOrganizationGroupMix("groupMixChart", filteredRows, {
    topN: 7,
    emptyMessage: "No organization-group mix is available for this summary slice.",
  });
  renderOrganizationGroupCards("groupStatsGrid", filteredRows);
  renderTopRolesByGender(filteredRows);
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
  const ids = ["sumSearch", "sumRegion", "sumConference", "sumConferenceType", "sumOrganization", "sumOrganizationType", "sumGroup", "sumRole", "sumRoleDetail", "sumGender", "sumYearMin", "sumYearMax"];
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
      if ($("sumConferenceType")) $("sumConferenceType").value = "";
      if ($("sumOrganization")) $("sumOrganization").value = "";
      if ($("sumOrganizationType")) $("sumOrganizationType").value = "";
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
