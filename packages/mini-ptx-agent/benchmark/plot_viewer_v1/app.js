const RELEASE_REFERENCE_PATHS = {
  model: "./data/model_release_dates.csv",
  ptx: "./data/ptx_release_dates.csv",
  fa: "./data/fa_release_dates.csv",
};

const metrics = [
  {
    id: "correctness_rate_by_release_date",
    label: "Correctness rate",
    csvPath: "./data/correctness_rate_by_release_date.csv",
    chartType: "turnSeries",
    valueKey: "correctness_rate",
    yLabel: "Correctness rate",
    percentY: true,
  },
  {
    id: "best_speedup_by_release_date",
    label: "Best speedup",
    csvPath: "./data/best_speedup_by_release_date.csv",
    chartType: "turnSeries",
    valueKey: "best_speedup",
    yLabel: "Best speedup",
    percentY: false,
    yMax: 1.2,
  },
  {
    id: "arch_instruction_correctness_by_release_date",
    label: "Correctness with verified SASS",
    csvPath: "./data/arch_instruction_correctness_by_release_date.csv",
    chartType: "turnSeries",
    valueKey: "correct_with_arch_rate",
    yLabel: "Correct + SASS rate",
    percentY: true,
    requiredEvidence: "dynamic_sass",
  },
  {
    id: "reasoning_tokens_vs_performance",
    label: "Perf-reasoning",
    csvPath: "./data/reasoning_tokens_vs_performance.csv",
    chartType: "scatter",
    xKey: "reasoning_tokens",
    yKey: "performance_score",
    xLabel: "Reasoning tokens",
    yLabel: "Speedup",
  },
];

const state = {
  groups: [],
  visibleGroups: [],
  metricRows: new Map(),
  releaseRefs: {
    model: [],
    ptx: [],
    fa: [],
  },
  selected: new Set(),
  modelsByGroup: new Map(),
  arch: "all",
  search: "",
  metricIds: new Set(metrics.map((metric) => metric.id)),
  columns: 2,
  size: "standard",
  assetCacheToken: "",
};

const els = {
  datasetSummary: document.querySelector("#datasetSummary"),
  refreshButton: document.querySelector("#refreshButton"),
  searchInput: document.querySelector("#searchInput"),
  groupList: document.querySelector("#groupList"),
  selectVisibleButton: document.querySelector("#selectVisibleButton"),
  clearButton: document.querySelector("#clearButton"),
  selectedSummary: document.querySelector("#selectedSummary"),
  plotGrid: document.querySelector("#plotGrid"),
  columnsSelect: document.querySelector("#columnsSelect"),
  sizeSelect: document.querySelector("#sizeSelect"),
  copyLinkButton: document.querySelector("#copyLinkButton"),
  presentationButton: document.querySelector("#presentationButton"),
  plotDialog: document.querySelector("#plotDialog"),
  dialogImage: document.querySelector("#dialogImage"),
  dialogChart: document.querySelector("#dialogChart"),
  dialogTitle: document.querySelector("#dialogTitle"),
  closeDialogButton: document.querySelector("#closeDialogButton"),
};

const chartPalette = [
  "#2457a6",
  "#13795b",
  "#a05a00",
  "#7a5195",
  "#d1495b",
  "#4d908e",
  "#8f6a00",
  "#c75146",
  "#2d5d7b",
  "#6a7f32",
  "#9b4d83",
  "#5a5a5a",
  "#006d77",
  "#b5651d",
  "#4458a8",
  "#8a3ffc",
  "#0b7a75",
  "#9d4edd",
];

const TURN_MARKERS = new Map([
  [1, "circle"],
  [4, "square"],
  [8, "triangle"],
]);

const DEFAULT_DATE_MIN = Date.UTC(2022, 6, 1);
const DEFAULT_DATE_MAX = Date.UTC(2026, 6, 1);
let chartIdCounter = 0;

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];

    if (ch === '"' && inQuotes && next === '"') {
      field += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const [headers, ...dataRows] = rows;
  return dataRows.map((values) => Object.fromEntries(headers.map((header, idx) => [header, values[idx] || ""])));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function groupId(group) {
  return `${group.workloadName}__${group.arch}`;
}

function rowGroupId(row) {
  return `${row.workload_name}__${row.arch}`;
}

function numberValue(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseDate(value) {
  if (!value) return null;
  const text = String(value).trim();
  const normal = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
  if (normal) {
    const time = Date.UTC(Number(normal[1]), Number(normal[2]) - 1, Number(normal[3]));
    return Number.isFinite(time) ? time : null;
  }
  const compact = /^(\d{4})-(\d{2})(\d{2})$/.exec(text);
  if (compact) {
    const time = Date.UTC(Number(compact[1]), Number(compact[2]) - 1, Number(compact[3]));
    return Number.isFinite(time) ? time : null;
  }
  const time = new Date(`${text}T00:00:00Z`).getTime();
  return Number.isFinite(time) ? time : null;
}

function formatDate(value) {
  return new Date(value).toISOString().slice(0, 10);
}

function formatNumber(value, percent = false) {
  if (!Number.isFinite(value)) return "";
  if (percent) return `${Math.round(value * 100)}%`;
  if (Math.abs(value) >= 1000) return value.toExponential(1);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function modelColor(model, modelOrder = []) {
  const index = modelOrder.indexOf(model);
  if (index >= 0) return chartPalette[index % chartPalette.length];
  return chartPalette[0];
}

function modelColorMap(rows, modelOrder = []) {
  const presentModels = new Set(rows.map((row) => row.model || "unknown"));
  const orderedModels = [
    ...modelOrder.filter((model) => presentModels.has(model)),
    ...[...presentModels]
      .filter((model) => !modelOrder.includes(model))
      .sort((a, b) => a.localeCompare(b)),
  ];
  return new Map(orderedModels.map((model) => [model, modelColor(model, modelOrder)]));
}

function selectedModelsForGroup(group) {
  const selectedModels = state.modelsByGroup.get(group.id);
  if (!selectedModels || !selectedModels.size) return null;
  const validModels = group.models.filter((model) => selectedModels.has(model));
  return validModels.length ? new Set(validModels) : null;
}

function selectedModelLabel(group, selectedModels) {
  const models = selectedModels?.size
    ? group.models.filter((model) => selectedModels.has(model))
    : group.models;
  if (!models.length) return "all models";
  return models.join(", ");
}

function captionModelListHtml(group, selectedModels) {
  const models = selectedModels?.size
    ? group.models.filter((model) => selectedModels.has(model))
    : group.models;
  if (!models.length) return "all models";
  return models
    .map(
      (model) =>
        `<span class="caption-model" style="color: ${modelColor(model, group.models)}">${escapeHtml(model)}</span>`
    )
    .join(", ");
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function reasoningTokenDensity(xs) {
  if (xs.length < 2) return null;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  if (xMin === xMax) return null;

  const n = xs.length;
  const xMean = mean(xs);
  const variance = xs.reduce((total, value) => total + (value - xMean) ** 2, 0) / (n - 1);
  const std = Math.sqrt(variance);
  const bandwidth = std > 0 ? 1.06 * std * n ** (-1 / 5) : 0;
  if (bandwidth <= 0) return null;

  const pad = 0.04 * (xMax - xMin);
  const grid = Array.from({ length: 200 }, (_unused, index) => xMin - pad + ((xMax - xMin + 2 * pad) * index) / 199);
  const scale = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));
  const density = grid.map((xi) =>
    scale * xs.reduce((total, value) => total + Math.exp(-0.5 * ((value - xi) / bandwidth) ** 2), 0)
  );
  const maxDensity = Math.max(...density);
  if (maxDensity <= 0) return null;
  return {
    grid,
    density: density.map((value) => value / maxDensity),
  };
}

function chartPoint(x, y, color, marker, label) {
  const safeLabel = escapeHtml(label);
  if (marker === "square") {
    return `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" rx="1.5" fill="${color}"><title>${safeLabel}</title></rect>`;
  }
  if (marker === "triangle") {
    return `<path d="M ${x} ${y - 5} L ${x + 5} ${y + 4} L ${x - 5} ${y + 4} Z" fill="${color}"><title>${safeLabel}</title></path>`;
  }
  return `<circle cx="${x}" cy="${y}" r="4.5" fill="${color}"><title>${safeLabel}</title></circle>`;
}

function renderEmptyChart(message) {
  return `<div class="client-chart empty-chart">${escapeHtml(message)}</div>`;
}

function loadStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const selected = params.get("groups");
  const metricIds = params.get("metrics");
  const columns = Number(params.get("columns"));
  const size = params.get("size");
  const modelFilters = params.get("models");

  if (selected) state.selected = new Set(selected.split(",").filter(Boolean));
  if (metricIds) {
    const valid = new Set(metrics.map((metric) => metric.id));
    state.metricIds = new Set(metricIds.split(",").filter((id) => valid.has(id)));
  }
  if ([1, 2, 3].includes(columns)) state.columns = columns;
  if (["compact", "standard", "large"].includes(size)) state.size = size;
  if (modelFilters) {
    for (const entry of modelFilters.split(",")) {
      const [encodedGroupId, encodedModels] = entry.split("=");
      if (!encodedGroupId || !encodedModels) continue;
      const selectedModels = new Set(
        encodedModels
          .split("|")
          .filter(Boolean)
          .map((encodedModel) => decodeURIComponent(encodedModel))
      );
      if (selectedModels.size) state.modelsByGroup.set(decodeURIComponent(encodedGroupId), selectedModels);
    }
  }
}

function writeStateToUrl() {
  const params = new URLSearchParams();
  if (state.selected.size) params.set("groups", [...state.selected].join(","));
  params.set("metrics", [...state.metricIds].join(","));
  params.set("columns", String(state.columns));
  params.set("size", state.size);
  const modelFilters = [...state.modelsByGroup.entries()]
    .filter(([, models]) => models && models.size)
    .map(
      ([id, models]) =>
        `${encodeURIComponent(id)}=${[...models].map((model) => encodeURIComponent(model)).join("|")}`
    );
  if (modelFilters.length) params.set("models", modelFilters.join(","));
  const nextUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, "", nextUrl);
}

function buildGroups(rows) {
  const byGroup = new Map();
  for (const row of rows) {
    const workloadName = row.workload_name;
    const arch = row.arch;
    const id = `${workloadName}__${arch}`;
    if (!byGroup.has(id)) {
      byGroup.set(id, {
        id,
        workloadName,
        arch,
        definition: row.definition,
        workload: row.workload,
        models: new Set(),
        turns: new Set(),
        rows: 0,
      });
    }
    const group = byGroup.get(id);
    group.models.add(row.model);
    group.turns.add(row.turn_limit);
    group.rows += 1;
  }

  return [...byGroup.values()]
    .map((group) => ({
      ...group,
      models: [...group.models].sort(),
      turns: [...group.turns].map(Number).sort((a, b) => a - b),
    }))
    .sort((a, b) => a.workloadName.localeCompare(b.workloadName) || a.arch.localeCompare(b.arch));
}

function parseReleaseRefs(kind, rows) {
  if (kind === "model") {
    return rows
      .map((row) => ({
        kind,
        model: row.model,
        date: parseDate(row.date),
      }))
      .filter((row) => row.model && row.date !== null);
  }
  if (kind === "ptx") {
    return rows
      .map((row) => ({
        kind,
        arch: row.arch?.trim().toLowerCase(),
        date: parseDate(row.date),
        label: `${row.arch} architecture`,
        color: "#8a3ffc",
        dash: "2 4",
      }))
      .filter((row) => row.arch && row.date !== null);
  }
  return rows
    .map((row) => ({
      kind,
      arch: row.arch?.trim().toLowerCase(),
      date: parseDate(row.date || row.dates),
      label: `${row.version} FA`,
      color: "#00876c",
      dash: "7 5",
    }))
    .filter((row) => row.arch && row.date !== null);
}

async function loadData() {
  els.datasetSummary.textContent = "Loading plots";
  els.plotGrid.innerHTML = "";
  try {
    state.assetCacheToken = String(Date.now());
    const metricPayloads = await Promise.all(
      metrics.map(async (metric) => {
        const response = await fetch(`${metric.csvPath}?t=${state.assetCacheToken}`);
        if (!response.ok) throw new Error(`Could not read ${metric.csvPath}`);
        return [metric.id, parseCsv(await response.text())];
      })
    );
    const refPayloads = await Promise.all(
      Object.entries(RELEASE_REFERENCE_PATHS).map(async ([kind, path]) => {
        const response = await fetch(`${path}?t=${state.assetCacheToken}`);
        if (!response.ok) return [kind, []];
        return [kind, parseReleaseRefs(kind, parseCsv(await response.text()))];
      })
    );
    state.metricRows = new Map(metricPayloads);
    state.releaseRefs = Object.fromEntries(refPayloads);
    const summaryRows = state.metricRows.get("correctness_rate_by_release_date") || [];
    state.groups = buildGroups(summaryRows);

    if (!state.selected.size) {
      for (const group of state.groups.slice(0, 2)) state.selected.add(group.id);
    }

    els.datasetSummary.textContent = `${state.groups.length} groups from ${summaryRows.length} rows`;
    render();
  } catch (error) {
    els.datasetSummary.textContent = "Could not load plots";
    els.plotGrid.innerHTML = `<div class="error-state">${escapeHtml(error.message)}</div>`;
  }
}

function filteredGroups() {
  const query = state.search.trim().toLowerCase();
  return state.groups.filter((group) => {
    const archMatch = state.arch === "all" || group.arch === state.arch;
    if (!archMatch) return false;
    if (!query) return true;
    const haystack = [
      group.workloadName,
      group.arch,
      group.definition,
      group.workload,
      ...group.models,
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function renderGroupList() {
  state.visibleGroups = filteredGroups();
  els.groupList.innerHTML = "";
  for (const group of state.visibleGroups) {
    const item = document.createElement("label");
    item.className = "group-item";
    item.innerHTML = `
      <input type="checkbox" value="${escapeHtml(group.id)}" ${state.selected.has(group.id) ? "checked" : ""} />
      <span>
        <span class="group-name">${escapeHtml(group.workloadName)}</span>
        <span class="group-meta">
          <span class="chip ${escapeHtml(group.arch)}">${escapeHtml(group.arch)}</span>
          <span class="chip">${group.models.length} models</span>
          <span class="chip">T=${group.turns.join("/")}</span>
        </span>
      </span>
    `;
    item.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.selected.add(group.id);
      else state.selected.delete(group.id);
      renderPlots();
      writeStateToUrl();
    });
    els.groupList.append(item);
  }
}

function selectedGroups() {
  const groupsById = new Map(state.groups.map((group) => [group.id, group]));
  return [...state.selected].map((id) => groupsById.get(id)).filter(Boolean);
}

function metricRowsForGroup(metric, group, selectedModels) {
  const rows = state.metricRows.get(metric.id) || [];
  return rows.filter(
    (row) =>
      rowGroupId(row) === group.id &&
      (!selectedModels || selectedModels.has(row.model)) &&
      (!metric.requiredEvidence || row.tag_evidence === metric.requiredEvidence)
  );
}

function workloadReferenceLines(group) {
  const arch = group.arch.trim().toLowerCase();
  const lines = [];
  const ptx = (state.releaseRefs.ptx || []).find((row) => row.arch === arch);
  if (ptx) lines.push(ptx);
  if (group.definition.toLowerCase().includes("mha")) {
    lines.push(...(state.releaseRefs.fa || []).filter((row) => row.arch === arch));
  }
  return lines;
}

function renderReferenceLines(referenceLines, x, top, bottom) {
  return referenceLines
    .filter((line) => line.date >= DEFAULT_DATE_MIN && line.date <= DEFAULT_DATE_MAX)
    .map(
      (line) => `
        <line x1="${x(line.date)}" x2="${x(line.date)}" y1="${top}" y2="${bottom}"
          stroke="${line.color}" stroke-width="1.8" stroke-dasharray="${line.dash}" opacity="0.82">
          <title>${escapeHtml(`${line.label}: ${formatDate(line.date)}`)}</title>
        </line>
      `
    )
    .join("");
}

function renderReferenceCaption(referenceLines, x, y) {
  const captionLines = referenceLines.filter((line) => line.kind === "ptx" || line.kind === "fa");
  if (!captionLines.length) return "";
  let cursorX = x;
  return captionLines
    .map((line) => {
      const label = `${line.label} ${formatDate(line.date)}`;
      const textWidth = Math.min(190, 8 * label.length + 18);
      const item = `
        <g class="reference-caption">
          <line x1="${cursorX}" x2="${cursorX + 24}" y1="${y - 4}" y2="${y - 4}"
            stroke="${line.color}" stroke-width="2" stroke-dasharray="${line.dash}" />
          <text x="${cursorX + 30}" y="${y}">${escapeHtml(label)}</text>
        </g>
      `;
      cursorX += textWidth;
      return item;
    })
    .join("");
}

function renderTurnSeriesChart(metric, rows, group, selectedModels, large = false) {
  const modelLabel = selectedModelLabel(group, selectedModels);
  const points = rows
    .map((row) => ({
      model: row.model || "unknown",
      date: parseDate(row.date),
      turn: Number(row.turn_limit),
      value: numberValue(row[metric.valueKey]),
      n: row.n_trajectories,
    }))
    .filter((point) => point.date !== null && Number.isFinite(point.turn) && point.value !== null);
  if (!points.length) return renderEmptyChart(`No ${metric.label.toLowerCase()} rows for ${modelLabel}`);

  const width = large ? 980 : 760;
  const height = large ? 470 : 330;
  const margin = { top: 46, right: 32, bottom: 50, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (value) => margin.left + ((value - DEFAULT_DATE_MIN) / (DEFAULT_DATE_MAX - DEFAULT_DATE_MIN)) * plotWidth;
  const observedMax = Math.max(...points.map((point) => point.value));
  const yMin = 0;
  const yMax = metric.yMax ?? (metric.percentY ? 1 : Math.max(1, observedMax * 1.12));
  const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin || 1)) * plotHeight;
  const turns = [...new Set(points.map((point) => point.turn))].sort((a, b) => a - b);
  const colorByModel = modelColorMap(rows, group.models);
  const yTicks = metric.percentY ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.25, 0.5, 0.75, 1].map((ratio) => ratio * yMax);
  const xTicks = [Date.UTC(2023, 0, 1), Date.UTC(2024, 0, 1), Date.UTC(2025, 0, 1), Date.UTC(2026, 0, 1)];
  const workloadRefLines = workloadReferenceLines(group);
  const referenceLines = workloadRefLines;

  const grid = [
    ...yTicks.map((tick) => `
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}" />
      <text x="${margin.left - 10}" y="${y(tick) + 4}" text-anchor="end">${formatNumber(tick, metric.percentY)}</text>
    `),
    ...xTicks.map((tick) => `
      <line x1="${x(tick)}" x2="${x(tick)}" y1="${margin.top}" y2="${height - margin.bottom}" />
      <text x="${x(tick)}" y="${height - 18}" text-anchor="middle">${new Date(tick).getUTCFullYear()}</text>
    `),
  ].join("");

  const series = [...colorByModel.keys()]
    .map((modelName) => {
      const color = colorByModel.get(modelName);
      const modelPoints = points.filter((point) => point.model === modelName);
      const lineGroups = new Map();
      for (const row of rows.filter((item) => item.model === modelName)) {
        const key = row.exp_dir || row.date;
        lineGroups.set(key, [...(lineGroups.get(key) || []), row]);
      }
      const lines = [...lineGroups.values()]
        .map((lineRows) => {
          const linePoints = lineRows
            .map((row) => ({
              date: parseDate(row.date),
              turn: Number(row.turn_limit),
              value: numberValue(row[metric.valueKey]),
            }))
            .filter((point) => point.date !== null && Number.isFinite(point.turn) && point.value !== null)
            .sort((a, b) => a.turn - b.turn);
          if (linePoints.length < 2) return "";
          return `<polyline points="${linePoints.map((point) => `${x(point.date)},${y(point.value)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" opacity="0.45" />`;
        })
        .join("");
      const markers = modelPoints
        .map((point) =>
          chartPoint(
            x(point.date),
            y(point.value),
            color,
            TURN_MARKERS.get(point.turn) || "circle",
            `${modelName}, T=${point.turn}, ${formatDate(point.date)}, ${formatNumber(point.value, metric.percentY)}, n=${point.n}`
          )
        )
        .join("");
      return `${lines}${markers}`;
    })
    .join("");

  const turnLegend = turns
    .slice(0, 3)
    .map((turn, index) => {
      const lx = margin.left + index * 58;
      const ly = height - 34;
      return `${chartPoint(lx, ly - 4, "#445063", TURN_MARKERS.get(turn) || "circle", `Turn ${turn}`)}
        <text x="${lx + 12}" y="${ly}" class="chart-legend">T=${turn}</text>`;
    })
    .join("");
  const labelTurn = Math.max(...turns);
  const countAnnotations = points
    .filter((point) => point.turn === labelTurn)
    .map((point) => {
      const color = colorByModel.get(point.model) || chartPalette[0];
      return `
        <text x="${x(point.date) + 5}" y="${y(point.value) - 4}" class="trajectory-count-label" fill="${color}">
          (${escapeHtml(point.n)})
        </text>
      `;
    })
    .join("");

  return `
    <svg class="client-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metric.label)} for ${escapeHtml(modelLabel)}">
      <rect width="${width}" height="${height}" fill="#fff" />
      <g class="chart-grid">${grid}</g>
      <g>${renderReferenceLines(referenceLines, x, margin.top, height - margin.bottom)}</g>
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" class="chart-axis" />
      <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}" class="chart-axis" />
      <text x="${margin.left}" y="18" class="chart-title">${escapeHtml(metric.label)}</text>
      <text x="${margin.left}" y="${height - 7}" class="chart-axis-label">Release date</text>
      <text transform="translate(16 ${margin.top + plotHeight / 2}) rotate(-90)" class="chart-axis-label">${escapeHtml(metric.yLabel)}</text>
      <g>${series}</g>
      <g>${countAnnotations}</g>
      <g>${turnLegend}</g>
      <g>${renderReferenceCaption(workloadRefLines, margin.left + 190, height - 34)}</g>
    </svg>
  `;
}

function renderScatterChart(metric, rows, group, selectedModels, large = false) {
  const modelLabel = selectedModelLabel(group, selectedModels);
  const points = rows
    .map((row) => ({
      model: row.model || "unknown",
      x: numberValue(row[metric.xKey]),
      y: numberValue(row[metric.yKey]),
      turn: row.turn,
      correctness: row.correctness,
      speedup: row.speedup,
    }))
    .filter((point) => point.x !== null && point.y !== null);
  if (!points.length) return renderEmptyChart(`No ${metric.label.toLowerCase()} rows for ${modelLabel}`);

  const width = large ? 980 : 760;
  const height = large ? 470 : 330;
  const margin = { top: 46, right: 72, bottom: 52, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMax = 102400;
  const yMax = 1.2;
  const x = (value) => margin.left + (value / xMax) * plotWidth;
  const y = (value) => margin.top + (1 - value / yMax) * plotHeight;
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => ratio * xMax);
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => ratio * yMax);
  const colorByModel = modelColorMap(rows, group.models);
  chartIdCounter += 1;
  const clipId = `scatterClip${chartIdCounter}`;

  const grid = [
    ...yTicks.map((tick) => `
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}" />
      <text x="${margin.left - 10}" y="${y(tick) + 4}" text-anchor="end">${formatNumber(tick)}</text>
    `),
    ...xTicks.map((tick) => `
      <line x1="${x(tick)}" x2="${x(tick)}" y1="${margin.top}" y2="${height - margin.bottom}" />
      <text x="${x(tick)}" y="${height - 18}" text-anchor="middle">${formatNumber(tick)}</text>
    `),
  ].join("");
  const dots = points
    .map((point) => {
      const color = colorByModel.get(point.model) || chartPalette[0];
      return `
        <circle cx="${x(point.x)}" cy="${y(point.y)}" r="3.3" fill="${color}" opacity="0.62">
          <title>${escapeHtml(`${point.model}, turn=${point.turn}, tokens=${point.x}, score=${formatNumber(point.y)}, ${point.correctness}, speedup=${point.speedup}`)}</title>
        </circle>
      `;
    })
    .join("");
  const densityY = (value) => margin.top + (1 - value) * plotHeight;
  const densityAxis = [0, 0.5, 1]
    .map(
      (tick) => `
        <line x1="${width - margin.right}" x2="${width - margin.right + 5}" y1="${densityY(tick)}" y2="${densityY(tick)}" class="chart-axis" />
        <text x="${width - margin.right + 9}" y="${densityY(tick) + 4}" class="chart-grid-label">${formatNumber(tick)}</text>
      `
    )
    .join("");
  const densityCurves = [...colorByModel.keys()]
    .map((modelName) => {
      const modelXs = points.filter((point) => point.model === modelName).map((point) => point.x);
      const density = reasoningTokenDensity(modelXs);
      if (density === null) return "";
      const color = colorByModel.get(modelName) || chartPalette[0];
      const densityPairs = density.grid
        .map((value, index) => [value, density.density[index]])
        .filter(([value]) => value >= 0 && value <= xMax);
      if (densityPairs.length < 2) return "";
      const linePoints = densityPairs.map(([value, densityValue]) => `${x(value)},${densityY(densityValue)}`).join(" ");
      const areaPoints = [
        `${x(densityPairs[0][0])},${densityY(0)}`,
        linePoints,
        `${x(densityPairs[densityPairs.length - 1][0])},${densityY(0)}`,
      ].join(" ");
      return `
        <polygon points="${areaPoints}" fill="${color}" opacity="0.12"></polygon>
        <polyline points="${linePoints}" fill="none" stroke="${color}" stroke-width="2" opacity="0.9">
          <title>${escapeHtml(`${modelName} reasoning-token density`)}</title>
        </polyline>
      `;
    })
    .join("");

  return `
    <svg class="client-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(metric.label)} for ${escapeHtml(modelLabel)}">
      <defs>
        <clipPath id="${clipId}">
          <rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" />
        </clipPath>
      </defs>
      <rect width="${width}" height="${height}" fill="#fff" />
      <g class="chart-grid">${grid}</g>
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" class="chart-axis" />
      <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}" class="chart-axis" />
      <line x1="${width - margin.right}" x2="${width - margin.right}" y1="${margin.top}" y2="${height - margin.bottom}" class="chart-axis" />
      <text x="${margin.left}" y="18" class="chart-title">${escapeHtml(metric.label)} (n=${points.length})</text>
      <text x="${margin.left}" y="${height - 7}" class="chart-axis-label">${escapeHtml(metric.xLabel)}</text>
      <text transform="translate(16 ${margin.top + plotHeight / 2}) rotate(-90)" class="chart-axis-label">${escapeHtml(metric.yLabel)}</text>
      <text transform="translate(${width - 12} ${margin.top + plotHeight / 2}) rotate(90)" class="chart-axis-label">Relative density</text>
      <g>${densityAxis}</g>
      <g clip-path="url(#${clipId})">${densityCurves}</g>
      <g clip-path="url(#${clipId})">${dots}</g>
    </svg>
  `;
}

function renderClientChart(metric, group, selectedModels, large = false) {
  const rows = metricRowsForGroup(metric, group, selectedModels);
  if (metric.chartType === "scatter") return renderScatterChart(metric, rows, group, selectedModels, large);
  return renderTurnSeriesChart(metric, rows, group, selectedModels, large);
}

function renderPlots() {
  const groups = selectedGroups();
  els.plotGrid.style.setProperty("--columns", state.columns);
  document.body.classList.toggle("size-compact", state.size === "compact");
  document.body.classList.toggle("size-standard", state.size === "standard");
  document.body.classList.toggle("size-large", state.size === "large");
  els.selectedSummary.textContent = `${groups.length} selected - ${state.metricIds.size} metrics`;
  els.plotGrid.innerHTML = "";

  if (!groups.length || !state.metricIds.size) {
    els.plotGrid.innerHTML = '<div class="empty-state">Select groups and metrics to build a board.</div>';
    return;
  }

  for (const group of groups) {
    const selectedModels = selectedModelsForGroup(group);
    const modelOptions = group.models
      .map(
        (model) => `
          <label class="model-option">
            <input type="checkbox" name="model-${escapeHtml(group.id)}" value="${escapeHtml(model)}" ${selectedModels?.has(model) ? "checked" : ""} />
            <span class="model-swatch" style="--model-color: ${modelColor(model, group.models)}"></span>
            <span>${escapeHtml(model)}</span>
          </label>
        `
      )
      .join("");
    const section = document.createElement("article");
    section.className = "plot-group";
    section.innerHTML = `
      <div class="plot-group-header">
        <div class="plot-group-heading">
          <div class="plot-group-title">${escapeHtml(group.workloadName)}</div>
          <div class="plot-group-controls">
            <div class="model-filter" data-group-id="${escapeHtml(group.id)}">
              <div class="model-filter-title">Models</div>
              <label class="model-option model-option-all">
                <input type="checkbox" value="__all__" ${selectedModels === null ? "checked" : ""} />
                <span>All models</span>
              </label>
              <div class="model-options">
                ${modelOptions}
              </div>
            </div>
          </div>
        </div>
        <div class="chip ${escapeHtml(group.arch)}">${escapeHtml(group.arch)}</div>
      </div>
      <div class="plot-strip"></div>
    `;

    section.querySelector(".model-filter").addEventListener("change", (event) => {
      if (!(event.target instanceof HTMLInputElement)) return;
      if (event.target.value === "__all__") {
        if (event.target.checked) state.modelsByGroup.delete(group.id);
      } else {
        const nextModels = selectedModelsForGroup(group) || new Set();
        if (event.target.checked) nextModels.add(event.target.value);
        else nextModels.delete(event.target.value);
        if (nextModels.size) state.modelsByGroup.set(group.id, nextModels);
        else state.modelsByGroup.delete(group.id);
      }
      renderPlots();
      writeStateToUrl();
    });

    const strip = section.querySelector(".plot-strip");
    for (const metric of metrics.filter((item) => state.metricIds.has(item.id))) {
      const card = document.createElement("figure");
      card.className = "plot-card";
      card.innerHTML = `
        <button type="button" title="${escapeHtml(metric.label)}">
          ${renderClientChart(metric, group, selectedModels)}
        </button>
        <figcaption class="plot-caption">${escapeHtml(metric.label)} - ${captionModelListHtml(group, selectedModels)}</figcaption>
      `;
      card.querySelector("button").addEventListener("click", () => openDialog(group, metric, selectedModels));
      strip.append(card);
    }
    els.plotGrid.append(section);
  }
}

function openDialog(group, metric, selectedModels) {
  const prefix = `${metric.label} - ${group.workloadName} - ${group.arch} - `;
  els.dialogTitle.innerHTML = `${escapeHtml(prefix)}${captionModelListHtml(group, selectedModels)}`;
  els.dialogImage.hidden = true;
  els.dialogImage.removeAttribute("src");
  els.dialogImage.alt = "";
  els.dialogChart.hidden = false;
  els.dialogChart.innerHTML = renderClientChart(metric, group, selectedModels, true);
  els.plotDialog.showModal();
}

function setPresentationMode(enabled) {
  document.body.classList.toggle("presentation", enabled);
  els.presentationButton.textContent = enabled ? "Exit" : "Present";
}

function syncControls() {
  document.querySelectorAll('input[name="metric"]').forEach((input) => {
    input.checked = state.metricIds.has(input.value);
  });
  els.columnsSelect.value = String(state.columns);
  els.sizeSelect.value = state.size;
}

function render() {
  syncControls();
  renderGroupList();
  renderPlots();
  writeStateToUrl();
}

function bindEvents() {
  els.refreshButton.addEventListener("click", loadData);
  els.searchInput.addEventListener("input", (event) => {
    state.search = event.target.value;
    renderGroupList();
  });

  document.querySelectorAll("[data-arch]").forEach((button) => {
    button.addEventListener("click", () => {
      state.arch = button.dataset.arch;
      document.querySelectorAll("[data-arch]").forEach((item) => item.classList.toggle("active", item === button));
      renderGroupList();
    });
  });

  els.selectVisibleButton.addEventListener("click", () => {
    for (const group of state.visibleGroups) state.selected.add(group.id);
    render();
  });

  els.clearButton.addEventListener("click", () => {
    state.selected.clear();
    render();
  });

  document.querySelectorAll('input[name="metric"]').forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.metricIds.add(input.value);
      else state.metricIds.delete(input.value);
      renderPlots();
      writeStateToUrl();
    });
  });

  els.columnsSelect.addEventListener("change", () => {
    state.columns = Number(els.columnsSelect.value);
    renderPlots();
    writeStateToUrl();
  });

  els.sizeSelect.addEventListener("change", () => {
    state.size = els.sizeSelect.value;
    renderPlots();
    writeStateToUrl();
  });

  els.copyLinkButton.addEventListener("click", async () => {
    writeStateToUrl();
    await navigator.clipboard.writeText(window.location.href);
    els.copyLinkButton.textContent = "Copied";
    setTimeout(() => {
      els.copyLinkButton.textContent = "Copy link";
    }, 1200);
  });

  els.presentationButton.addEventListener("click", async () => {
    const nextMode = !document.body.classList.contains("presentation");
    setPresentationMode(nextMode);
    if (nextMode && document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen().catch(() => {});
    } else if (!nextMode && document.fullscreenElement && document.exitFullscreen) {
      await document.exitFullscreen().catch(() => {});
    }
  });

  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && document.body.classList.contains("presentation")) {
      setPresentationMode(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("presentation") && !document.fullscreenElement) {
      setPresentationMode(false);
    }
  });

  els.closeDialogButton.addEventListener("click", () => els.plotDialog.close());
  els.plotDialog.addEventListener("click", (event) => {
    if (event.target === els.plotDialog) els.plotDialog.close();
  });
}

loadStateFromUrl();
bindEvents();
loadData();
