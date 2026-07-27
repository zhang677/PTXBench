const CSV_PATH = "../figures/correctness_rate_by_release_date.csv";
const PLOT_DIR = "../figures/by_workload_arch";

const metrics = [
  {
    id: "correctness_rate_by_release_date",
    label: "Correctness rate",
  },
  {
    id: "best_speedup_by_release_date",
    label: "Best speedup",
  },
  {
    id: "arch_instruction_correctness_by_release_date",
    label: "Correctness with architecture instructions",
  },
  {
    id: "reasoning_tokens_vs_performance",
    label: "Perf-reasoning",
  },
];

const state = {
  groups: [],
  visibleGroups: [],
  selected: new Set(),
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
  dialogTitle: document.querySelector("#dialogTitle"),
  closeDialogButton: document.querySelector("#closeDialogButton"),
};

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

function slugify(value) {
  return (
    value
      .trim()
      .split("")
      .map((ch) => (/[a-zA-Z0-9]/.test(ch) ? ch.toLowerCase() : "_"))
      .join("")
      .split("_")
      .filter(Boolean)
      .join("_") || "unknown"
  );
}

function groupId(group) {
  return `${group.workloadName}__${group.arch}`;
}

function plotPath(group, metricId) {
  const path = `${PLOT_DIR}/${metricId}__${slugify(group.workloadName)}__${slugify(group.arch)}.png`;
  return state.assetCacheToken ? `${path}?t=${state.assetCacheToken}` : path;
}

function loadStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const selected = params.get("groups");
  const metricIds = params.get("metrics");
  const columns = Number(params.get("columns"));
  const size = params.get("size");

  if (selected) state.selected = new Set(selected.split(",").filter(Boolean));
  if (metricIds) {
    const valid = new Set(metrics.map((metric) => metric.id));
    state.metricIds = new Set(metricIds.split(",").filter((id) => valid.has(id)));
  }
  if ([1, 2, 3].includes(columns)) state.columns = columns;
  if (["compact", "standard", "large"].includes(size)) state.size = size;
}

function writeStateToUrl() {
  const params = new URLSearchParams();
  if (state.selected.size) params.set("groups", [...state.selected].join(","));
  params.set("metrics", [...state.metricIds].join(","));
  params.set("columns", String(state.columns));
  params.set("size", state.size);
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

async function loadData() {
  els.datasetSummary.textContent = "Loading plots";
  els.plotGrid.innerHTML = "";
  try {
    state.assetCacheToken = String(Date.now());
    const response = await fetch(`${CSV_PATH}?t=${state.assetCacheToken}`);
    if (!response.ok) throw new Error(`Could not read ${CSV_PATH}`);
    const rows = parseCsv(await response.text());
    state.groups = buildGroups(rows);

    if (!state.selected.size) {
      for (const group of state.groups.slice(0, 2)) state.selected.add(group.id);
    }

    els.datasetSummary.textContent = `${state.groups.length} groups from ${rows.length} rows`;
    render();
  } catch (error) {
    els.datasetSummary.textContent = "Could not load plots";
    els.plotGrid.innerHTML = `<div class="error-state">${error.message}</div>`;
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
      <input type="checkbox" value="${group.id}" ${state.selected.has(group.id) ? "checked" : ""} />
      <span>
        <span class="group-name">${group.workloadName}</span>
        <span class="group-meta">
          <span class="chip ${group.arch}">${group.arch}</span>
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
    const section = document.createElement("article");
    section.className = "plot-group";
    section.innerHTML = `
      <div class="plot-group-header">
        <div class="plot-group-title">${group.workloadName}</div>
        <div class="chip ${group.arch}">${group.arch}</div>
      </div>
      <div class="plot-strip"></div>
    `;

    const strip = section.querySelector(".plot-strip");
    for (const metric of metrics.filter((item) => state.metricIds.has(item.id))) {
      const src = plotPath(group, metric.id);
      const card = document.createElement("figure");
      card.className = "plot-card";
      card.innerHTML = `
        <button type="button" title="${metric.label}">
          <img src="${src}" alt="${metric.label} for ${group.workloadName} on ${group.arch}" loading="lazy" />
        </button>
        <figcaption class="plot-caption">${metric.label}</figcaption>
      `;
      card.querySelector("button").addEventListener("click", () => openDialog(group, metric, src));
      strip.append(card);
    }
    els.plotGrid.append(section);
  }
}

function openDialog(group, metric, src) {
  els.dialogTitle.textContent = `${metric.label} - ${group.workloadName} - ${group.arch}`;
  els.dialogImage.src = src;
  els.dialogImage.alt = els.dialogTitle.textContent;
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
