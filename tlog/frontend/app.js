/* tlog frontend — shared by `tlog export` (data inlined) and `tlog serve`
 * (data fetched + polled). Vanilla JS + uPlot, no build step. */
"use strict";

const MODE = window.TLOG_MODE; // "export" | "serve"
const POLL_MS = 3000;

const PALETTE = [
  "#5aa9e6", "#f4845f", "#7fc96b", "#c792ea",
  "#ffd166", "#4dd0e1", "#ef6292", "#b5c95a",
];
const GROUP_ORDER = ["loss", "eval", "training", "timing", "memory", "gpu", "cpu", "ram"];

const state = {
  runs: [], // [{id, name, project, status, step, config, summary, metrics?, media?, console?}]
  selected: new Set(),
  smoothing: 0,
  logScale: false,
  tab: "charts",
  mediaKey: null,
  consoleRun: null,
};
const charts = new Map(); // key -> {plot, sig}
let chartsStructureSig = "";

/* ---------- helpers ---------- */

const $ = (sel, el) => (el || document).querySelector(sel);

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

function runColor(run) {
  const i = state.runs.findIndex((r) => r.id === run.id);
  return PALETTE[(i >= 0 ? i : 0) % PALETTE.length];
}

function selectedRuns() {
  return state.runs.filter((r) => state.selected.has(r.id));
}

function fmtStep(s) {
  if (s == null) return "-";
  if (s >= 1e6) return (s / 1e6).toFixed(2) + "M";
  if (s >= 1e4) return (s / 1e3).toFixed(1) + "k";
  return String(s);
}

function ema(arr, w) {
  if (!w) return arr;
  let last = 0, debias = 0;
  return arr.map((v) => {
    if (v == null || !isFinite(v)) return null;
    last = last * w + (1 - w) * v;
    debias = debias * w + (1 - w);
    return last / debias;
  });
}

function groupKeys(keys) {
  const groups = new Map();
  for (const k of keys) {
    const g = k.includes("/") ? k.split("/")[0] : "metrics";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(k);
  }
  const ordered = [];
  for (const g of GROUP_ORDER) if (groups.has(g)) ordered.push([g, groups.get(g)]);
  for (const [g, ks] of [...groups].sort()) {
    if (!GROUP_ORDER.includes(g)) ordered.push([g, ks]);
  }
  return ordered;
}

/* ---------- data (serve mode) ---------- */

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function poll() {
  try {
    const payload = await fetchJSON("/api/runs");
    const byId = new Map(state.runs.map((r) => [r.id, r]));
    const merged = [];
    for (const summary of payload.runs) {
      const prev = byId.get(summary.id) || {};
      merged.push(Object.assign({}, prev, summary, {
        metrics: prev.metrics,
        media: prev.media,
        _metricsMtime: prev._metricsMtime,
        _mediaMtime: prev._mediaMtime,
        metrics_mtime: summary.metrics_mtime,
        media_mtime: summary.media_mtime,
      }));
    }
    state.runs = merged;
    if (state.selected.size === 0 && merged.length) {
      const running = merged.find((r) => r.status === "running");
      state.selected.add((running || merged[0]).id);
    }
    await Promise.all(selectedRuns().map(refreshRunData));
    render();
  } catch (e) {
    $("#status-note").textContent = "connection lost — retrying…";
  }
}

async function refreshRunData(run) {
  if (run.metrics === undefined || run._metricsMtime !== run.metrics_mtime) {
    const m = await fetchJSON(`/api/run/${run.id}/metrics`);
    run.metrics = m.metrics;
    run._metricsMtime = run.metrics_mtime;
  }
  if (run.media === undefined || run._mediaMtime !== run.media_mtime) {
    const m = await fetchJSON(`/api/run/${run.id}/media`);
    run.media = m.media;
    run._mediaMtime = run.media_mtime;
  }
}

/* ---------- sidebar ---------- */

function renderSidebar() {
  const side = $("#sidebar");
  side.replaceChildren(
    el("div", { id: "logo" }, "tlog", el("span", { class: "dot" }, "▮")),
    el("div", { id: "subtitle" },
      MODE === "serve" ? "live dashboard" : "exported report"),
    el("div", { class: "side-section" }, "runs"),
    ...renderRunList(),
    el("div", { class: "side-section" }, "display"),
    el("div", { class: "control" },
      el("label", {}, el("span", {}, "smoothing"),
        el("span", { id: "smooth-val", class: "mono" }, state.smoothing.toFixed(2))),
      el("input", {
        type: "range", min: "0", max: "0.99", step: "0.01",
        value: String(state.smoothing),
        oninput: (e) => {
          state.smoothing = parseFloat(e.target.value);
          $("#smooth-val").textContent = state.smoothing.toFixed(2);
          if (state.tab === "charts") renderCharts();
        },
      })),
    el("label", { class: "toggle" },
      el("input", {
        type: "checkbox",
        ...(state.logScale ? { checked: "" } : {}),
        onchange: (e) => {
          state.logScale = e.target.checked;
          if (state.tab === "charts") renderCharts(true);
        },
      }),
      "log scale (y)"),
  );
}

function renderRunList() {
  const byProject = new Map();
  for (const r of state.runs) {
    if (!byProject.has(r.project)) byProject.set(r.project, []);
    byProject.get(r.project).push(r);
  }
  const rows = [];
  const multi = byProject.size > 1;
  for (const [project, runs] of byProject) {
    if (multi) rows.push(el("div", { class: "side-section" }, project));
    for (const r of runs) {
      const sel = state.selected.has(r.id);
      rows.push(
        el("div", {
          class: `run-row ${sel ? "selected" : "deselected"}`,
          title: `${r.project}/${r.name} (${r.id})`,
          onclick: async () => {
            if (sel) state.selected.delete(r.id);
            else {
              state.selected.add(r.id);
              if (MODE === "serve") await refreshRunData(r);
            }
            render();
          },
        },
          el("span", { class: "run-dot", style: `background:${runColor(r)}` }),
          el("span", { class: "run-name" }, r.name),
          el("span", { class: "run-meta mono" }, fmtStep(r.step)),
          el("span", { class: `badge ${r.status}` }, r.status)),
      );
    }
  }
  if (!rows.length) rows.push(el("div", { class: "empty-note" }, "no runs found"));
  return rows;
}

/* ---------- tabs / topbar ---------- */

const TABS = ["charts", "media", "config", "console"];

function renderTopbar() {
  $("#tabs").replaceChildren(
    ...TABS.map((t) =>
      el("button", {
        class: `tab ${state.tab === t ? "active" : ""}`,
        onclick: () => { state.tab = t; render(); },
      }, t)),
  );
  const note = $("#status-note");
  if (MODE === "serve") {
    note.replaceChildren(el("span", { id: "live-dot" }), `auto-refresh ${POLL_MS / 1000}s`);
  } else {
    const gen = window.TLOG_DATA && window.TLOG_DATA.generated_at;
    note.textContent = gen ? `exported ${gen}` : "";
  }
}

/* ---------- charts tab ---------- */

function destroyCharts() {
  for (const { plot } of charts.values()) plot.destroy();
  charts.clear();
  chartsStructureSig = "";
}

function axisOpts() {
  return {
    stroke: "#8b93a3",
    grid: { stroke: "#232936", width: 1 },
    ticks: { stroke: "#232936" },
    font: "11px ui-monospace, Menlo, monospace",
  };
}

function chartData(key, runs) {
  const stepSet = new Set();
  for (const r of runs) {
    const m = r.metrics && r.metrics[key];
    if (m) for (const s of m.steps) stepSet.add(s);
  }
  const xs = [...stepSet].sort((a, b) => a - b);
  const idx = new Map(xs.map((s, i) => [s, i]));
  const tf = state.logScale ? (v) => (v > 0 ? v : null) : (v) => v;
  const series = runs.map((r) => {
    const m = r.metrics && r.metrics[key];
    const arr = new Array(xs.length).fill(null);
    if (m) m.steps.forEach((s, i) => { arr[idx.get(s)] = tf(m.values[i]); });
    return ema(arr, state.smoothing);
  });
  return [xs, ...series];
}

function renderCharts(forceRebuild) {
  const content = $("#content");
  const runs = selectedRuns();
  const keySet = new Set();
  for (const r of runs) for (const k of Object.keys(r.metrics || {})) keySet.add(k);
  const keys = [...keySet].sort();

  const sig = JSON.stringify([keys, runs.map((r) => r.id), state.logScale]);
  if (!forceRebuild && sig === chartsStructureSig && state.tab === "charts") {
    for (const [key, entry] of charts) entry.plot.setData(chartData(key, runs));
    return;
  }

  destroyCharts();
  chartsStructureSig = sig;
  content.replaceChildren();
  if (!runs.length) {
    content.append(el("div", { class: "empty-note" }, "select a run in the sidebar"));
    return;
  }
  if (!keys.length) {
    content.append(el("div", { class: "empty-note" }, "no metrics logged yet"));
    return;
  }

  for (const [group, groupKeysList] of groupKeys(keys)) {
    content.append(el("div", { class: "group-title" }, group));
    const grid = el("div", { class: "chart-grid" });
    content.append(grid);
    for (const key of groupKeysList) {
      const card = el("div", { class: "chart-card" }, el("h4", {}, key));
      grid.append(card);
      const data = chartData(key, runs);
      const plot = new uPlot({
        width: Math.max(320, card.clientWidth - 26),
        height: 200,
        scales: { x: { time: false }, y: { distr: state.logScale ? 3 : 1 } },
        axes: [axisOpts(), Object.assign(axisOpts(), { size: 56 })],
        series: [
          { label: "step" },
          ...runs.map((r) => ({
            label: r.name,
            stroke: runColor(r),
            width: 1.6,
            spanGaps: true,
            points: { show: false },
          })),
        ],
        cursor: { sync: { key: "tlog" }, points: { size: 5 } },
        legend: { live: true },
      }, data, card);
      charts.set(key, { plot, sig });
    }
  }
}

window.addEventListener("resize", () => {
  for (const { plot } of charts.values()) {
    const card = plot.root.closest(".chart-card");
    if (card) plot.setSize({ width: Math.max(320, card.clientWidth - 26), height: 200 });
  }
});

/* ---------- media tab ---------- */

function mediaSrc(run, file) {
  return MODE === "serve" ? `/media/${run.id}/${file}` : file;
}

function renderMedia() {
  const content = $("#content");
  destroyCharts();
  content.replaceChildren();
  const runs = selectedRuns();
  const keySet = new Set();
  for (const r of runs) for (const rec of r.media || []) keySet.add(rec.key);
  const keys = [...keySet].sort();
  if (!keys.length) {
    content.append(el("div", { class: "empty-note" },
      "no media logged — use tlog.log_images(key, images, step=...)"));
    return;
  }
  if (!keys.includes(state.mediaKey)) state.mediaKey = keys[0];

  content.append(el("div", { class: "media-chips" },
    keys.map((k) => el("button", {
      class: `chip ${k === state.mediaKey ? "active" : ""}`,
      onclick: () => { state.mediaKey = k; renderMedia(); },
    }, k))));

  const stepSet = new Set();
  for (const r of runs) {
    for (const rec of r.media || []) if (rec.key === state.mediaKey) stepSet.add(rec.step);
  }
  const steps = [...stepSet].sort((a, b) => b - a);

  const header = el("tr", {}, el("th", {}, "step"),
    runs.map((r) => el("th", { style: `color:${runColor(r)}` }, r.name)));
  const body = steps.map((step) =>
    el("tr", {},
      el("td", { class: "media-step mono" }, fmtStep(step)),
      runs.map((r) => {
        const recs = (r.media || []).filter(
          (m) => m.key === state.mediaKey && m.step === step);
        return el("td", { class: "media-cell" },
          recs.flatMap((rec) =>
            rec.files.map((f) =>
              el("img", {
                src: mediaSrc(r, f),
                loading: "lazy",
                onclick: () => showLightbox(mediaSrc(r, f),
                  `${r.name} · ${state.mediaKey} · step ${step}` +
                  (rec.caption ? ` · ${rec.caption}` : "")),
              }))),
          recs.length && recs[0].caption
            ? el("div", { class: "media-caption" }, recs[0].caption) : null);
      })));
  content.append(el("table", { class: "media-table" },
    el("thead", {}, header), el("tbody", {}, body)));
}

function showLightbox(src, caption) {
  const box = $("#lightbox");
  box.replaceChildren(
    el("img", { src }),
    caption ? el("div", { class: "media-caption" }, caption) : null);
  box.classList.remove("hidden");
  box.onclick = () => box.classList.add("hidden");
}

/* ---------- config tab ---------- */

function renderConfig() {
  const content = $("#content");
  destroyCharts();
  content.replaceChildren();
  const runs = selectedRuns();
  if (!runs.length) {
    content.append(el("div", { class: "empty-note" }, "select a run in the sidebar"));
    return;
  }

  content.append(el("div", { class: "group-title" }, "run info"));
  const infoKeys = ["id", "status", "created", "step", "host", "slurm job", "git", "dir"];
  const infoRows = runs.map((r) => [
    r.id, r.status, r.summary?.created || "", fmtStep(r.step),
    r.summary?.hostname || "", r.summary?.slurm_job || "",
    (r.summary?.git_commit || "").slice(0, 8) + (r.summary?.git_dirty ? " (dirty)" : ""),
    r.summary?.dir || "",
  ]);
  content.append(el("table", { class: "config-table" },
    el("thead", {}, el("tr", {}, el("th", {}, ""),
      runs.map((r) => el("th", { style: `color:${runColor(r)}` }, r.name)))),
    el("tbody", {}, infoKeys.map((k, ki) =>
      el("tr", {}, el("td", { class: "key" }, k),
        infoRows.map((row) => el("td", { class: "val" }, String(row[ki] ?? ""))))))));

  content.append(el("div", { class: "group-title" }, "config"));
  const keySet = new Set();
  for (const r of runs) for (const k of Object.keys(r.config || {})) keySet.add(k);
  const keys = [...keySet].sort();
  const rows = keys.map((k) => {
    const vals = runs.map((r) => JSON.stringify(r.config?.[k] ?? null));
    const differs = runs.length > 1 && new Set(vals).size > 1;
    return el("tr", { class: differs ? "diff" : "" },
      el("td", { class: "key" }, k),
      vals.map((v) => el("td", { class: "val" }, v ?? "")));
  });
  content.append(el("table", { class: "config-table" },
    el("thead", {}, el("tr", {}, el("th", {}, "key"),
      runs.map((r) => el("th", { style: `color:${runColor(r)}` }, r.name)))),
    el("tbody", {}, rows)));
}

/* ---------- console tab ---------- */

async function renderConsole() {
  const content = $("#content");
  destroyCharts();
  content.replaceChildren();
  const runs = selectedRuns();
  if (!runs.length) {
    content.append(el("div", { class: "empty-note" }, "select a run in the sidebar"));
    return;
  }
  if (!runs.find((r) => r.id === state.consoleRun)) state.consoleRun = runs[0].id;
  const active = runs.find((r) => r.id === state.consoleRun);

  content.append(el("div", { class: "console-tabs" },
    runs.map((r) => el("button", {
      class: `chip ${r.id === state.consoleRun ? "active" : ""}`,
      onclick: () => { state.consoleRun = r.id; renderConsole(); },
    }, r.name))));
  const pre = el("pre", { id: "console-pre" }, "");
  content.append(pre);

  if (MODE === "serve") {
    try {
      const res = await fetch(`/api/run/${active.id}/console`);
      pre.textContent = await res.text();
    } catch { pre.textContent = "(console unavailable)"; }
  } else {
    pre.textContent = active.console || "(no console captured)";
  }
  pre.scrollTop = pre.scrollHeight;
}

/* ---------- root render ---------- */

function render() {
  renderSidebar();
  renderTopbar();
  if (state.tab === "charts") renderCharts();
  else if (state.tab === "media") renderMedia();
  else if (state.tab === "config") renderConfig();
  else renderConsole();
}

function init() {
  document.title = MODE === "serve" ? "tlog — live" : "tlog — report";
  if (MODE === "export") {
    state.runs = window.TLOG_DATA.runs;
    for (const r of state.runs) state.selected.add(r.id);
    render();
  } else {
    render();
    poll();
    setInterval(poll, POLL_MS);
  }
}

init();
