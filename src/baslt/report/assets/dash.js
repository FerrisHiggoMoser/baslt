(() => {
"use strict";
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const M = JSON.parse($("#manifest").textContent);
const selftest = new URLSearchParams(location.search).has("selftest");
const started = performance.now();
const errors = [];
const VERDICTS = ["pass", "warn", "fail", "not_applicable", "error", "not checked"];
const COLOR = ["--pass", "--warn", "--fail", "--na", "--error", "--line"];
const RANK = [3, 2, 0, 4, 1, 5];
const ELLIPSIS = String.fromCharCode(0x2026);
let buffer = null;
let codes = null;
const state = { req: 0, run: -1, param: "", pct: false, visible: [], sort: null };
window.addEventListener("error", (e) => errors.push(String(e.message || e)));

// ----- data ---------------------------------------------------------------------------------------------

async function inflate(text) {
  const bin = atob(text.trim());
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  return new Response(stream).arrayBuffer();
}

const decoded = new Map();
function decode(ref) {
  if (!ref) return null;
  let out = decoded.get(ref);
  if (out) return out;
  if (ref.k === "f4") out = new Float32Array(buffer, ref.o, ref.n);
  else if (ref.k === "u1") out = new Uint8Array(buffer, ref.o, ref.n);
  else {
    const wide = ref.k === "u2";
    const raw = wide ? new Uint16Array(buffer, ref.o, ref.n) : new Uint8Array(buffer, ref.o, ref.n);
    const top = wide ? 65534 : 254;
    const step = (ref.hi - ref.lo) / top;
    out = new Float64Array(ref.n);
    for (let i = 0; i < ref.n; i++) out[i] = raw[i] > top ? NaN : ref.lo + raw[i] * step;
  }
  decoded.set(ref, out);
  return out;
}

function allRefs() {
  const refs = [];
  const walk = (v) => {
    if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") {
      if (typeof v.o === "number" && typeof v.k === "string") refs.push(v);
      else Object.values(v).forEach(walk);
    }
  };
  walk(M.matrix); walk(M.reqs); walk(M.series);
  return refs.sort((a, b) => a.o - b.o);
}

// ----- formatting ---------------------------------------------------------------------------------------

const PREFIXED = { Pa: [["MPa", 1e6], ["kPa", 1e3]], N: [["MN", 1e6], ["kN", 1e3]], m: [["km", 1e3]],
  W: [["MW", 1e6], ["kW", 1e3]], J: [["MJ", 1e6], ["kJ", 1e3]] };

function sig(v, digits = 4) {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "n/a" : (v > 0 ? "inf" : "-inf");
  let text = v.toPrecision(digits);
  if (text.includes("e")) {
    if (Math.abs(v) >= 10 ** digits && Math.abs(v) < 1e9) return v.toFixed(0);
    return text.replace(/\.?0+e/, "e");
  }
  if (text.includes(".")) text = text.replace(/\.?0+$/, "");
  return text === "-0" ? "0" : text;
}

function withUnit(v, unit) {
  if (!Number.isFinite(v)) return sig(v);
  if (!unit) return sig(v);
  for (const [name, factor] of PREFIXED[unit] || []) if (Math.abs(v) >= factor) return `${sig(v / factor)} ${name}`;
  return `${sig(v)} ${unit}`;
}

function ticks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function tickText(v, step) {
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function badge(verdict) {
  const span = document.createElement("span");
  const cls = { pass: "pass", warn: "warn", fail: "fail", not_applicable: "na", error: "error" }[verdict] || "na";
  span.className = `badge v-${cls}`;
  span.textContent = M.labels[verdict] || verdict;
  return span;
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(rect.width, 10), h = Math.max(rect.height, 10);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif";
  return [ctx, w, h];
}

function readout(box, x, lines, limit) {
  box.replaceChildren(...lines.map((line) => { const d = document.createElement("div"); d.textContent = line; return d; }));
  box.hidden = false;
  const width = box.offsetWidth;
  box.style.left = `${x + 14 + width > limit ? Math.max(0, x - 14 - width) : x + 14}px`;
}

function openRun(index) {
  const run = M.runs[index];
  if (run) location.href = run.page || run.file;
}

function marginOf(req, i) {
  const v = decode(state.pct && req.pct ? req.pct : req.margin)[i];
  if (!state.pct) return v;
  return req.pct ? v * 100 : v * req.pct_scale * 100;
}

function marginText(req, i) {
  const v = marginOf(req, i);
  return state.pct ? `${sig(v)} %` : withUnit(v, req.unit);
}

// ----- filter -------------------------------------------------------------------------------------------

function parseFilter(text) {
  const tests = [];
  for (const word of text.trim().split(/\s+/).filter(Boolean)) {
    const m = word.match(/^([A-Za-z_][\w.]*)(<=|>=|!=|=|<|>)(.+)$/);
    if (!m) { const low = word.toLowerCase(); tests.push((run) => haystack(run).includes(low)); continue; }
    const [, name, op, raw] = m;
    const num = Number(raw);
    tests.push((run) => {
      const value = name === "status" ? run.status : (name === "run" || name === "id" ? run.id : run.params[name]);
      if (value === undefined || value === null || value === "") return false;
      if (Number.isFinite(num) && typeof value === "number") {
        return { "<": value < num, "<=": value <= num, ">": value > num, ">=": value >= num, "=": value === num,
          "!=": value !== num }[op];
      }
      const a = String(value).toLowerCase(), b = raw.toLowerCase();
      if (name === "status" && b === "n/a") return op === "!=" ? a !== "not_applicable" : a === "not_applicable";
      return op === "!=" ? a !== b : op === "=" ? a === b : false;
    });
  }
  return (run) => tests.every((t) => t(run));
}

const hay = new Map();
function haystack(run) {
  let text = hay.get(run);
  if (text === undefined) {
    text = [run.id, run.status, ...Object.entries(run.params).map(([k, v]) => `${k}=${v}`)].join(" ").toLowerCase();
    hay.set(run, text);
  }
  return text;
}

function applyFilter() {
  const test = parseFilter($("#run-filter").value || "");
  const order = M.runs.map((run, i) => i).filter((i) => test(M.runs[i]));
  const rank = { fail: 0, error: 1, warn: 2, pass: 3, not_applicable: 4 };
  order.sort((a, b) => (rank[M.runs[a].status] - rank[M.runs[b].status]) || a - b);
  state.visible = order;
  $("#run-count").textContent = `${order.length} of ${M.runs.length} runs`;
  drawAll();
  runTable.refresh();
}

// ----- verdict matrix -----------------------------------------------------------------------------------

const matrix = {
  frame: null,
  draw() {
    const canvas = $(".matrix canvas");
    const rows = state.visible.length, cols = M.reqs.length;
    const width = canvas.parentElement.clientWidth;
    const cellW = Math.max(3, Math.min(40, Math.floor((width - 150) / Math.max(cols, 1))));
    const cellH = Math.max(2, Math.min(18, Math.floor(640 / Math.max(rows, 1))));
    const left = cellH >= 11 ? 140 : 8, top = cellW >= 11 ? 70 : 8;
    canvas.style.height = `${top + rows * cellH + 8}px`;
    canvas.style.width = `${Math.max(width, left + cols * cellW + 8)}px`;
    const [ctx, w] = fitCanvas(canvas);
    this.frame = { left, top, cellW, cellH, rows, cols };
    const colors = COLOR.map(css);
    for (let r = 0; r < rows; r++) {
      const i = state.visible[r];
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = colors[codes[i * cols + c]];
        ctx.fillRect(left + c * cellW, top + r * cellH, Math.max(cellW - (cellW > 4 ? 1 : 0), 1),
          Math.max(cellH - (cellH > 4 ? 1 : 0), 1));
      }
    }
    ctx.fillStyle = css("--muted");
    if (left > 8) {
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let r = 0; r < rows; r++) {
        const id = M.runs[state.visible[r]].id;
        ctx.fillText(id.length > 22 ? id.slice(0, 21) + ELLIPSIS : id, left - 6, top + r * cellH + cellH / 2);
      }
    }
    if (top > 8) {
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      M.reqs.forEach((req, c) => {
        ctx.save();
        ctx.translate(left + c * cellW + cellW / 2, top - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(req.id.length > 12 ? req.id.slice(0, 11) + ELLIPSIS : req.id, 0, 0);
        ctx.restore();
      });
    }
    ctx.strokeStyle = css("--fg");
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left + state.req * cellW - 0.5, top - 0.5, cellW, rows * cellH);
    const selected = state.visible.indexOf(state.run);
    if (selected >= 0) ctx.strokeRect(left - 0.5, top + selected * cellH - 0.5, cols * cellW, cellH);
    return w;
  },
  cell(e) {
    const f = this.frame;
    if (!f) return null;
    const c = Math.floor((e.offsetX - f.left) / f.cellW), r = Math.floor((e.offsetY - f.top) / f.cellH);
    if (c < 0 || c >= f.cols || r < 0 || r >= f.rows) return null;
    return [state.visible[r], c];
  },
  bind() {
    const canvas = $(".matrix canvas"), box = $(".matrix .readout");
    canvas.addEventListener("pointermove", (e) => {
      const hit = this.cell(e);
      if (!hit) { box.hidden = true; return; }
      const [i, c] = hit;
      const req = M.reqs[c], run = M.runs[i];
      const code = codes[i * M.reqs.length + c];
      const lines = [`${run.id}: ${req.id}`, `${VERDICTS[code] === "not checked" ? "not checked" : M.labels[VERDICTS[code]]}`];
      const m = marginOf(req, i);
      if (Number.isFinite(m)) lines.push(`margin ${marginText(req, i)}`);
      readout(box, e.offsetX, lines, canvas.clientWidth);
      box.style.top = `${Math.max(0, e.offsetY - 50)}px`;
    });
    canvas.addEventListener("pointerleave", () => { box.hidden = true; });
    canvas.addEventListener("click", (e) => {
      const hit = this.cell(e);
      if (!hit) return;
      select(hit[1], hit[0]);
    });
  },
};

function select(req, run) {
  state.req = req;
  if (run !== undefined) state.run = run;
  $("#req-select").value = String(req);
  $$("table.reqs tr.pick").forEach((tr) => tr.classList.toggle("chosen", Number(tr.dataset.req) === req));
  const r = M.reqs[req];
  $("#req-sentence").textContent = `${r.id} ${r.title}: ${r.sentence}`;
  const line = $("#selection");
  line.replaceChildren();
  if (state.run >= 0) {
    const runInfo = M.runs[state.run];
    const code = codes[state.run * M.reqs.length + req];
    line.append(document.createTextNode(`${runInfo.id} `), badge(VERDICTS[code] === "not checked" ? "not_applicable" : VERDICTS[code]),
      document.createTextNode(` ${r.id}`));
    const m = marginOf(r, state.run);
    if (Number.isFinite(m)) line.append(document.createTextNode(`, margin ${marginText(r, state.run)}`));
    const link = document.createElement("a");
    link.href = runInfo.page || runInfo.file;
    link.textContent = runInfo.page ? "open its report" : "open its results";
    line.append(document.createTextNode(" - "), link);
  }
  drawAll();
}

// ----- overlay ------------------------------------------------------------------------------------------

const PAD = { left: 62, right: 12, top: 10, bottom: 26 };

function frameFor(canvas, xlo, xhi, ylo, yhi) {
  const [ctx, w, h] = fitCanvas(canvas);
  const L = PAD.left, R = w - PAD.right, T = PAD.top, B = h - PAD.bottom;
  const sx = (x) => L + ((x - xlo) / (xhi - xlo || 1)) * (R - L);
  const sy = (y) => B - ((y - ylo) / (yhi - ylo || 1)) * (B - T);
  return { ctx, w, h, L, R, T, B, sx, sy, xlo, xhi, ylo, yhi };
}

function axes(f, xlabel, labels, yscale = 1) {
  const { ctx, L, R, T, B, sx, sy } = f;
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--axis"); ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const yt = labels ? labels.map(([code]) => code).filter((c) => c >= f.ylo && c <= f.yhi)
    : ticks(f.ylo / yscale, f.yhi / yscale, Math.max(2, Math.floor((B - T) / 40))).map((v) => v * yscale);
  const ystep = yt.length > 1 ? (yt[1] - yt[0]) / yscale : 1;
  for (const v of yt) {
    const y = Math.round(sy(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
    let text = labels ? String((labels.find(([c]) => c === v) || [0, v])[1]) : tickText(v / yscale, ystep);
    if (text.length > 9) text = text.slice(0, 8) + ELLIPSIS;
    ctx.fillText(text, L - 6, y);
  }
  const xt = ticks(f.xlo, f.xhi, Math.max(2, Math.floor((R - L) / 80)));
  const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of xt) {
    const x = Math.round(sx(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
    ctx.fillText(tickText(v, xstep), x, B + 6);
  }
  ctx.textAlign = "right";
  ctx.fillText(xlabel, R, B + 6 + 11);
}

function unitScale(lo, hi, unit) {
  const top = Math.max(Math.abs(lo), Math.abs(hi));
  for (const [name, factor] of PREFIXED[unit] || []) if (top >= factor) return [factor, name];
  return [1, unit || ""];
}

function legend(el, items) {
  el.replaceChildren(...items.map(([cls, color, text]) => {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = cls;
    if (cls === "dash") swatch.style.color = color; else swatch.style.background = color;
    span.append(swatch, document.createTextNode(text));
    return span;
  }));
}

const overlay = {
  frame: null,
  draw() {
    const el = $(".plot.overlay");
    const req = M.reqs[state.req];
    const canvas = $("canvas", el);
    const plot = req.plot;
    if (!plot) {
      const [ctx, w, h] = fitCanvas(canvas);
      ctx.fillStyle = css("--muted"); ctx.textAlign = "center";
      ctx.fillText("This requirement reads no signal to draw.", w / 2, h / 2);
      legend($(".legend", el), []);
      this.frame = null;
      return;
    }
    const s = M.series[plot.series];
    const bands = Object.fromEntries(Object.entries(s.bands).map(([k, v]) => [k, decode(v)]));
    const [x0, x1] = M.span;
    const n = M.bins;
    const xs = (k) => x0 + (k + 0.5) * (x1 - x0) / n;
    const shown = new Set(state.visible);
    const allLo = decode(s.trace_lo), allHi = decode(s.trace_hi);
    const traces = plot.traces.filter((t) => shown.has(t.run)).map((t) => {
      const at = s.trace_runs.indexOf(t.run) * n;
      return { ...t, lo: allLo.subarray(at, at + n), hi: allHi.subarray(at, at + n) };
    });
    let lo = Infinity, hi = -Infinity;
    const take = (v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
    bands.min.forEach(take); bands.max.forEach(take);
    for (const side of ["upper", "lower"]) (plot[side] || []).forEach(take);
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
    const pad = s.labels ? 0.5 : (hi - lo) * 0.06;
    const f = frameFor(canvas, x0, x1, lo - pad, hi + pad);
    this.frame = f;
    const [factor, unitName] = unitScale(lo, hi, s.unit);
    axes(f, M.t0_event ? `s after ${M.t0_event}` : "s", s.labels, factor);
    const { ctx, sx, sy, L, R, T, B } = f;
    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, R - L, B - T); ctx.clip();
    const area = (a, b, color) => {
      ctx.fillStyle = color;
      let open = false;
      ctx.beginPath();
      const flush = (from, to) => {
        if (to <= from) return;
        ctx.moveTo(sx(xs(from)), sy(a[from]));
        for (let k = from + 1; k < to; k++) ctx.lineTo(sx(xs(k)), sy(a[k]));
        for (let k = to - 1; k >= from; k--) ctx.lineTo(sx(xs(k)), sy(b[k]));
        ctx.closePath();
      };
      let start = 0;
      for (let k = 0; k <= n; k++) {
        const ok = k < n && Number.isFinite(a[k]) && Number.isFinite(b[k]);
        if (ok && !open) { start = k; open = true; }
        if (!ok && open) { flush(start, k); open = false; }
      }
      ctx.fill();
    };
    const line = (v, color, width, dash = []) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath();
      let open = false;
      for (let k = 0; k < n; k++) {
        if (!Number.isFinite(v[k])) { open = false; continue; }
        if (open) ctx.lineTo(sx(xs(k)), sy(v[k])); else { ctx.moveTo(sx(xs(k)), sy(v[k])); open = true; }
      }
      ctx.stroke();
    };
    ctx.globalAlpha = 0.18; area(bands.min, bands.max, css("--trace-1"));
    ctx.globalAlpha = 0.28; area(bands.p05, bands.p95, css("--trace-1"));
    ctx.globalAlpha = 1;
    line(bands.p50, css("--trace-1"), 1.5);
    const traceColor = { 2: css("--fail"), 1: css("--warn"), 4: css("--error") };
    // each run is drawn on the side its limit looks at: the highest values for an upper limit
    const upperOnly = (plot.upper || plot.upper_curve) && !(plot.lower || plot.lower_curve);
    const lowerOnly = (plot.lower || plot.lower_curve) && !(plot.upper || plot.upper_curve);
    for (const t of traces.slice().reverse()) {
      ctx.globalAlpha = t.run === state.run ? 1 : 0.75;
      const color = traceColor[t.v] || css("--trace-2");
      const width = t.run === state.run ? 2.2 : 1;
      if (upperOnly) line(t.hi, color, width);
      else if (lowerOnly) line(t.lo, color, width);
      else if (plot.upper || plot.lower || plot.upper_curve || plot.lower_curve) {
        line(t.hi, color, width); line(t.lo, color, width);
      } else line(t.lo.map((v, k) => (v + t.hi[k]) / 2), color, width);
    }
    ctx.globalAlpha = 1;
    for (const side of ["upper", "lower"]) {
      for (const v of plot[side] || []) {
        const y = Math.round(sy(v)) + 0.5;
        ctx.strokeStyle = css("--limit"); ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
      }
      if (plot[`${side}_curve`]) line(decode(plot[`${side}_curve`]), css("--limit"), 1.5, [6, 4]);
    }
    ctx.setLineDash([]);
    ctx.restore();
    const items = [["box", css("--view"), `${s.name}${unitName ? ` (${unitName})` : ""}: range and 5-95 %`],
      ["", css("--trace-1"), "median"]];
    if (traces.some((t) => t.v === 2)) items.push(["", css("--fail"), "failing runs"]);
    if (traces.some((t) => t.v !== 2)) items.push(["", css("--trace-2"), "closest runs"]);
    if (plot.upper || plot.lower || plot.upper_curve || plot.lower_curve) items.push(["dash", css("--limit"), "limit"]);
    legend($(".legend", el), items);
    this.traces = traces;
    this.bands = bands;
    this.unit = s.unit;
  },
  bind() {
    const el = $(".plot.overlay"), canvas = $("canvas", el), box = $(".readout", el);
    canvas.addEventListener("pointermove", (e) => {
      const f = this.frame;
      if (!f || e.offsetX < f.L || e.offsetX > f.R) { box.hidden = true; return; }
      const x = f.xlo + ((e.offsetX - f.L) / (f.R - f.L)) * (f.xhi - f.xlo);
      const k = Math.min(M.bins - 1, Math.max(0, Math.floor((x - M.span[0]) / (M.span[1] - M.span[0]) * M.bins)));
      const b = this.bands;
      const lines = [M.t0_event ? `T+${x.toFixed(2)} s` : `${x.toFixed(2)} s`,
        `median ${withUnit(b.p50[k], this.unit)}`, `range ${withUnit(b.min[k], this.unit)} to ${withUnit(b.max[k], this.unit)}`];
      for (const t of this.traces.slice(0, 4)) {
        const text = t.lo[k] === t.hi[k] ? withUnit(t.lo[k], this.unit)
          : `${withUnit(t.lo[k], this.unit)} to ${withUnit(t.hi[k], this.unit)}`;
        lines.push(`${M.runs[t.run].id}: ${text}`);
      }
      readout(box, e.offsetX, lines, canvas.clientWidth);
    });
    canvas.addEventListener("pointerleave", () => { box.hidden = true; });
  },
};

// ----- scatter ------------------------------------------------------------------------------------------

const scatter = {
  points: [],
  draw() {
    const el = $(".plot.scatter");
    const canvas = $("canvas", el);
    const req = M.reqs[state.req];
    const name = state.param;
    const numeric = !name || (M.params.find((p) => p.name === name) || {}).numeric;
    const cats = [];
    const xOf = (i, r) => {
      if (!name) return r;
      const v = M.runs[i].params[name];
      if (numeric) return typeof v === "number" ? v : NaN;
      const key = v === undefined || v === null ? "" : String(v);
      let k = cats.indexOf(key);
      if (k < 0) { cats.push(key); k = cats.length - 1; }
      return k;
    };
    const order = name ? state.visible : state.visible.slice().sort((a, b) => a - b);
    const pts = [];
    order.forEach((i, r) => {
      const y = marginOf(req, i);
      const x = xOf(i, name ? 0 : i);
      if (Number.isFinite(x) && Number.isFinite(y)) pts.push({ i, x, y, code: codes[i * M.reqs.length + state.req] });
    });
    let xlo = Infinity, xhi = -Infinity, ylo = 0, yhi = 0;
    for (const p of pts) { xlo = Math.min(xlo, p.x); xhi = Math.max(xhi, p.x); ylo = Math.min(ylo, p.y); yhi = Math.max(yhi, p.y); }
    if (!Number.isFinite(xlo)) { xlo = 0; xhi = 1; }
    const xpad = (xhi - xlo) * 0.04 || 0.5, ypad = (yhi - ylo) * 0.08 || 1;
    const f = frameFor(canvas, xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad);
    const [factor] = state.pct ? [1] : unitScale(ylo, yhi, req.unit);
    axes(f, name || "run", null, factor);
    const { ctx, sx, sy, L, R, T, B } = f;
    if (!numeric && cats.length) {
      ctx.fillStyle = css("--bg"); ctx.fillRect(L, B + 1, R - L, 14);
      ctx.fillStyle = css("--axis"); ctx.textAlign = "center"; ctx.textBaseline = "top";
      cats.forEach((c, k) => ctx.fillText(c.length > 14 ? c.slice(0, 13) + ELLIPSIS : c || "(none)", sx(k), B + 6));
    }
    ctx.strokeStyle = css("--limit"); ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(L, Math.round(sy(0)) + 0.5); ctx.lineTo(R, Math.round(sy(0)) + 0.5); ctx.stroke();
    ctx.setLineDash([]);
    const colors = COLOR.map(css);
    this.points = pts.map((p) => {
      const jitter = !numeric ? (((p.i * 2654435761) % 1000) / 1000 - 0.5) * 0.5 : 0;
      return { ...p, px: sx(p.x + jitter), py: sy(p.y) };
    });
    for (const p of this.points) {
      ctx.fillStyle = colors[p.code];
      ctx.globalAlpha = p.code === 0 ? 0.55 : 0.95;
      ctx.beginPath(); ctx.arc(p.px, p.py, p.i === state.run ? 5 : 3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const selected = this.points.find((p) => p.i === state.run);
    if (selected) {
      ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(selected.px, selected.py, 7, 0, Math.PI * 2); ctx.stroke();
    }
    this.frame = f;
    legend($(".legend", el), [["box", colors[0], "pass"], ["box", colors[1], "warn"], ["box", colors[2], "fail"],
      ["dash", css("--limit"), `margin 0 (the limit)${state.pct ? "" : req.unit ? ` in ${unitScale(ylo, yhi, req.unit)[1]}` : ""}`]]);
  },
  nearest(e) {
    let best = null, dist = 64;
    for (const p of this.points) {
      const d = (p.px - e.offsetX) ** 2 + (p.py - e.offsetY) ** 2;
      if (d < dist) { dist = d; best = p; }
    }
    return best;
  },
  bind() {
    const el = $(".plot.scatter"), canvas = $("canvas", el), box = $(".readout", el);
    canvas.addEventListener("pointermove", (e) => {
      const p = this.nearest(e);
      if (!p) { box.hidden = true; return; }
      const run = M.runs[p.i];
      const req = M.reqs[state.req];
      const lines = [run.id, `margin ${marginText(req, p.i)}`];
      if (state.param) lines.push(`${state.param} = ${run.params[state.param]}`);
      readout(box, e.offsetX, lines, canvas.clientWidth);
      box.style.top = `${Math.max(0, e.offsetY - 60)}px`;
    });
    canvas.addEventListener("pointerleave", () => { box.hidden = true; });
    canvas.addEventListener("click", (e) => { const p = this.nearest(e); if (p) openRun(p.i); });
  },
};

// ----- runs table ---------------------------------------------------------------------------------------

const runTable = {
  rowHeight: 28,
  columns: [],
  init() {
    const head = $("#run-table .vhead");
    this.columns = [
      { key: "id", title: "Run", value: (r) => r.id },
      { key: "status", title: "Status", value: (r) => RANK[VERDICTS.indexOf(r.status)] },
      { key: "fail", title: "Fail", num: true, value: (r) => r.counts[2] },
      { key: "warn", title: "Warn", num: true, value: (r) => r.counts[1] },
      { key: "error", title: "Error", num: true, value: (r) => r.counts[4] },
      ...M.params.map((p) => ({ key: `p:${p.name}`, title: p.name, num: p.numeric, value: (r) => r.params[p.name] })),
    ];
    const template = `minmax(9em, 2fr) 5.5em repeat(3, 4em)${M.params.length ? ` repeat(${M.params.length}, minmax(6em, 1fr))` : ""}`;
    $("#run-table").style.setProperty("--run-columns", template);
    head.replaceChildren(...this.columns.map((col) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = col.title;
      if (col.num) b.className = "num";
      b.addEventListener("click", () => {
        const dir = state.sort && state.sort.key === col.key && state.sort.dir === "asc" ? "desc" : "asc";
        state.sort = { key: col.key, dir };
        $$("#run-table .vhead button").forEach((x) => x.removeAttribute("data-dir"));
        b.dataset.dir = dir;
        this.refresh();
      });
      return b;
    }));
    $("#run-table .vbody").addEventListener("scroll", () => this.paint());
  },
  refresh() {
    let rows = state.visible.slice();
    if (state.sort) {
      const col = this.columns.find((c) => c.key === state.sort.key);
      const sign = state.sort.dir === "asc" ? 1 : -1;
      rows.sort((a, b) => {
        const va = col.value(M.runs[a]), vb = col.value(M.runs[b]);
        if (va === vb) return a - b;
        if (va === undefined || va === null || va === "") return 1;
        if (vb === undefined || vb === null || vb === "") return -1;
        return (typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb), undefined, { numeric: true })) * sign;
      });
    }
    this.rows = rows;
    $("#run-table .vspace").style.height = `${rows.length * this.rowHeight}px`;
    this.paint();
  },
  paint() {
    const body = $("#run-table .vbody");
    const space = $("#run-table .vspace");
    const first = Math.max(0, Math.floor(body.scrollTop / this.rowHeight) - 5);
    const last = Math.min(this.rows.length, first + Math.ceil(body.clientHeight / this.rowHeight) + 10);
    const out = [];
    for (let k = first; k < last; k++) {
      const i = this.rows[k];
      const run = M.runs[i];
      const row = document.createElement("div");
      row.className = "vrow";
      row.style.top = `${k * this.rowHeight}px`;
      const link = document.createElement("a");
      link.href = run.page || run.file;
      link.textContent = run.id;
      if (run.error) link.title = run.error;
      const status = document.createElement("span");
      status.append(badge(run.status));
      row.append(link, status);
      for (const n of [run.counts[2], run.counts[1], run.counts[4]]) {
        const cell = document.createElement("span"); cell.className = "num"; cell.textContent = String(n); row.append(cell);
      }
      for (const p of M.params) {
        const cell = document.createElement("span");
        const v = run.params[p.name];
        cell.textContent = v === undefined || v === null ? "" : (typeof v === "number" ? sig(v, 6) : String(v));
        if (p.numeric) cell.className = "num";
        row.append(cell);
      }
      out.push(row);
    }
    space.replaceChildren(...out);
  },
};

// ----- page ---------------------------------------------------------------------------------------------

function drawAll() {
  matrix.draw();
  overlay.draw();
  scatter.draw();
}

async function start() {
  $$("table.reqs tr.pick").forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    select(Number(tr.dataset.req));
    $("#details").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  if (typeof DecompressionStream === "undefined") {
    const p = document.createElement("p");
    p.className = "banner";
    p.textContent = "This browser cannot unpack the dashboard data; results.xlsx has everything.";
    $("main").prepend(p);
    return;
  }
  buffer = await inflate($("#blob").textContent);
  codes = decode(M.matrix);
  $("#req-select").addEventListener("change", (e) => select(Number(e.target.value)));
  $("#param-select").addEventListener("change", (e) => { state.param = e.target.value; scatter.draw(); });
  $("#pct").addEventListener("change", (e) => { state.pct = e.target.checked; drawAll(); });
  let timer = 0;
  $("#run-filter").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(applyFilter, 120); });
  const numeric = M.params.find((p) => p.numeric);
  if (numeric) { state.param = numeric.name; $("#param-select").value = numeric.name; }
  const first = M.reqs.findIndex((r) => r.verdict === "fail");
  state.req = first >= 0 ? first : 0;
  matrix.bind(); overlay.bind(); scatter.bind(); runTable.init();
  let pending = 0;
  new ResizeObserver(() => { cancelAnimationFrame(pending); pending = requestAnimationFrame(drawAll); }).observe($("main"));
  applyFilter();
  select(state.req);
  if (selftest) await runSelftest();
}

async function runSelftest() {
  const digests = [];
  for (const ref of allRefs()) {
    const size = ref.n * ({ f4: 4, u2: 2 }[ref.k] || 1);
    const hash = await crypto.subtle.digest("SHA-256", new Uint8Array(buffer, ref.o, size));
    digests.push(Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join(""));
  }
  let drawn = 0;
  for (let k = 0; k < M.reqs.length; k++) {
    try { select(k, 0); drawn += 1; } catch (err) { errors.push(`${M.reqs[k].id}: ${err}`); }
  }
  const filters = ["status=fail", "nothing-matches-this", ""];
  for (const text of filters) { $("#run-filter").value = text; applyFilter(); }
  const out = $("#selftest");
  out.textContent = JSON.stringify({ digests, reqs: M.reqs.length, runs: M.runs.length, drawn,
    rows: $$("#run-table .vrow").length, ms: Math.round(performance.now() - started), errors });
  out.dataset.done = "1";
}

start().catch((err) => {
  errors.push(String(err && err.stack || err));
  if (selftest) {
    const out = $("#selftest");
    out.textContent = JSON.stringify({ digests: [], drawn: 0, errors });
    out.dataset.done = "1";
  }
});
})();
