(() => {
"use strict";
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
const manifest = JSON.parse($("#manifest").textContent);
const selftest = new URLSearchParams(location.search).has("selftest");
const started = performance.now();
const errors = [];
const state = { view: null, link: true };
const plots = [];
let buffer = null;
let timeline = null;

window.addEventListener("error", (e) => errors.push(String(e.message || e)));

function banner(text) {
  const p = document.createElement("p");
  p.className = "banner";
  p.textContent = text;
  $("main").prepend(p);
}

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
  if (ref.k === "f4") {
    out = new Float32Array(buffer, ref.o, ref.n);
  } else {
    const codes = new Uint16Array(buffer, ref.o, ref.n);
    const step = (ref.hi - ref.lo) / 65534;
    out = new Float64Array(ref.n);
    for (let i = 0; i < ref.n; i++) out[i] = codes[i] === 65535 ? NaN : ref.lo + codes[i] * step;
  }
  decoded.set(ref, out);
  return out;
}

function allRefs() {
  const refs = [];
  const walk = (value) => {
    if (Array.isArray(value)) value.forEach(walk);
    else if (value && typeof value === "object") {
      if (typeof value.o === "number" && typeof value.k === "string") refs.push(value);
      else Object.values(value).forEach(walk);
    }
  };
  walk(manifest.series);
  walk(manifest.plots);
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
  if (!unit) return sig(v);
  for (const [name, factor] of PREFIXED[unit] || []) {
    if (Math.abs(v) >= factor) return `${sig(v / factor)} ${name}`;
  }
  return `${sig(v)} ${unit}`;
}

function axisScale(lo, hi, unit) {
  const top = Math.max(Math.abs(lo), Math.abs(hi));
  for (const [name, factor] of PREFIXED[unit] || []) if (top >= factor) return [factor, name];
  return [1, unit || ""];
}

function fmtTime(s) {
  if (!Number.isFinite(s)) return "-";
  if (manifest.t0 !== null && manifest.t0 !== undefined) return `T${s >= 0 ? "+" : "-"}${Math.abs(s).toFixed(3)} s`;
  return `${(s + manifest.base).toFixed(3)} s`;
}

function ticks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

function tickText(v, step) {
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}

function upper(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// overview points with the [t, v] pieces (sorted, not overlapping) put in place of the points they cover
function splice(t, v, pieces) {
  const used = pieces.filter(([pt, pv]) => pv && pt.length);
  if (!used.length) return [t, v];
  const outT = [], outV = [];
  let i = 0;
  for (const [pt, pv] of used) {
    const a = lower(t, pt[0]);
    for (; i < a; i++) { outT.push(t[i]); outV.push(v[i]); }
    for (let j = 0; j < pt.length; j++) { outT.push(pt[j]); outV.push(pv[j]); }
    i = Math.max(i, upper(t, pt[pt.length - 1]));
  }
  for (; i < t.length; i++) { outT.push(t[i]); outV.push(v[i]); }
  return [Float64Array.from(outT), Float64Array.from(outV)];
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function lower(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// ----- plots --------------------------------------------------------------------------------------------

const PAD = { left: 62, right: 14, top: 12, bottom: 26 };
const LABEL_CHARS = 16;

// one left margin for every plot, wide enough for state names, so time axes line up
function leftPad() {
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = "11px system-ui, sans-serif";
  let widest = 0;
  for (const s of manifest.series) {
    for (const [, name] of s.labels || []) widest = Math.max(widest, probe.measureText(name.slice(0, LABEL_CHARS)).width);
  }
  for (const p of manifest.phases) widest = Math.max(widest, probe.measureText(p.name).width);
  return Math.min(Math.max(PAD.left, Math.ceil(widest) + 14), 170);
}
let hatch = null;

function hatchPattern(ctx) {
  if (hatch && hatch.color === css("--limit")) return hatch.pattern;
  const tile = document.createElement("canvas");
  tile.width = tile.height = 8;
  const g = tile.getContext("2d");
  g.strokeStyle = css("--limit");
  g.globalAlpha = 0.45;
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, 8); g.lineTo(8, 0);
  g.stroke();
  hatch = { color: css("--limit"), pattern: ctx.createPattern(tile, "repeat") };
  return hatch.pattern;
}

class Plot {
  constructor(el, spec, id) {
    this.el = el;
    this.id = id;
    this.spec = spec;
    this.canvas = $("canvas", el);
    this.readout = $(".readout", el);
    this.local = null;
    this.visible = false;
    this.dirty = true;
    this.drawn = 0;
    this.hover = null;
    this.drag = null;
    this.data = null;
    this.bind();
  }

  get view() { return state.link ? state.view : this.local; }

  setView(view) {
    if (state.link) { state.view = view; redrawAll(); } else { this.local = view; this.draw(); }
  }

  load() {
    if (this.data) return this.data;
    const spec = this.spec;
    const colors = ["--trace-1", "--trace-2", "--trace-3", "--trace-4"];
    const series = spec.series.map((index, k) => {
      const s = manifest.series[index];
      return { name: s.name, unit: s.unit, discrete: s.discrete, labels: s.labels, t: decode(s.t), v: decode(s.v),
        color: colors[k % colors.length], k };
    });
    const limits = spec.limits.map((l) => ({ side: l.side, step: l.step, t: decode(l.t), v: decode(l.v) }));
    // full-resolution windows replace the overview points they cover
    const details = spec.details.map((d) => ({ t: decode(d.t), v: d.v.map(decode), limits: d.limits }))
      .filter((d) => d.t.length).sort((a, b) => a.t[0] - b.t[0]);
    for (const s of series) {
      [s.t, s.v] = splice(s.t, s.v, details.map((d) => [d.t, d.v[s.k] || null]));
    }
    for (const l of limits) {
      [l.t, l.v] = splice(l.t, l.v, details.map((d) => {
        const hit = d.limits.find((x) => x.side === l.side);
        return [d.t, hit ? decode(hit.v) : null];
      }));
    }
    this.data = { series, limits };
    this.legend();
    return this.data;
  }

  legend() {
    const box = $(".legend", this.el);
    if (!box) return;
    const items = [];
    const add = (cls, color, text) => {
      const span = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.className = cls;
      if (cls.includes("dash")) swatch.style.color = color; else swatch.style.background = color;
      span.append(swatch, document.createTextNode(text));
      items.push(span);
    };
    for (const s of this.data.series) add("", css(s.color), s.name + (s.unit ? ` (${s.unit})` : ""));
    if (this.spec.limits.length || this.spec.lines.some((l) => l.kind === "upper" || l.kind === "lower")) {
      add("dash", css("--limit"), "limit");
    }
    if (this.spec.lines.some((l) => l.kind === "threshold")) add("dash", css("--threshold"), "threshold");
    if (this.spec.spans.some((s) => s.held)) add("box", css("--held"), "condition holds");
    if (this.spec.spans.some((s) => !s.held && !s.tol)) add("box", css("--span"), "violation");
    if (this.spec.spans.some((s) => s.tol)) add("box", css("--span"), "tolerated");
    if (this.spec.windows.length) add("box", css("--inactive"), "not checked");
    if (this.spec.band) add("box", css("--band"), "allowed time");
    box.replaceChildren(...items);
  }

  bind() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.drag = { x0: e.offsetX, x1: e.offsetX };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      this.hover = e.offsetX;
      if (this.drag) this.drag.x1 = e.offsetX;
      this.schedule();
    });
    c.addEventListener("pointerup", (e) => {
      const drag = this.drag;
      this.drag = null;
      if (drag && Math.abs(drag.x1 - drag.x0) > 4 && this.frame) {
        const a = this.frame.invX(Math.min(drag.x0, drag.x1));
        const b = this.frame.invX(Math.max(drag.x0, drag.x1));
        this.setView([a, b]);
      } else this.schedule();
      c.releasePointerCapture(e.pointerId);
    });
    c.addEventListener("pointerleave", () => { this.hover = null; this.schedule(); });
    c.addEventListener("dblclick", () => this.setView(null));
  }

  schedule() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => { this.pending = 0; this.draw(); });
  }

  xRange() {
    return this.view || manifest.span;
  }

  draw() {
    if (!buffer) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 10) return;
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px system-ui, sans-serif";
    this.paint(ctx, w, h);
    this.dirty = false;
    this.drawn += 1;
  }

  paint(ctx, w, h) {
    const spec = this.spec;
    const [x0, x1] = this.xRange();
    const { series, limits } = this.load();
    // y range from what is visible
    let lo = Infinity, hi = -Infinity;
    const take = (v) => { if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } };
    const scan = (t, v) => {
      const a = Math.max(0, lower(t, x0) - 1), b = Math.min(t.length, lower(t, x1) + 1);
      for (let i = a; i < b; i++) take(v[i]);
    };
    for (const s of series) scan(s.t, s.v);
    for (const l of limits) scan(l.t, l.v);
    for (const line of spec.lines) take(line.v);
    if (spec.marker && spec.marker.t >= x0 && spec.marker.t <= x1) take(spec.marker.v);
    const labels = series.length && series[0].labels;
    if (labels) for (const [code] of labels) take(code);
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12 * Math.max(1, Math.abs(hi))) { const d = Math.abs(hi) * 0.05 || 1; lo -= d; hi += d; }
    const pad = labels ? 0.5 : (hi - lo) * 0.06;
    lo -= pad; hi += pad;
    const L = Math.min(PAD.left, w * 0.35), R = w - PAD.right, T = PAD.top, B = h - PAD.bottom;
    const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (R - L);
    const sy = (y) => B - ((y - lo) / (hi - lo)) * (B - T);
    this.frame = { invX: (px) => x0 + ((px - L) / (R - L)) * (x1 - x0), sx, x0, x1, L, R, T, B };

    // grid and axes
    const unit = series.length ? series[0].unit : spec.unit;
    const [factor, shown] = axisScale(lo, hi, unit);
    ctx.strokeStyle = css("--grid");
    ctx.fillStyle = css("--axis");
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yt = labels ? labels.map(([code]) => code).filter((c) => c >= lo && c <= hi) : ticks(lo / factor, hi / factor, Math.max(2, Math.floor((B - T) / 42))).map((v) => v * factor);
    const ystep = yt.length > 1 ? (yt[1] - yt[0]) / factor : 1;
    for (const v of yt) {
      const y = Math.round(sy(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
      let text = labels ? String((labels.find(([code]) => code === v) || [0, v])[1]) : tickText(v / factor, ystep);
      if (text.length > LABEL_CHARS) text = text.slice(0, LABEL_CHARS - 1) + "\u2026";
      ctx.fillText(text, L - 6, y);
    }
    if (shown && !labels) {
      ctx.save(); ctx.translate(11, (T + B) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
      ctx.fillText(shown, 0, 0); ctx.restore();
    }
    const xt = ticks(x0, x1, Math.max(2, Math.floor((R - L) / 90)));
    const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of xt) {
      const x = Math.round(sx(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
      ctx.fillText(tickText(v, xstep), x, B + 6);
    }
    ctx.textAlign = "right";
    ctx.fillText(manifest.t0 !== null && manifest.t0 !== undefined ? `s after ${manifest.t0_event}` : "s", R, B + 6 + 11);

    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, R - L, B - T); ctx.clip();
    // where the check does not look
    if (spec.windows.length) {
      ctx.fillStyle = css("--inactive");
      let from = x0;
      for (const [a, b] of spec.windows) {
        if (a > from) ctx.fillRect(sx(from), T, sx(a) - sx(from), B - T);
        from = Math.max(from, b);
      }
      if (from < x1) ctx.fillRect(sx(from), T, sx(x1) - sx(from), B - T);
    }
    if (spec.band) {
      const a = spec.band[0] ?? x0 - (x1 - x0), b = spec.band[1] ?? x1 + (x1 - x0);
      ctx.fillStyle = css("--band");
      ctx.fillRect(sx(a), T, sx(b) - sx(a), B - T);
    }
    for (const span of spec.spans) {
      if (span.e < x0 || span.s > x1) continue;
      const a = sx(span.s), width = Math.max(sx(span.e) - a, 1.5);
      ctx.fillStyle = span.held ? css("--held") : (span.tol ? hatchPattern(ctx) : css("--span"));
      ctx.fillRect(a, T, width, B - T);
    }
    // events
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = css("--event");
    ctx.fillStyle = css("--axis");
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    let lastLabel = -Infinity;
    for (const ev of manifest.events) {
      for (const tv of ev.times.slice(0, 50)) {
        if (tv === null || tv < x0 || tv > x1) continue;
        const x = Math.round(sx(tv)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
        if (x - lastLabel > 60) { ctx.fillText(ev.name, x + 3, T + 2); lastLabel = x; }
      }
    }
    // limits and lines
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    for (const l of limits) {
      ctx.strokeStyle = css("--limit");
      this.path(ctx, l.t, l.v, sx, sy, x0, x1, l.step);
    }
    for (const line of spec.lines) {
      if (line.kind === "value") { ctx.setLineDash([2, 3]); ctx.strokeStyle = css(series[0] ? series[0].color : "--trace-1"); }
      else { ctx.setLineDash([6, 4]); ctx.strokeStyle = css(line.kind === "threshold" ? "--threshold" : "--limit"); }
      const y = Math.round(sy(line.v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
    }
    // traces
    ctx.setLineDash([]);
    ctx.lineWidth = 1.4;
    for (const s of series) {
      ctx.strokeStyle = css(s.color);
      this.path(ctx, s.t, s.v, sx, sy, x0, x1, s.discrete);
    }
    // worst point
    const m = spec.marker;
    if (m && m.t !== null && m.v !== null && m.t >= x0 && m.t <= x1) {
      ctx.fillStyle = css("--limit");
      ctx.strokeStyle = css("--bg");
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx(m.t), sy(m.v), 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    // drag selection and hover
    if (this.drag) {
      ctx.fillStyle = css("--view");
      ctx.fillRect(Math.min(this.drag.x0, this.drag.x1), T, Math.abs(this.drag.x1 - this.drag.x0), B - T);
    }
    ctx.restore();
    this.showReadout(ctx, series, sx, sy, L, R, T, B);
  }

  // a polyline that breaks at NaN; a step line holds each value until the next point
  path(ctx, t, v, sx, sy, x0, x1, step) {
    const a = Math.max(0, lower(t, x0) - 1), b = Math.min(t.length, lower(t, x1) + 2);
    ctx.beginPath();
    let open = false;
    for (let i = a; i < b; i++) {
      const y = v[i];
      if (!Number.isFinite(y)) { open = false; continue; }
      const px = sx(t[i]), py = sy(y);
      if (open) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      open = true;
      if (step && i + 1 < t.length) ctx.lineTo(sx(t[i + 1]), py);
    }
    ctx.stroke();
  }

  showReadout(ctx, series, sx, sy, L, R, T, B) {
    const box = this.readout;
    if (this.hover === null || this.hover < L || this.hover > R || !box) {
      if (box) box.hidden = true;
      return;
    }
    const tx = this.frame.invX(this.hover);
    ctx.strokeStyle = css("--axis");
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(this.hover + 0.5, T); ctx.lineTo(this.hover + 0.5, B); ctx.stroke();
    const lines = [fmtTime(tx)];
    for (const s of series) {
      let i = lower(s.t, tx);
      if (s.discrete) i = Math.max(0, s.t[i] === tx ? i : i - 1);
      else if (i > 0 && (i >= s.t.length || tx - s.t[i - 1] < s.t[i] - tx)) i -= 1;
      if (i < 0 || i >= s.t.length) continue;
      const v = s.v[i];
      let text = withUnit(v, s.unit);
      if (s.labels && Number.isFinite(v)) {
        const hit = s.labels.find(([code]) => code === Math.round(v));
        if (hit) text = hit[1];
      }
      lines.push(`${s.name}: ${text}`);
      if (Number.isFinite(v)) {
        ctx.fillStyle = css(s.color);
        ctx.beginPath(); ctx.arc(sx(s.t[i]), sy(v), 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    box.replaceChildren(...lines.map((line) => { const d = document.createElement("div"); d.textContent = line; return d; }));
    box.hidden = false;
    const width = box.offsetWidth;
    box.style.left = `${this.hover + 12 + width > R ? this.hover - 12 - width : this.hover + 12}px`;
  }
}

// ----- timeline -----------------------------------------------------------------------------------------

class Timeline extends Plot {
  constructor(el) {
    super(el, { series: [], limits: [], lines: [], spans: [], windows: [], details: [] }, "timeline");
    this.rows = manifest.phases.filter((p) => p.spans.length);
    this.canvas.style.height = `${46 + 18 * this.rows.length}px`;
  }

  get view() { return null; }

  setView(view) { state.view = view; if (!state.link) plots.forEach((p) => { p.local = view; }); redrawAll(); }

  load() { return { series: [], limits: [] }; }

  paint(ctx, w, h) {
    const [x0, x1] = manifest.span;
    const L = Math.min(PAD.left, w * 0.3), R = w - PAD.right, T = 6, B = h - PAD.bottom;
    const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (R - L);
    this.frame = { invX: (px) => x0 + ((px - L) / (R - L)) * (x1 - x0), sx, L, R, T, B };
    ctx.fillStyle = css("--axis");
    ctx.strokeStyle = css("--grid");
    ctx.textBaseline = "middle";
    this.rows.forEach((row, k) => {
      const y = T + 22 + k * 18;
      ctx.textAlign = "right";
      ctx.fillStyle = css("--axis");
      ctx.fillText(row.name, L - 6, y + 6);
      ctx.fillStyle = css("--phase");
      for (const [a, b] of row.spans) ctx.fillRect(sx(a), y, Math.max(sx(b) - sx(a), 1), 12);
    });
    const xt = ticks(x0, x1, Math.max(2, Math.floor((R - L) / 90)));
    const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = css("--axis");
    for (const v of xt) {
      const x = Math.round(sx(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
      ctx.fillText(tickText(v, xstep), x, B + 6);
    }
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = css("--event");
    ctx.textAlign = "left";
    let lastLabel = -Infinity;
    for (const ev of manifest.events) {
      for (const tv of ev.times.slice(0, 50)) {
        if (tv === null) continue;
        const x = Math.round(sx(tv)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
        if (x - lastLabel > 50) { ctx.fillText(ev.name, x + 3, T); lastLabel = x; }
      }
    }
    ctx.setLineDash([]);
    ctx.fillStyle = css("--span");
    for (const row of $$(".wrong tr.jump[data-s]")) {
      const a = Number(row.dataset.s), b = Number(row.dataset.e);
      ctx.fillRect(sx(a) - 1, B - 6, Math.max(sx(b) - sx(a), 2) + 2, 6);
    }
    if (state.view) {
      ctx.fillStyle = css("--view");
      ctx.fillRect(sx(state.view[0]), T, Math.max(sx(state.view[1]) - sx(state.view[0]), 2), B - T);
    }
    if (this.drag) {
      ctx.fillStyle = css("--view");
      ctx.fillRect(Math.min(this.drag.x0, this.drag.x1), T, Math.abs(this.drag.x1 - this.drag.x0), B - T);
    }
    if (this.hover !== null && this.hover >= L && this.hover <= R && this.readout) {
      this.readout.textContent = fmtTime(this.frame.invX(this.hover));
      this.readout.hidden = false;
      this.readout.style.left = `${Math.min(this.hover + 12, R - this.readout.offsetWidth)}px`;
    } else if (this.readout) this.readout.hidden = true;
  }
}

function redrawAll() {
  for (const p of plots) {
    p.dirty = true;
    if (p.visible) p.schedule();
  }
  if (timeline) timeline.schedule();
}

// ----- page ---------------------------------------------------------------------------------------------

function zoomTo(s, e) {
  const span = manifest.span[1] - manifest.span[0];
  const pad = Math.max((e - s) * 1.5, span * 0.01, 0.05);
  const view = [s - pad, e + pad];
  state.view = view;
  if (!state.link) plots.forEach((p) => { p.local = view; });
  redrawAll();
}

function setupList() {
  const cards = $$(".card");
  const list = $(".list");
  const nomatch = $("#nomatch");
  const shown = new Set(["fail", "error", "warn", "pass", "not_applicable"]);
  const filter = $("#filter");
  const apply = () => {
    const words = (filter.value || "").toLowerCase().split(/\s+/).filter(Boolean);
    let visible = 0;
    for (const card of cards) {
      const text = card.textContent.toLowerCase();
      const ok = shown.has(card.dataset.verdict) && words.every((w) => text.includes(w));
      card.hidden = !ok;
      if (ok) visible += 1;
    }
    nomatch.hidden = visible > 0;
  };
  filter.addEventListener("input", apply);
  for (const chip of $$(".chip[data-filter]")) {
    chip.addEventListener("click", () => {
      const verdict = chip.dataset.filter;
      if (shown.has(verdict)) shown.delete(verdict); else shown.add(verdict);
      chip.setAttribute("aria-pressed", String(shown.has(verdict)));
      apply();
    });
  }
  for (const button of $$(".list-head button[data-sort]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      const dir = button.dataset.dir === "asc" ? "desc" : "asc";
      $$(".list-head button").forEach((b) => b.removeAttribute("data-dir"));
      button.dataset.dir = dir;
      const value = (card) => {
        const raw = card.dataset[key];
        return raw === "" || raw === undefined ? null : Number(raw);
      };
      const sorted = cards.slice().sort((a, b) => {
        const va = value(a), vb = value(b);
        if (va === null && vb === null) return Number(a.dataset.order) - Number(b.dataset.order);
        if (va === null) return 1;
        if (vb === null) return -1;
        return (dir === "asc" ? va - vb : vb - va) || Number(a.dataset.order) - Number(b.dataset.order);
      });
      for (const card of sorted) list.insertBefore(card, nomatch);
    });
  }
  const expand = $("#expand");
  expand.addEventListener("click", () => {
    const open = expand.textContent === "Open all";
    for (const card of cards) if (!card.hidden) card.open = open;
    expand.textContent = open ? "Close all" : "Open all";
  });
  $("#link").addEventListener("change", (e) => {
    state.link = e.target.checked;
    if (state.link) plots.forEach((p) => { p.local = null; });
    redrawAll();
  });
  $("#reset").addEventListener("click", () => {
    state.view = null;
    plots.forEach((p) => { p.local = null; });
    redrawAll();
  });
  for (const row of $$(".wrong tr.jump")) {
    row.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const card = document.getElementById(`req-${row.dataset.req}`);
      if (row.dataset.s !== undefined) zoomTo(Number(row.dataset.s), Number(row.dataset.e));
      if (card) { card.hidden = false; card.open = true; card.scrollIntoView({ behavior: "smooth", block: "start" }); }
    });
  }
}

function initialView() {
  const wanted = new URLSearchParams(location.search).get("t");
  if (!wanted) return;
  const [a, b] = wanted.split(",").map(Number);
  if (Number.isFinite(a) && Number.isFinite(b) && b > a) state.view = [a, b];
}

async function start() {
  PAD.left = leftPad();
  initialView();
  setupList();
  const blob = $("#blob");
  if (!blob) return;
  if (typeof DecompressionStream === "undefined") {
    banner("This browser cannot unpack the plot data; the tables are complete without it.");
    return;
  }
  try {
    buffer = await inflate(blob.textContent);
  } catch (err) {
    banner(`The plot data could not be read: ${err}`);
    errors.push(String(err));
    return;
  }
  const seen = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const plot = entry.target.plot;
      plot.visible = entry.isIntersecting;
      if (plot.visible && plot.dirty) plot.draw();
    }
  }, { rootMargin: "200px" });
  const resized = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const plot = entry.target.plot;
      plot.dirty = true;
      if (plot.visible || plot === timeline) plot.schedule();
    }
  });
  for (const el of $$(".plot[data-req]")) {
    const spec = manifest.plots[el.dataset.req];
    if (!spec) continue;
    const plot = new Plot(el, spec, el.dataset.req);
    el.plot = plot;
    plots.push(plot);
    seen.observe(el);
    resized.observe(el);
  }
  const height = window.innerHeight || document.documentElement.clientHeight;
  for (const plot of plots) {
    const rect = plot.el.getBoundingClientRect();
    if (rect.width > 0 && rect.bottom > -200 && rect.top < height + 200) {
      plot.visible = true;
      plot.draw();
    }
  }
  const tl = $(".plot.timeline");
  if (tl) {
    timeline = new Timeline(tl);
    tl.plot = timeline;
    resized.observe(tl);
    timeline.draw();
  }
  if (selftest) await runSelftest();
}

async function runSelftest() {
  const digests = [];
  for (const ref of allRefs()) {
    const size = ref.n * (ref.k === "f4" ? 4 : 2);
    const hash = await crypto.subtle.digest("SHA-256", new Uint8Array(buffer, ref.o, size));
    digests.push(Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join(""));
  }
  for (const card of $$(".card")) card.open = true;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  let drawn = 0;
  for (const plot of plots) {
    try { plot.draw(); if (plot.drawn) drawn += 1; } catch (err) { errors.push(`${plot.id}: ${err}`); }
  }
  const out = $("#selftest");
  out.textContent = JSON.stringify({ digests, plots: plots.length, drawn, series: manifest.series.length,
    ms: Math.round(performance.now() - started), errors });
  out.dataset.done = "1";
}

start().catch((err) => {
  errors.push(String(err && err.stack || err));
  banner(`The plots could not be drawn: ${err}`);
  if (selftest) {
    const out = $("#selftest");
    out.textContent = JSON.stringify({ digests: [], plots: plots.length, drawn: 0, errors });
    out.dataset.done = "1";
  }
});
})();
