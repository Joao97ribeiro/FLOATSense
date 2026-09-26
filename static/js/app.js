// FLOATSense project page: dataset explorer and leaderboard along the height.

// Paper palette (paper_style.py and fs_common.py).
const NAVY = "#294366", RED = "#b02c27", REF = "#b8b8b8", REF_DARK = "#8f8f8f", CYAN = "#7cc0cd", SCADA = "#4383ad",
  TAUPE = "#8f7a6e", SLATE = "#9aa3ab", CORAL = "#d06662", PINK = "#dd9c98", WINE = "#6f1b17", INK = "#333333";
const TOWERS = ["ref", "opt1", "opt2"];
const TOWER_NAME = {ref: "REF", opt1: "OPT1", opt2: "OPT2"};
const TOWER_COLOR = {ref: REF, opt1: NAVY, opt2: RED};
const NAMES = {tcn: "TCN", prob_tcn: "Prob-TCN", fits: "FITS", dlinear: "DLinear", chronos: "Chronos", s4: "S4D", lstm: "LSTM",
  spectral: "Spectral gain", hybrid: "Hybrid", hybrid_tcn: "Hybrid-TCN", timesnet: "TimesNet", mamba: "Mamba", transformer: "PatchTST",
  physics: "Physics", naive: "Floor", fno: "FNO", unet: "U-Net", itransformer: "iTransformer", moment: "MOMENT", moment_ft: "MOMENT-ft",
  timesfm: "TimesFM", timesfm_ft: "TimesFM-ft"};
// Families of the model table of the paper.
const FAMILY = {naive: "floor", physics: "physics", dlinear: "convolutional", tcn: "convolutional", unet: "convolutional",
  prob_tcn: "convolutional", lstm: "recurrent", transformer: "attention", itransformer: "attention", s4: "state-space",
  mamba: "state-space", spectral: "spectral", fits: "spectral", fno: "spectral", timesnet: "period-folding",
  chronos: "pretrained", moment: "pretrained", moment_ft: "pretrained", timesfm: "pretrained", timesfm_ft: "pretrained",
  hybrid: "physics-anchored", hybrid_tcn: "physics-anchored"};
const FAMILY_COLOR = {convolutional: NAVY, recurrent: CYAN, attention: TAUPE, "state-space": SLATE, spectral: WINE,
  "period-folding": PINK, pretrained: CORAL, "physics-anchored": "#565656", physics: "#222222", floor: REF_DARK};
const MODEL_COLOR = {tcn: NAVY, prob_tcn: RED, lstm: CYAN, transformer: TAUPE, s4: SLATE, fits: WINE, timesnet: PINK,
  mamba: CYAN, naive: REF_DARK, physics: "#222222", spectral: CORAL, unet: "#565656", fno: "#6b6b6b",
  itransformer: "#aaaaaa", dlinear: "#d2d2d2", chronos: CORAL, hybrid: CORAL};
const MODEL_DASH = {mamba: "dash", naive: "dot", hybrid: "dash"};
const MOMENT_SCALE = [[0, NAVY], [0.5, "#f7f7f7"], [1, RED]];
// Height colors: base RED to top RED_LIGHT, the height gradient of the paper figures.
function heightColor(k) { const a = [176, 44, 39], b = [240, 179, 176], f = k / 10;
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(",")})`; }
function heightChips() {
  $("height-chips").innerHTML = GAUGE.map((g, k) => `<button type="button" class="chip" data-k="${k}" aria-pressed="${st.heights.includes(k)}"><span class="dot" style="background:${heightColor(k)}"></span>${g}</button>`).join("");
}
const INPUTS = [
  ["tower_top_afa_mod", "FA accel. corr. [m/s²]", NAVY, "FA accel., corrected (input)"], ["tower_top_afa", "FA accel. [m/s²]", NAVY, "FA accel., raw"],
  ["tower_top_ass_mod", "SS accel. corr. [m/s²]", "#5b7fa6", "SS accel., corrected"], ["tower_top_ass", "SS accel. [m/s²]", "#5b7fa6", "SS accel., raw"],
  ["wind_speed", "Wind [m/s]", SCADA, "Wind speed (input)"], ["rotor_speed", "Rotor [rpm]", SCADA, "Rotor speed (input)"],
  ["blade_pitch", "Pitch [deg]", SCADA, "Blade pitch (input)"], ["electrical_power", "Power [kW]", SCADA, "Generator power"],
  ["plat_surge", "Surge [m]", CYAN, "Platform surge"], ["plat_sway", "Sway [m]", CYAN, "Platform sway"], ["plat_heave", "Heave [m]", CYAN, "Platform heave"],
  ["plat_roll", "Roll [deg]", CYAN, "Platform roll"], ["plat_pitch", "Platform pitch [deg]", CYAN, "Platform pitch"], ["plat_yaw", "Yaw [deg]", CYAN, "Platform yaw"],
  ["wave_elev", "Wave [m]", CYAN, "Wave elevation"]];
function inputChips() {
  $("input-chips").innerHTML = INPUTS.map(([c, , col, label]) => `<button type="button" class="chip" data-c="${c}" aria-pressed="${st.inputs.includes(c)}"><span class="dot" style="background:${col}"></span>${label}</button>`).join("");
}
function setWindow(r) {
  let [a, b] = r.map(Number); if (!(a < b)) return;
  a = Math.max(400, a); b = Math.min(1000, b); st.xr = [a, b];
  $("tw-from").value = Math.round(a); $("tw-to").value = Math.round(b);
  Plotly.relayout("plot-inputs", {"xaxis.range": st.xr}); Plotly.relayout("plot-targets", {"xaxis.range": st.xr});
}
const GAUGE = ["Base", "Gauge 1", "Gauge 2", "Gauge 3", "Gauge 4", "Gauge 5", "Gauge 6", "Gauge 7", "Gauge 8", "Gauge 9", "Top"];
const GROUP_TAG = {"In-train": "tag-it", "Interpolate": "tag-ip", "Extrapolate": "tag-ex"};
const MET = [["r2", "R²"], ["within2", "×2"], ["median_ratio", "Ratio"], ["within_corr", "Within-corr"], ["mre", "MRE"]];

const LAYOUT = {margin: {l: 70, r: 20, t: 30, b: 50}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: {family: "Helvetica, Arial, sans-serif", size: 12, color: INK}, showlegend: false};
const CFG = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"]};
const AX = o => Object.assign({gridcolor: "#ececec", zerolinecolor: "#d0d0d0", linecolor: "#bdbdbd", tickfont: {size: 10}}, o);
const $ = id => document.getElementById(id);
const L = o => Object.assign({}, LAYOUT, o);

let D, S, time;
const st = {tower: 2, sim: 11, heights: [0, 5, 10], inputs: ["tower_top_afa_mod", "wind_speed", "rotor_speed", "blade_pitch", "plat_pitch", "wave_elev"], xr: [400, 1000], models: ["tcn", "prob_tcn", "transformer", "naive"],
  proto: "within", lbTower: "mean", metric: "r2", height: 10, search: "", fams: new Set(), sel: "tcn"};

// ---------- explorer ----------
const CACHE = {};
async function loadTower(t) {
  if (!CACHE[t]) CACHE[t] = fetch(`static/data/series_${TOWERS[t]}.txt`).then(r => r.text()).then(txt => {
    const bin = atob(txt.trim()), u = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) u[j] = bin.charCodeAt(j);
    return new Int16Array(u.buffer); });
  return CACHE[t];
}
function series(t, i) {
  const n = D.n, nc = D.channels.length, [lo, sc] = D.scales[`${TOWERS[t]}/${D.ids[i]}`], A = S[t];
  const block = i * nc * n, out = {};
  D.channels.forEach((c, k) => {
    const a = new Float32Array(n), off = block + k * n;
    for (let j = 0; j < n; j++) a[j] = (A[off + j] + 32500) * sc[k] + lo[k];
    out[c] = a;
  });
  return out;
}

function simCard() {
  const sid = D.ids[st.sim], op = D.ops[sid];
  const item = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
  $("sim-card").innerHTML = item("sim_id", sid) + item("Tower", TOWER_NAME[TOWERS[st.tower]]) + item("Wind speed [m/s]", op.wind.toFixed(1)) +
    item("Seed", 1) + item("Wave height Hs [m]", op.hs.toFixed(2)) + item("Wave period Tp [s]", op.tp.toFixed(2)) +
    item("Wind regime", `<span class="tag ${GROUP_TAG[op.wind_group]}">${op.wind_group}</span>`) +
    item("Wave regime", `<span class="tag ${GROUP_TAG[op.wave_group]}">${op.wave_group}</span>`);
  $("ex-wind-out").textContent = `${op.wind.toFixed(1)} m/s`;
  $("ex-sim").value = st.sim; $("ex-wind").value = st.sim;
}

function drawSeries() {
  const x = series(st.tower, st.sim);
  const rows = INPUTS.filter(r => st.inputs.includes(r[0]));
  const n = rows.length, gap = n > 1 ? 0.2 / n : 0, h = (1 - gap * (n - 1)) / n, lay = L({margin: {l: 80, r: 20, t: 16, b: 50}});
  $("plot-inputs").style.height = `${Math.max(260, 90 * n + 90)}px`;
  const tr = rows.map(([c, name, col], k) => ({x: time, y: x[c], type: "scattergl", mode: "lines", line: {width: 1, color: col},
    xaxis: "x", yaxis: `y${k ? k + 1 : ""}`, hovertemplate: `%{x:.1f} s<br>${name} %{y:.3f}<extra></extra>`}));
  rows.forEach(([, name], k) => { lay[`yaxis${k ? k + 1 : ""}`] = AX({domain: [1 - (k + 1) * h - k * gap, 1 - k * h - k * gap], title: {text: name, font: {size: 10}}, nticks: 3}); });
  lay.xaxis = AX({anchor: `y${n}`, title: {text: "Time [s]"}, range: st.xr});
  Plotly.react("plot-inputs", tr, lay, CFG);

  const H = ["tower_bottom", "tower_1", "tower_2", "tower_3", "tower_4", "tower_5", "tower_6", "tower_7", "tower_8", "tower_9", "tower_top"];
  const T = L({margin: {l: 80, r: 20, t: 24, b: 50}, showlegend: true,
    legend: {orientation: "h", x: 0, y: 1.08, yanchor: "bottom", font: {size: 10}}});
  T.yaxis = AX({domain: [0.54, 1], title: {text: "Fore-aft M<sub>FA</sub> [MN m]", font: {size: 10}}});
  T.yaxis2 = AX({domain: [0, 0.46], title: {text: "Side-side M<sub>SS</sub> [MN m]", font: {size: 10}}});
  T.xaxis = AX({anchor: "y2", title: {text: "Time [s]"}, range: st.xr});
  T.annotations = [
    {text: "<b>Target: fore-aft moment (scored)</b>", xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "right", yanchor: "bottom", showarrow: false, font: {size: 11, color: RED}},
    {text: "<b>Side-side moment (released, not scored)</b>", xref: "paper", yref: "paper", x: 1, y: 0.46, xanchor: "right", yanchor: "bottom", showarrow: false, font: {size: 11, color: RED}}];
  const t2 = [];
  st.heights.slice().sort((a, b) => a - b).forEach(k => {
    const col = heightColor(k);
    t2.push({x: time, y: x[`${H[k]}_mfa`], type: "scattergl", mode: "lines", line: {width: 1, color: col}, xaxis: "x", yaxis: "y", name: GAUGE[k], legendgroup: `h${k}`,
      hovertemplate: `${GAUGE[k]} FA %{y:.1f}<extra></extra>`});
    t2.push({x: time, y: x[`${H[k]}_mss`], type: "scattergl", mode: "lines", line: {width: 1, color: col}, xaxis: "x", yaxis: "y2", name: GAUGE[k], legendgroup: `h${k}`, showlegend: false,
      hovertemplate: `${GAUGE[k]} SS %{y:.1f}<extra></extra>`});
  });
  Plotly.react("plot-targets", t2, T, CFG);
}

function syncZoom(src) {
  $(src).on("plotly_relayout", ev => {
    let r = null;
    if (ev["xaxis.range[0]"] !== undefined) r = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
    else if (ev["xaxis.range"]) r = ev["xaxis.range"];
    else if (ev["xaxis.autorange"]) r = [400, 1000];
    if (!r || (Math.abs(r[0] - st.xr[0]) < 1e-6 && Math.abs(r[1] - st.xr[1]) < 1e-6)) return;
    setWindow(r);
  });
}

function drawProfile() {
  const P = D.profiles[`${TOWERS[st.tower]}/${D.ids[st.sim]}`];
  const tr = [{x: P.true, y: D.heights, mode: "lines+markers", line: {color: "#000000", width: 3}, marker: {size: 7, color: "#000000"}, name: "True",
    hovertemplate: "True %{x:.2e}<br>%{y:.1f} m<extra></extra>"}];
  st.models.forEach(m => tr.push({x: P[m], y: D.heights, mode: "lines+markers", name: NAMES[m],
    line: {color: MODEL_COLOR[m], width: 2, dash: MODEL_DASH[m] || "solid"}, marker: {size: 5, color: MODEL_COLOR[m]},
    hovertemplate: `${NAMES[m]} %{x:.2e}<br>%{y:.1f} m<extra></extra>`}));
  Plotly.react("plot-profile", tr, L({showlegend: true, legend: {orientation: "h", y: 1.02, yanchor: "bottom", x: 0}, margin: {l: 70, r: 20, t: 50, b: 50},
    xaxis: AX({type: "log", title: {text: "Fatigue damage over 600 s (log)"}, exponentformat: "power"}),
    yaxis: AX({title: {text: "Gauge height [m]"}, range: [-4, 154]})}), CFG);
  const top = D.heights.length - 1;
  $("profile-note").textContent = st.models.length ? "Top gauge, reconstructed / true damage: " +
    st.models.map(m => `${NAMES[m]} ${(P[m][top] / P.true[top]).toFixed(2)}`).join(", ") + "." : "Pick models to compare.";
  $("models").querySelectorAll("label").forEach(l => {
    const m = l.dataset.m, on = st.models.includes(m);
    l.querySelector("input").checked = on;
    l.querySelector(".sw").style.background = on ? MODEL_COLOR[m] : "transparent";
  });
}

function selectSim(i) { st.sim = i; simCard(); drawSeries(); drawProfile(); }
async function selectTower(t) { st.tower = t; $("ex-tower").value = t; S[t] = await loadTower(t); simCard(); drawSeries(); drawProfile(); }

// ---------- regime grid ----------
function regimeGrid() {
  const lab = {"In-train": "In-train", "Interpolate": "Interpolate", "Extrapolate": "Extrapolate"}, K = ["In-train", "Interpolate", "Extrapolate"], ab = {"In-train": "IT", "Interpolate": "IP", "Extrapolate": "EX"};
  let h = `<div></div>` + K.map(k => `<div class="h">wave ${lab[k]}</div>`).join("");
  K.forEach(a => { h += `<div class="h">wind ${lab[a]}</div>` + K.map(b => `<div class="c ${a === "Extrapolate" && b === "Extrapolate" ? "deep" : ""}"><b>${D.regime_grid[`${a}|${b}`].toLocaleString("en-US")}</b><span>${ab[a]}_${ab[b]}</span></div>`).join(""); });
  $("regime-grid").innerHTML = h;
}

// ---------- leaderboard ----------
function towerOptions() {
  const o = st.proto === "within"
    ? [["mean", "Mean of the three towers"]].concat(TOWERS.map((t, k) => [k, TOWER_NAME[t]]))
    : [["mean", "Mean of the six pairs"]].concat(D.lb.zeroshot.tcn.pairs.map((p, k) => { const [s, t] = p.split(">"); return [k, `${TOWER_NAME[s]} → ${TOWER_NAME[t]}`]; }));
  $("lb-tower").innerHTML = o.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
  st.lbTower = "mean"; $("lb-tower").value = "mean";
}
function cell(e, met, h) {
  const rows = e[met];
  if (st.lbTower === "mean") return rows.reduce((s, r) => s + r[h], 0) / rows.length;
  return rows[+st.lbTower][h];
}
function worst(e, met, h) {
  if (st.lbTower !== "mean") return null;
  const v = e[met].map(r => r[h]);
  return met === "mre" ? Math.max(...v) : met === "median_ratio" ? v.reduce((a, b) => Math.abs(Math.log(b)) > Math.abs(Math.log(a)) ? b : a) : Math.min(...v);
}
const score = (met, v) => met === "median_ratio" ? -Math.abs(Math.log(v)) : met === "mre" ? -v : v;
function ranking(h) {
  const B = D.lb[st.proto], ms = Object.keys(B);
  const rows = ms.map(m => ({m, v: score(st.metric, cell(B[m], st.metric, h))}));
  rows.sort((a, b) => b.v - a.v);
  const r = {}; rows.forEach((x, k) => { r[x.m] = k + 1; }); return r;
}
function fmt(v, k) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (k === "r2" && v < -1) return "&lt;−1";
  if (k === "within2") return `${Math.round(v * 100)}%`;
  return v.toFixed(2).replace("-", "−");
}
function drawBoard() {
  const B = D.lb[st.proto], h = st.height, rk = ranking(h), rk0 = ranking(0);
  let ms = Object.keys(B).sort((a, b) => rk[a] - rk[b]);
  ms = ms.filter(m => (!st.fams.size || st.fams.has(FAMILY[m])) && (!st.search || (NAMES[m] || m).toLowerCase().includes(st.search)));
  const head = `<thead><tr><th class="l">#</th><th class="l">Model</th><th class="l">Family</th>` +
    MET.map(([k, t]) => `<th class="sortable ${st.metric === k ? "sorted" : ""}" data-k="${k}">${t}</th>`).join("") +
    `<th>Base #</th><th>Shift</th></tr></thead>`;
  const body = ms.map(m => {
    const e = B[m], shift = rk0[m] - rk[m];
    const sh = h === 0 ? "" : shift > 0 ? `<span class="up">▲ ${shift}</span>` : shift < 0 ? `<span class="down">▼ ${-shift}</span>` : "=";
    return `<tr data-m="${m}" class="${m === st.sel ? "sel" : ""} ${m === "physics" || m === "naive" ? "refrow" : ""}"><td class="l">${rk[m]}</td>` +
      `<td class="l"><span class="fam-dot" style="background:${FAMILY_COLOR[FAMILY[m]]}"></span>${NAMES[m] || m}${e.seeds < 3 ? " †" : ""}</td><td class="l">${FAMILY[m]}</td>` +
      MET.map(([k]) => { const v = cell(e, k, h), w = worst(e, k, h);
        return `<td class="${k === "r2" && v < 0 ? "neg" : ""}">${k === st.metric ? `<b>${fmt(v, k)}</b>` : fmt(v, k)}${w !== null ? ` <small>${fmt(w, k)}</small>` : ""}</td>`; }).join("") +
      `<td>${rk0[m]}</td><td>${sh}</td></tr>`;
  }).join("");
  $("lb-table").innerHTML = head + `<tbody>${body}</tbody>`;
  drawHeights(); drawCrossover(rk, rk0);
}
function drawHeights() {
  const B = D.lb[st.proto], h = st.height, rk = ranking(h);
  const top3 = Object.keys(B).filter(m => m !== "physics" && m !== "naive").sort((a, b) => rk[a] - rk[b]).slice(0, 3);
  const show = [...new Set([st.sel, ...top3, "physics", "naive"])].filter(m => B[m]);
  const tr = show.map(m => ({x: D.zh, y: D.zh.map((_, k) => Math.max(-1, cell(B[m], "r2", k))), mode: "lines+markers", name: NAMES[m],
    line: {color: MODEL_COLOR[m] || REF_DARK, width: m === st.sel ? 3.5 : 1.8, dash: MODEL_DASH[m] || "solid"}, marker: {size: m === st.sel ? 7 : 4, color: MODEL_COLOR[m] || REF_DARK},
    hovertemplate: `${NAMES[m]}<br>z/H %{x:.2f}: R² %{y:.3f}<extra></extra>`}));
  Plotly.react("plot-heights", tr, L({showlegend: true, legend: {orientation: "h", y: 1.12, x: 0, font: {size: 10}}, margin: {l: 60, r: 20, t: 50, b: 50},
    title: {text: `${st.proto === "within" ? "Within tower" : "Zero-shot"}: R² of log₁₀ damage along the tower`, font: {size: 12}, y: 0.99},
    shapes: [{type: "line", x0: D.zh[h], x1: D.zh[h], y0: -1, y1: 1, line: {color: "#bdbdbd", dash: "dot", width: 1}}],
    xaxis: AX({title: {text: "z/H"}, range: [-0.03, 1.03]}), yaxis: AX({title: {text: "R²"}, range: [-1.05, 1.05]})}), CFG);
}
function drawCrossover(rk, rk0) {
  const ms = Object.keys(D.lb[st.proto]), n = ms.length;
  Plotly.react("plot-crossover", [
    {x: [1, n], y: [1, n], mode: "lines", line: {color: "#d0d0d0", dash: "dash", width: 1}, hoverinfo: "skip"},
    {x: ms.map(m => rk0[m]), y: ms.map(m => rk[m]), mode: "markers+text", text: ms.map(m => m === st.sel || rk[m] <= 3 || rk0[m] <= 3 ? NAMES[m] : ""),
      textposition: "top center", textfont: {size: 10}, customdata: ms, hovertemplate: ms.map(m => `${NAMES[m]}<br>base #%{x}, here #%{y}<extra></extra>`),
      marker: {size: ms.map(m => m === st.sel ? 14 : 9), color: ms.map(m => FAMILY_COLOR[FAMILY[m]]), line: {color: "#ffffff", width: 1}}}],
    L({margin: {l: 60, r: 20, t: 40, b: 50}, title: {text: `Rank at the base vs rank at ${GAUGE[st.height].toLowerCase()}`, font: {size: 12}, y: 0.98},
      xaxis: AX({title: {text: "Rank at the base"}, range: [0, n + 1]}), yaxis: AX({title: {text: `Rank at ${GAUGE[st.height].toLowerCase()}`}, range: [n + 1, 0]})}), CFG);
}
function families() {
  const fams = [...new Set(Object.values(FAMILY))];
  $("lb-families").innerHTML = fams.map(f => `<button type="button" class="chip" data-f="${f}" aria-pressed="true"><span class="dot" style="background:${FAMILY_COLOR[f]}"></span>${f}</button>`).join("");
  $("lb-families").onclick = e => {
    const b = e.target.closest(".chip"); if (!b) return;
    const all = [...$("lb-families").querySelectorAll(".chip")];
    if (!st.fams.size) { st.fams = new Set([b.dataset.f]); }
    else if (st.fams.has(b.dataset.f)) { st.fams.delete(b.dataset.f); }
    else st.fams.add(b.dataset.f);
    all.forEach(c => c.setAttribute("aria-pressed", String(!st.fams.size || st.fams.has(c.dataset.f))));
    drawBoard();
  };
}

// ---------- init ----------
async function init() {
  $("copy-bibtex").onclick = () => { const txt = $("bibtex-content").textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).catch(() => {
      const r = document.createRange(); r.selectNodeContents($("bibtex-content")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }); };
  D = await fetch("static/data/data.json").then(r => r.json());
  S = {}; S[st.tower] = await loadTower(st.tower);
  time = Float32Array.from({length: D.n}, (_, j) => D.t0 + j * D.dt);
  $("ex-source").innerHTML = `<span class="icon"><i class="fas fa-check-circle"></i></span> ${D.ids.length} simulations per tower, inputs and fore-aft and side-side moments at 5 Hz.`;

  $("ex-sim").innerHTML = D.ids.map((s, k) => { const o = D.ops[s]; return `<option value="${k}">${s} · ${o.wind.toFixed(1)} m/s · Hs ${o.hs.toFixed(2)} m</option>`; }).join("");
  $("ex-tower").onchange = e => selectTower(+e.target.value);
  heightChips(); inputChips();
  $("input-chips").onclick = e => { const b = e.target.closest(".chip"); if (!b) return; const c = b.dataset.c, i = st.inputs.indexOf(c);
    if (i >= 0) { if (st.inputs.length > 1) st.inputs.splice(i, 1); } else st.inputs.push(c);
    inputChips(); drawSeries(); };
  $("tw-apply").onclick = () => setWindow([$("tw-from").value, $("tw-to").value]);
  ["tw-from", "tw-to"].forEach(id => $(id).addEventListener("keydown", e => { if (e.key === "Enter") setWindow([$("tw-from").value, $("tw-to").value]); }));
  $("tw-presets").onclick = e => { const b = e.target.closest("button"); if (!b) return; const w = +b.dataset.w;
    if (!w) setWindow([400, 1000]); else { const c = (st.xr[0] + st.xr[1]) / 2; let a = Math.max(400, c - w / 2); a = Math.min(a, 1000 - w); setWindow([a, a + w]); } };
  $("height-chips").onclick = e => { const b = e.target.closest(".chip"); if (!b) return; const k = +b.dataset.k, i = st.heights.indexOf(k);
    if (i >= 0) { if (st.heights.length > 1) st.heights.splice(i, 1); } else st.heights.push(k);
    heightChips(); inputChips();
  $("input-chips").onclick = e => { const b = e.target.closest(".chip"); if (!b) return; const c = b.dataset.c, i = st.inputs.indexOf(c);
    if (i >= 0) { if (st.inputs.length > 1) st.inputs.splice(i, 1); } else st.inputs.push(c);
    inputChips(); drawSeries(); };
  $("tw-apply").onclick = () => setWindow([$("tw-from").value, $("tw-to").value]);
  ["tw-from", "tw-to"].forEach(id => $(id).addEventListener("keydown", e => { if (e.key === "Enter") setWindow([$("tw-from").value, $("tw-to").value]); }));
  $("tw-presets").onclick = e => { const b = e.target.closest("button"); if (!b) return; const w = +b.dataset.w;
    if (!w) setWindow([400, 1000]); else { const c = (st.xr[0] + st.xr[1]) / 2; let a = Math.max(400, c - w / 2); a = Math.min(a, 1000 - w); setWindow([a, a + w]); } }; drawSeries(); };
  $("ex-sim").onchange = e => selectSim(+e.target.value);
  $("ex-wind").oninput = e => selectSim(+e.target.value);


  const order = ["tcn", "prob_tcn", "transformer", "mamba", "lstm", "s4", "timesnet", "unet", "fno", "itransformer", "fits", "dlinear", "spectral", "naive"].filter(m => D.models.includes(m));
  $("models").innerHTML = order.map(m => `<label data-m="${m}"><input type="checkbox" id="m-${m}"><span class="sw"></span>${NAMES[m]}</label>`).join("");
  $("models").onchange = e => { const m = e.target.closest("label").dataset.m, k = st.models.indexOf(m);
    if (k >= 0) st.models.splice(k, 1); else st.models.push(m); drawProfile(); };

  simCard(); drawSeries(); drawProfile(); regimeGrid();
  syncZoom("plot-inputs"); syncZoom("plot-targets");

  towerOptions(); families();
  $("lb-protocol").onchange = e => { st.proto = e.target.value; if (!D.lb[st.proto][st.sel]) st.sel = "tcn"; towerOptions(); drawBoard(); };
  $("lb-tower").onchange = e => { st.lbTower = e.target.value; drawBoard(); };
  $("lb-metric").onchange = e => { st.metric = e.target.value; drawBoard(); };
  $("lb-search").oninput = e => { st.search = e.target.value.trim().toLowerCase(); drawBoard(); };
  $("lb-height").onclick = e => { const b = e.target.closest("button"); if (!b) return; st.height = +b.dataset.h;
    $("lb-height").querySelectorAll("button").forEach(x => { const on = x === b; x.classList.toggle("is-dark", on); x.classList.toggle("is-selected", on); }); drawBoard(); };
  $("lb-table").onclick = e => {
    const th = e.target.closest("th.sortable"); if (th) { st.metric = th.dataset.k; $("lb-metric").value = st.metric; drawBoard(); return; }
    const tr = e.target.closest("tr[data-m]"); if (tr) { st.sel = tr.dataset.m; drawBoard(); } };
  drawBoard();
  $("plot-crossover").on("plotly_click", ev => { const m = ev.points[0].customdata; if (m) { st.sel = m; drawBoard(); } });
}
document.addEventListener("DOMContentLoaded", () => init().catch(err => {
  $("ex-source").innerHTML = `<span class="has-text-danger">The data files did not load (${err.message}). Reload the page to try again.</span>`; }));
