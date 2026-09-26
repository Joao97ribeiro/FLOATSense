// FLOATSense project page: dataset explorer, envelope, leaderboard along the height.

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
const DAMAGE_SCALE = [[0, "#f7f7f7"], [0.35, "#f0b3b0"], [0.7, RED], [1, WINE]];
const MOMENT_SCALE = [[0, NAVY], [0.5, "#f7f7f7"], [1, RED]];
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
const st = {tower: 2, sim: 11, gauge: 10, sec: "top", xr: null, models: ["tcn", "prob_tcn", "transformer", "naive"],
  proto: "within", lbTower: "mean", metric: "r2", height: 10, search: "", fams: new Set(), sel: "tcn"};

// ---------- explorer ----------
function series(t, i) {
  const n = D.n, nc = D.channels.length, [lo, sc] = D.scales[`${TOWERS[t]}/${D.ids[i]}`];
  const block = (t * D.ids.length + i) * nc * n, out = {};
  D.channels.forEach((c, k) => {
    const a = new Float32Array(n), off = block + k * n;
    for (let j = 0; j < n; j++) a[j] = (S[off + j] + 32500) * sc[k] + lo[k];
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
  const rows = [["tower_top_afa_mod", "Accel. [m/s²]", NAVY], ["wind_speed", "Wind [m/s]", SCADA], ["rotor_speed", "Rotor [rpm]", SCADA],
    ["blade_pitch", "Pitch [deg]", SCADA], ["wave_elev", "Wave [m]", CYAN], ["plat_pitch", "Platform pitch [deg]", CYAN]];
  const n = rows.length, gap = 0.035, h = (1 - gap * (n - 1)) / n, lay = L({margin: {l: 80, r: 20, t: 24, b: 50}});
  const tr = rows.map(([c, name, col], k) => ({x: time, y: x[c], type: "scattergl", mode: "lines", line: {width: 1, color: col},
    xaxis: "x", yaxis: `y${k ? k + 1 : ""}`, hovertemplate: `%{x:.1f} s<br>${name} %{y:.3f}<extra></extra>`}));
  rows.forEach(([, name], k) => { lay[`yaxis${k ? k + 1 : ""}`] = AX({domain: [1 - (k + 1) * h - k * gap, 1 - k * h - k * gap], title: {text: name, font: {size: 10}}, nticks: 3}); });
  lay.xaxis = AX({anchor: `y${n}`, title: {text: "Time [s]"}, range: st.xr || [400, 1000]});
  lay.annotations = [
    {text: "<b>Model inputs</b>", xref: "paper", yref: "paper", x: 0, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false, font: {size: 11, color: NAVY}},
    {text: "<b>Context (not inputs)</b>", xref: "paper", yref: "paper", x: 0, y: 2 * h + gap, xanchor: "left", yanchor: "bottom", showarrow: false, font: {size: 11, color: CYAN}}];
  Plotly.react("plot-inputs", tr, lay, CFG);

  const mom = D.channels.slice(8), g = st.gauge;
  const T = L({margin: {l: 80, r: 20, t: 24, b: 50}, showlegend: true,
    legend: {orientation: "h", x: 1, xanchor: "right", y: 0.37, yanchor: "bottom", font: {size: 11}, bgcolor: "rgba(255,255,255,0.7)"}});
  T.yaxis = AX({domain: [0.45, 1], title: {text: "Gauge height [m]", font: {size: 10}}});
  T.yaxis2 = AX({domain: [0, 0.37], title: {text: "M<sub>FA</sub> [MN m]", font: {size: 10}}});
  T.xaxis = AX({anchor: "y2", title: {text: "Time [s]"}, range: st.xr || [400, 1000]});
  T.annotations = [{text: "<b>Target: fore-aft moment along the tower</b>", xref: "paper", yref: "paper", x: 0, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false, font: {size: 11, color: RED}}];
  const t2 = [{type: "heatmap", x: time, y: D.heights, z: mom.map(c => Array.from(x[c])), xaxis: "x", yaxis: "y", colorscale: MOMENT_SCALE, zmid: 0, showlegend: false,
      colorbar: {title: {text: "MN m", side: "right", font: {size: 10}}, thickness: 10, len: 0.55, y: 0.725, tickfont: {size: 9}},
      hovertemplate: "%{x:.1f} s · %{y:.1f} m<br>%{z:.1f} MN m<extra></extra>"},
    {x: time, y: x[mom[0]], type: "scattergl", mode: "lines", line: {width: 1, color: INK}, xaxis: "x", yaxis: "y2", name: "Base", hovertemplate: "Base %{y:.1f}<extra></extra>"}];
  if (g !== 0) t2.push({x: time, y: x[mom[g]], type: "scattergl", mode: "lines", line: {width: 1, color: RED}, xaxis: "x", yaxis: "y2", name: GAUGE[g], hovertemplate: `${GAUGE[g]} %{y:.1f}<extra></extra>`});
  Plotly.react("plot-targets", t2, T, CFG);
}

function syncZoom(src, dst) {
  $(src).on("plotly_relayout", ev => {
    let r = null;
    if (ev["xaxis.range[0]"] !== undefined) r = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
    else if (ev["xaxis.range"]) r = ev["xaxis.range"];
    else if (ev["xaxis.autorange"]) r = [400, 1000];
    if (!r || (st.xr && r[0] === st.xr[0] && r[1] === st.xr[1])) return;
    st.xr = r; Plotly.relayout(dst, {"xaxis.range": r});
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

function drawEnvelope() {
  const z = D.env[TOWERS[st.tower]][st.sec], hl = 24;
  const zt = z[0].map((_, j) => z.map(r => r[j]));
  const text = zt.map((row, j) => row.map((_, w) => { const [hs, tp] = D.waves[w + 1][j], g = D.groups[w + 1][j];
    return `U ${D.winds[w]} m/s · Hs ${hs} m · Tp ${tp} s<br>${g[0]} / ${g[1]} · ${g[2]}`; }));
  const ids = D.ids.map((_, k) => k);
  Plotly.react("plot-envelope", [
    {type: "heatmap", z: zt, text, colorscale: DAMAGE_SCALE, hovertemplate: "%{text}<br>log₁₀ D %{z:.2f}<extra></extra>",
      colorbar: {title: {text: "log₁₀ D", side: "right", font: {size: 10}}, thickness: 10, tickfont: {size: 9}}},
    {x: ids, y: ids.map(() => hl), mode: "markers", marker: {size: 9, color: "rgba(0,0,0,0)", line: {color: INK, width: 1.2}}, hovertemplate: "Explorer simulation<extra></extra>"},
    {x: [st.sim], y: [hl], mode: "markers", marker: {size: 14, color: "rgba(0,0,0,0)", line: {color: NAVY, width: 3}}, hoverinfo: "skip"}],
    L({margin: {l: 60, r: 10, t: 10, b: 45},
      xaxis: AX({title: {text: "Wind speed [m/s]"}, tickvals: [0, 5, 10, 15, 21], ticktext: [0, 5, 10, 15, 21].map(k => D.winds[k]), showgrid: false}),
      yaxis: AX({title: {text: "Sea state (Hs, Tp)"}, tickvals: [3, 17, 31, 45], ticktext: ["Hs 1", "Hs 3", "Hs 5", "Hs 7"], showgrid: false})}), CFG);
}

function drawLife() {
  const tr = TOWERS.map((t, k) => ({x: D.life[t], y: D.heights, mode: "lines+markers", name: TOWER_NAME[t],
    line: {color: TOWER_COLOR[t], width: 2.5}, marker: {size: 6, color: TOWER_COLOR[t]}, hovertemplate: `${TOWER_NAME[t]} %{x:.3f}<br>%{y:.1f} m<extra></extra>`}));
  Plotly.react("plot-life", tr, L({showlegend: true, legend: {orientation: "h", y: -0.2, x: 0.5, xanchor: "center"}, margin: {l: 60, r: 20, t: 10, b: 70},
    xaxis: AX({title: {text: "25-year fore-aft damage"}, rangemode: "tozero"}), yaxis: AX({title: {text: "Gauge height [m]"}})}), CFG);
}

function selectSim(i) { st.sim = i; simCard(); drawSeries(); drawProfile(); drawEnvelope(); }
function selectTower(t) { st.tower = t; $("ex-tower").value = t; simCard(); drawSeries(); drawProfile(); drawEnvelope(); }

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
  $("lb-height-out").textContent = `${GAUGE[h]} (z/H ${D.zh[h].toFixed(2)})`;
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
  const [d, b] = await Promise.all([
    fetch("static/data/data.json").then(r => r.json()),
    fetch("static/data/series.txt").then(r => r.text()).then(t => { const bin = atob(t.trim()), u = new Uint8Array(bin.length);
      for (let j = 0; j < bin.length; j++) u[j] = bin.charCodeAt(j); return u.buffer; })]);
  D = d; S = new Int16Array(b);
  time = Float32Array.from({length: D.n}, (_, j) => D.t0 + j * D.dt);
  $("ex-source").innerHTML = `<span class="icon"><i class="fas fa-check-circle"></i></span> Loaded ${D.ids.length * 3} simulations (${D.ids.length} per tower), 19 channels at 5 Hz.`;

  $("ex-sim").innerHTML = D.ids.map((s, k) => { const o = D.ops[s]; return `<option value="${k}">${s} · ${o.wind.toFixed(1)} m/s · Hs ${o.hs.toFixed(2)} m</option>`; }).join("");
  $("ex-gauge").innerHTML = GAUGE.map((g, k) => `<option value="${k}">${g} (z/H ${D.zh[k].toFixed(2)})</option>`).join("");
  $("ex-gauge").value = st.gauge;
  $("ex-tower").onchange = e => selectTower(+e.target.value);
  $("ex-sim").onchange = e => selectSim(+e.target.value);
  $("ex-wind").oninput = e => selectSim(+e.target.value);
  $("ex-gauge").onchange = e => { st.gauge = +e.target.value; drawSeries(); };
  $("env-sec").onclick = e => { const btn = e.target.closest("button"); if (!btn) return; st.sec = btn.dataset.sec;
    $("env-sec").querySelectorAll("button").forEach(x => { const on = x.dataset.sec === st.sec; x.classList.toggle("is-dark", on); x.classList.toggle("is-selected", on); }); drawEnvelope(); };

  const order = ["tcn", "prob_tcn", "transformer", "mamba", "lstm", "s4", "timesnet", "unet", "fno", "itransformer", "fits", "dlinear", "spectral", "naive"].filter(m => D.models.includes(m));
  $("models").innerHTML = order.map(m => `<label data-m="${m}"><input type="checkbox" id="m-${m}"><span class="sw"></span>${NAMES[m]}</label>`).join("");
  $("models").onchange = e => { const m = e.target.closest("label").dataset.m, k = st.models.indexOf(m);
    if (k >= 0) st.models.splice(k, 1); else st.models.push(m); drawProfile(); };

  simCard(); drawSeries(); drawProfile(); drawEnvelope(); drawLife(); regimeGrid();
  syncZoom("plot-inputs", "plot-targets"); syncZoom("plot-targets", "plot-inputs");
  $("plot-envelope").on("plotly_click", ev => { const p = ev.points[0]; if (p.curveNumber > 0 || p.y === 24) selectSim(p.x); });

  towerOptions(); families();
  $("lb-protocol").onchange = e => { st.proto = e.target.value; if (!D.lb[st.proto][st.sel]) st.sel = "tcn"; towerOptions(); drawBoard(); };
  $("lb-tower").onchange = e => { st.lbTower = e.target.value; drawBoard(); };
  $("lb-metric").onchange = e => { st.metric = e.target.value; drawBoard(); };
  $("lb-search").oninput = e => { st.search = e.target.value.trim().toLowerCase(); drawBoard(); };
  $("lb-height").oninput = e => { st.height = +e.target.value; drawBoard(); };
  $("lb-table").onclick = e => {
    const th = e.target.closest("th.sortable"); if (th) { st.metric = th.dataset.k; $("lb-metric").value = st.metric; drawBoard(); return; }
    const tr = e.target.closest("tr[data-m]"); if (tr) { st.sel = tr.dataset.m; drawBoard(); } };
  drawBoard();
  $("plot-crossover").on("plotly_click", ev => { const m = ev.points[0].customdata; if (m) { st.sel = m; drawBoard(); } });
}
document.addEventListener("DOMContentLoaded", () => init().catch(err => {
  $("ex-source").innerHTML = `<span class="has-text-danger">The data files did not load (${err.message}). Reload the page to try again.</span>`; }));
