// FLOATSense project page: dataset explorer, damage profiles, envelope and leaderboard.

// Links: null while the anonymized repositories are pending.
const LINKS = {
  paper: null,
  code: "https://github.com/Joao97ribeiro/FLOATSense",
  data: null,
  subset: null,
};

const ICONS = {
  gh: '<svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>',
  doc: '<svg viewBox="0 0 16 16"><path d="M3 0h7l4 4v11a1 1 0 01-1 1H3a1 1 0 01-1-1V1a1 1 0 011-1zm6 1v4h4L9 1zM4 8h8v1H4V8zm0 2h8v1H4v-1zm0 2h5v1H4v-1z"/></svg>',
  db: '<svg viewBox="0 0 16 16"><path d="M8 0C4.1 0 1 1.1 1 2.5v11C1 14.9 4.1 16 8 16s7-1.1 7-2.5v-11C15 1.1 11.9 0 8 0zm0 1c3.3 0 6 .8 6 1.5S11.3 4 8 4 2 3.2 2 2.5 4.7 1 8 1zM2 4.1C3.3 4.7 5.5 5 8 5s4.7-.3 6-.9V7c0 .7-2.7 1.5-6 1.5S2 7.7 2 7V4.1zm0 4.5c1.3.6 3.5.9 6 .9s4.7-.3 6-.9v2.9c0 .7-2.7 1.5-6 1.5s-6-.8-6-1.5V8.6z"/></svg>',
  box: '<svg viewBox="0 0 16 16"><path d="M8 0l7 3.5v9L8 16l-7-3.5v-9L8 0zm0 1.1L2.3 4 8 6.9 13.7 4 8 1.1zM2 4.8v7.1l5.5 2.8V7.6L2 4.8zm12 0L8.5 7.6v7.1l5.5-2.8V4.8z"/></svg>',
  copy: '<svg viewBox="0 0 16 16"><path d="M4 4V1a1 1 0 011-1h9a1 1 0 011 1v9a1 1 0 01-1 1h-3v4a1 1 0 01-1 1H1a1 1 0 01-1-1V5a1 1 0 011-1h3zm1 0h5a1 1 0 011 1v5h3V1H5v3zM1 5v10h9V5H1z"/></svg>',
};

const NAMES = {tcn:"TCN", prob_tcn:"Prob-TCN", fits:"FITS", dlinear:"DLinear", chronos:"Chronos", s4:"S4D", lstm:"LSTM",
  spectral:"Spectral gain", hybrid:"Hybrid", hybrid_tcn:"Hybrid-TCN", timesnet:"TimesNet", mamba:"Mamba", transformer:"PatchTST",
  physics:"Physics", naive:"Floor", fno:"FNO", unet:"U-Net", itransformer:"iTransformer", moment:"MOMENT", moment_ft:"MOMENT-ft",
  timesfm:"TimesFM", timesfm_ft:"TimesFM-ft"};
const FAMILY = {tcn:"convolutional", prob_tcn:"convolutional", unet:"convolutional", timesnet:"convolutional", fits:"linear",
  dlinear:"linear", spectral:"linear", lstm:"recurrent", s4:"state space", mamba:"state space", transformer:"attention",
  itransformer:"attention", fno:"operator", chronos:"foundation", moment:"foundation", moment_ft:"foundation",
  timesfm:"foundation", timesfm_ft:"foundation", hybrid:"physics + ML", hybrid_tcn:"physics + ML", physics:"reference",
  naive:"reference"};
// Colors of the paper figures (paper_style.py and fs_common.py).
const NAVY = "#294366", RED = "#b02c27", REF = "#b8b8b8", REF_DARK = "#8f8f8f", CYAN = "#7cc0cd", SCADA = "#4383ad",
  TAUPE = "#8f7a6e", SLATE = "#9aa3ab", CORAL = "#d06662", PINK = "#dd9c98", WINE = "#6f1b17", INK = "#333333";
const MODEL_COLOR = {tcn:NAVY, prob_tcn:RED, lstm:CYAN, transformer:TAUPE, s4:SLATE, fits:WINE, timesnet:PINK,
  mamba:CYAN, naive:REF_DARK, physics:"#222222", spectral:CORAL, unet:"#565656", fno:"#6b6b6b", itransformer:"#aaaaaa", dlinear:"#d2d2d2"};
const MODEL_DASH = {mamba:"dash", naive:"dot"};
const TOWER_COLOR = {ref:REF, opt1:NAVY, opt2:RED};
const TOWER_TEXT = {ref:REF_DARK, opt1:NAVY, opt2:RED};
const DAMAGE_SCALE = [[0,"#f7f7f7"],[0.35,"#f0b3b0"],[0.7,RED],[1,WINE]];
const MOMENT_SCALE = [[0,NAVY],[0.5,"#f7f7f7"],[1,RED]];
const GAUGE = ["Base","Gauge 1","Gauge 2","Gauge 3","Gauge 4","Gauge 5","Gauge 6","Gauge 7","Gauge 8","Gauge 9","Top"];
const GROUP_TAG = {"In-train":"grp-it", "Interpolate":"grp-ip", "Extrapolate":"grp-ex"};
const MET = [["r2","R²"],["within2","×2"],["median_ratio","ratio"],["within_corr","within-corr"],["mre","MRE"]];

const COMMON = {margin:{l:70,r:20,t:30,b:50}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
  font:{family:"Helvetica, Arial, sans-serif", size:12, color:"#363636"}, showlegend:false};
const CFG = {displaylogo:false, responsive:true, modeBarButtonsToRemove:["lasso2d","select2d"]};
const AX = o => Object.assign({gridcolor:"#ececec", zerolinecolor:"#d0d0d0", linecolor:"#bdbdbd", tickfont:{size:10}}, o);
const $ = id => document.getElementById(id);

let D, S, time;
const state = {tower:2, wind:11, gauge:10, sec:"top", h:"top", sortKey:"r2", asc:false, models:["tcn","prob_tcn","transformer","naive"], xr:null};

function icons(){ document.querySelectorAll(".ico").forEach(el => { el.innerHTML = ICONS[el.dataset.i] || ""; }); }
function links(){
  const btn = (key, label, icon, note) => LINKS[key]
    ? `<a href="${LINKS[key]}" target="_blank" rel="noopener" class="button is-rounded is-dark"><span class="ico">${ICONS[icon]}</span><span>${label}</span></a>`
    : `<button class="button is-rounded is-dark" disabled title="${note}"><span class="ico">${ICONS[icon]}</span><span>${label}</span></button>`;
  $("links").innerHTML = btn("paper","Paper","doc","On OpenReview") + btn("code","Code","gh","Anonymized link pending") +
    btn("data","Dataset","db","Released on publication") + btn("subset","Review subset","box","Anonymized link pending");
  document.querySelectorAll("a[data-link]").forEach(a => { const u = LINKS[a.dataset.link]; if(u){ a.href = u; a.target = "_blank"; a.rel = "noopener"; } });
}
function seg(el, items, cur, on){
  el.innerHTML = items.map(([v,t]) => `<button type="button" class="button is-small ${String(v)===String(cur)?"is-selected":""}" data-v="${v}">${t}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if(b) on(b.dataset.v); };
}

function series(t, i){
  const n = D.n, nc = D.channels.length, [lo, sc] = D.scales[`${D.towers[t]}/${D.ids[i]}`];
  const block = (t*D.ids.length + i)*nc*n, out = {};
  D.channels.forEach((c, k) => {
    const a = new Float32Array(n), off = block + k*n;
    for(let j=0;j<n;j++) a[j] = (S[off+j]+32500)*sc[k]+lo[k];
    out[c] = a;
  });
  return out;
}

function drawSeries(){
  const t = state.tower, i = state.wind, x = series(t, i), sid = D.ids[i], op = D.ops[sid];
  $("windVal").textContent = `${op.wind.toFixed(1)} m/s`;
  $("op").innerHTML = [`sim_id ${sid}`, `${D.towers[t].toUpperCase()}`, `U ${op.wind.toFixed(1)} m/s`, `Hs ${op.hs.toFixed(2)} m`, `Tp ${op.tp.toFixed(2)} s`, "seed 1", "test split"]
    .map(s => `<span class="tag is-white">${s}</span>`).join("") +
    `<span class="tag ${GROUP_TAG[op.wind_group]}">wind: ${op.wind_group}</span><span class="tag ${GROUP_TAG[op.wave_group]}">wave: ${op.wave_group}</span>`;
  const rows = [["tower_top_afa_mod","Accel. [m/s²]",NAVY],["wind_speed","Wind [m/s]",SCADA],["rotor_speed","Rotor [rpm]",SCADA],
    ["blade_pitch","Pitch [deg]",SCADA],["wave_elev","Wave [m]",CYAN],["plat_pitch","Platform pitch [deg]",CYAN]];
  const n = rows.length, gap = 0.035, h = (1 - gap*(n-1))/n, L = Object.assign({}, COMMON, {margin:{l:80,r:20,t:24,b:50}});
  const tr = rows.map(([c,name,col],k) => ({x:time, y:x[c], type:"scattergl", mode:"lines", line:{width:1, color:col}, xaxis:"x", yaxis:`y${k?k+1:""}`,
    hovertemplate:`%{x:.1f} s<br>${name} %{y:.3f}<extra></extra>`}));
  rows.forEach(([,name],k) => { L[`yaxis${k?k+1:""}`] = AX({domain:[1-(k+1)*h-k*gap, 1-k*h-k*gap], title:{text:name, font:{size:10}}, nticks:3}); });
  L.xaxis = AX({anchor:`y${n}`, title:{text:"Time [s]"}, range: state.xr || [400,1000]});
  L.annotations = [{text:"<b>Model inputs</b>", xref:"paper", yref:"paper", x:0, y:1, xanchor:"left", yanchor:"bottom", showarrow:false, font:{size:11,color:NAVY}},
    {text:"<b>Context (not inputs)</b>", xref:"paper", yref:"paper", x:0, y:2*h+gap, xanchor:"left", yanchor:"bottom", showarrow:false, font:{size:11,color:CYAN}}];
  Plotly.react("inputs", tr, L, CFG);

  const mom = D.channels.slice(8), g = state.gauge;
  const T = Object.assign({}, COMMON, {margin:{l:80,r:20,t:24,b:50}, showlegend:true,
    legend:{orientation:"h", x:1, xanchor:"right", y:0.37, yanchor:"bottom", font:{size:11}, bgcolor:"rgba(255,255,255,0.7)"}});
  T.yaxis = AX({domain:[0.45,1], title:{text:"Gauge height [m]", font:{size:10}}});
  T.yaxis2 = AX({domain:[0,0.37], title:{text:"M<sub>FA</sub> [MN m]", font:{size:10}}});
  T.xaxis = AX({anchor:"y2", title:{text:"Time [s]"}, range: state.xr || [400,1000]});
  T.annotations = [{text:"<b>Target: fore-aft moment along the tower</b>", xref:"paper", yref:"paper", x:0, y:1, xanchor:"left", yanchor:"bottom", showarrow:false, font:{size:11,color:RED}}];
  const t2 = [{type:"heatmap", x:time, y:D.heights, z:mom.map(c => Array.from(x[c])), xaxis:"x", yaxis:"y", colorscale:MOMENT_SCALE, zmid:0, showlegend:false,
      colorbar:{title:{text:"MN m", side:"right", font:{size:10}}, thickness:10, len:0.55, y:0.725, tickfont:{size:9}},
      hovertemplate:"%{x:.1f} s · %{y:.1f} m<br>%{z:.1f} MN m<extra></extra>"},
    {x:time, y:x[mom[0]], type:"scattergl", mode:"lines", line:{width:1, color:INK}, xaxis:"x", yaxis:"y2", name:"Base", hovertemplate:"Base %{y:.1f}<extra></extra>"}];
  if(g !== 0) t2.push({x:time, y:x[mom[g]], type:"scattergl", mode:"lines", line:{width:1, color:RED}, xaxis:"x", yaxis:"y2", name:GAUGE[g], hovertemplate:`${GAUGE[g]} %{y:.1f}<extra></extra>`});
  Plotly.react("targets", t2, T, CFG);
}
function syncZoom(src, dst){
  $(src).on("plotly_relayout", ev => {
    let r = null;
    if(ev["xaxis.range[0]"] !== undefined) r = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
    else if(ev["xaxis.range"]) r = ev["xaxis.range"];
    else if(ev["xaxis.autorange"]) r = [400,1000];
    if(!r || (state.xr && r[0] === state.xr[0] && r[1] === state.xr[1])) return;
    state.xr = r; Plotly.relayout(dst, {"xaxis.range": r});
  });
}

function drawProfile(){
  const P = D.profiles[`${D.towers[state.tower]}/${D.ids[state.wind]}`];
  const tr = [{x:P.true, y:D.heights, mode:"lines+markers", line:{color:"#000000", width:3}, marker:{size:7, color:"#000000"}, name:"True",
    hovertemplate:"True %{x:.2e}<br>%{y:.1f} m<extra></extra>"}];
  state.models.forEach(m => tr.push({x:P[m], y:D.heights, mode:"lines+markers", name:NAMES[m],
    line:{color:MODEL_COLOR[m], width:2, dash:MODEL_DASH[m] || "solid"}, marker:{size:5, color:MODEL_COLOR[m]},
    hovertemplate:`${NAMES[m]} %{x:.2e}<br>%{y:.1f} m<extra></extra>`}));
  const L = Object.assign({}, COMMON, {showlegend:true, legend:{orientation:"h", y:1.02, yanchor:"bottom", x:0}, margin:{l:70,r:20,t:50,b:50}});
  L.xaxis = AX({type:"log", title:{text:"Fatigue damage over 600 s (log scale)"}, exponentformat:"power"});
  L.yaxis = AX({title:{text:"Gauge height [m]"}, range:[-4,154]});
  Plotly.react("profile", tr, L, CFG);
  const top = D.heights.length - 1;
  $("profileNote").textContent = "Top gauge, reconstructed / true damage: " + state.models.map(m => `${NAMES[m]} ${(P[m][top]/P.true[top]).toFixed(2)}`).join(", ") + ". Dotted: one-gain floor.";
  $("models").querySelectorAll("label").forEach(l => {
    const m = l.dataset.m, on = state.models.includes(m);
    l.querySelector("input").checked = on;
    l.querySelector(".sw").style.background = on ? MODEL_COLOR[m] : "transparent";
  });
}
function modelPicker(){
  const order = ["tcn","prob_tcn","transformer","mamba","lstm","s4","timesnet","unet","fno","itransformer","fits","dlinear","spectral","naive"].filter(m => D.models.includes(m));
  $("models").innerHTML = order.map(m => `<label data-m="${m}"><input type="checkbox" id="m-${m}"><span class="sw"></span>${NAMES[m]}</label>`).join("");
  $("models").onchange = e => {
    const m = e.target.closest("label").dataset.m, k = state.models.indexOf(m);
    if(k >= 0) state.models.splice(k, 1); else { if(state.models.length >= 4) state.models.shift(); state.models.push(m); }
    drawProfile();
  };
}

function drawHeat(){
  const t = D.towers[state.tower], z = D.env[t][state.sec], hl = 24;
  const zt = z[0].map((_, j) => z.map(r => r[j]));
  const text = zt.map((row, j) => row.map((_, w) => { const [hs, tp] = D.waves[w+1][j], g = D.groups[w+1][j];
    return `U ${D.winds[w]} m/s · Hs ${hs} m · Tp ${tp} s<br>${g[0]} / ${g[1]} · ${g[2]}`; }));
  const L = Object.assign({}, COMMON, {margin:{l:70,r:20,t:10,b:50}});
  L.xaxis = AX({title:{text:"Wind speed [m/s]"}, tickvals:[0,3,7,11,15,19,21], ticktext:[0,3,7,11,15,19,21].map(k => D.winds[k]), showgrid:false});
  L.yaxis = AX({title:{text:"Wave state (Hs level, Tp within)"}, tickvals:[3,10,17,24,31,38,45], ticktext:["Hs 1","Hs 2","Hs 3","Hs 4","Hs 5","Hs 6","Hs 7"], showgrid:false});
  const ids = D.ids.map((_, k) => k);
  Plotly.react("heat", [
    {type:"heatmap", z:zt, text, colorscale:DAMAGE_SCALE, hovertemplate:"%{text}<br>log₁₀ damage %{z:.2f}<extra></extra>",
     colorbar:{title:{text:"log₁₀ D", side:"right", font:{size:10}}, thickness:10, tickfont:{size:9}}},
    {x:ids, y:ids.map(() => hl), mode:"markers", marker:{size:11, color:"rgba(0,0,0,0)", line:{color:INK, width:1.5}}, hovertemplate:"Review simulation<extra></extra>"},
    {x:[state.wind], y:[hl], mode:"markers", marker:{size:16, color:"rgba(0,0,0,0)", line:{color:NAVY, width:3}}, hoverinfo:"skip"}], L, CFG);
}
function drawLife(){
  const tr = D.towers.map((t, k) => ({x:D.life[t], y:D.heights, mode:"lines+markers", name:t.toUpperCase(),
    line:{color:TOWER_COLOR[t], width:k === state.tower ? 3 : 1.5}, marker:{size:k === state.tower ? 7 : 4, color:TOWER_COLOR[t]},
    hovertemplate:`${t.toUpperCase()} %{x:.3f}<br>%{y:.1f} m<extra></extra>`}));
  const L = Object.assign({}, COMMON, {showlegend:true, legend:{orientation:"h", y:1.02, yanchor:"bottom", x:0}, margin:{l:70,r:20,t:40,b:50}});
  L.xaxis = AX({title:{text:"25-year fore-aft fatigue damage"}, rangemode:"tozero"});
  L.yaxis = AX({title:{text:"Gauge height [m]"}});
  L.annotations = [{text:"Lifetime damage, all 6,468 simulations", xref:"paper", yref:"paper", x:1, y:0.02, xanchor:"right", showarrow:false, font:{size:10, color:"#7a7a7a"}}];
  Plotly.react("life", tr, L, CFG);
}

function drawBoard(){
  const h = state.h, mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const rows = D.board.map(r => { const o = {model:r.model, seeds:r.seeds};
    MET.forEach(([k]) => { const a = r[`${h}_${k}`]; o[k] = mean(a); o[k + "_w"] = k === "mre" ? Math.max(...a) : Math.min(...a); }); return o; });
  const isRef = r => r.model === "physics" || r.model === "naive";
  const learned = rows.filter(r => !isRef(r)), refs = rows.filter(isRef), key = state.sortKey;
  const val = r => key === "model" ? NAMES[r.model] : key === "median_ratio" ? -Math.abs(Math.log(r[key])) : key === "mre" ? -r[key] : r[key];
  learned.sort((a, b) => { const x = val(a), y = val(b); return (x > y ? -1 : x < y ? 1 : 0) * (state.asc ? -1 : 1); });
  const fmt = (v, k) => k === "r2" && v < -1 ? "&lt;−1" : k === "within2" ? `${Math.round(v * 100)}%` : v.toFixed(2).replace("-", "−");
  $("board").tHead.innerHTML = `<tr>${[["model","Model"]].concat(MET).map(([k, t]) => `<th data-k="${k}" class="${key === k ? "sorted" + (state.asc ? " asc" : "") : ""}">${t}</th>`).join("")}</tr>`;
  $("board").tBodies[0].innerHTML = learned.concat(refs).map(r =>
    `<tr class="${isRef(r) ? "refrow" : ""}"><td><strong>${NAMES[r.model] || r.model}</strong>${r.seeds < 3 ? " †" : ""}<span class="fam">${FAMILY[r.model] || ""}</span></td>` +
    MET.map(([k]) => `<td class="${k === "r2" && r.r2 < 0 ? "neg" : ""}">${k === "r2" ? `<span class="bar" style="width:${Math.max(0, Math.min(1, r.r2)) * 50}px"></span>` : ""}${fmt(r[k], k)} <small>${fmt(r[k + "_w"], k)}</small></td>`).join("") + "</tr>").join("");
}

function controls(){
  seg($("towerSeg"), D.towers.map((t, k) => [k, t.toUpperCase()]), state.tower, v => { state.tower = +v; controls(); drawSeries(); drawProfile(); drawHeat(); drawLife(); });
  seg($("secSeg"), [["base","Base"],["top","Top"]], state.sec, v => { state.sec = v; controls(); drawHeat(); });
  seg($("hSeg"), [["base","Base"],["upper","Gauge 8"],["top","Top"]], state.h, v => { state.h = v; controls(); drawBoard(); });
}
function selectWind(w){ state.wind = w; $("wind").value = w; drawSeries(); drawProfile(); drawHeat(); }

async function init(){
  icons(); links();
  $("copy-bibtex").onclick = () => { const txt = $("bibtex-content").textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).catch(() => {
      const r = document.createRange(); r.selectNodeContents($("bibtex-content")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }); };
  const [d, b] = await Promise.all([
    fetch("static/data/data.json").then(r => r.json()),
    fetch("static/data/series.txt").then(r => r.text()).then(t => { const bin = atob(t.trim()), u = new Uint8Array(bin.length);
      for(let j = 0; j < bin.length; j++) u[j] = bin.charCodeAt(j); return u.buffer; })]);
  D = d; S = new Int16Array(b);
  time = Float32Array.from({length:D.n}, (_, j) => D.t0 + j * D.dt);
  $("inputs").innerHTML = "";
  $("gauge").innerHTML = GAUGE.map((g, k) => `<option value="${k}">${g} (z/H ${D.zh[k].toFixed(2)})</option>`).join("");
  $("gauge").value = state.gauge; $("wind").value = state.wind;
  $("wind").oninput = e => selectWind(+e.target.value);
  $("gauge").onchange = e => { state.gauge = +e.target.value; drawSeries(); };
  $("board").onclick = e => { const th = e.target.closest("th"); if(!th) return;
    if(state.sortKey === th.dataset.k) state.asc = !state.asc; else { state.sortKey = th.dataset.k; state.asc = th.dataset.k === "model"; } drawBoard(); };
  controls(); modelPicker(); drawSeries(); drawProfile(); drawHeat(); drawLife(); drawBoard();
  syncZoom("inputs", "targets"); syncZoom("targets", "inputs");
  $("heat").on("plotly_click", ev => { const p = ev.points[0]; if(p.curveNumber > 0 || p.y === 24){ selectWind(p.x);
    $("wind").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block:"center"}); } });
}
init().catch(err => { $("inputs").innerHTML = `<p class="has-text-danger">The data files did not load (${err.message}). Reload the page to try again.</p>`; });
