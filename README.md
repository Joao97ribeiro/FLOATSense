<p align="center">
  <img src="docs/figures/logo.png" alt="FLOATSense" width="500"/>
</p>

# FLOATSense: A Time-Series Dataset and Benchmark for Fatigue Load Reconstruction on Floating Offshore Wind Turbine Towers
<p align="center">
  <a href="https://huggingface.co/datasets/DeCoDELab/FLOATSense">
    <img src="https://img.shields.io/badge/dataset-DeCoDELab%2FFLOATSense-ffcc00.svg?logo=huggingface&logoColor=white">
  </a>
  <a href="https://github.com/Joao97ribeiro/FLOATBench">
    <img src="https://img.shields.io/badge/builds%20on-FLOATBench-2c5282.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/code--license-MIT-blue.svg">
  </a>
  <a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://img.shields.io/badge/data--license-CC--BY--4.0-blue.svg">
  </a>
</p>

<p align="center"><strong>Reconstruct the tower bending moment of a 22 MW floating wind turbine from a tower-top accelerometer and SCADA, and score it by the fatigue damage it implies.</strong></p>

**FLOATSense** is a public dataset and benchmark for fatigue load
reconstruction on floating offshore wind turbine (FOWT) towers. It
releases **717,948 time series at 10 Hz** from the **19,404 OpenFAST
simulations** of [FLOATBench](https://github.com/Joao97ribeiro/FLOATBench)
on three 22 MW tower geometries: 37 channels per simulation, with the
accelerometer and SCADA signals as inputs and the fore-aft bending
moment at **eleven heights** as target. A reconstruction is scored by
the fatigue damage it implies, not by waveform error. The dataset is
hosted on Hugging Face at
[`DeCoDELab/FLOATSense`](https://huggingface.co/datasets/DeCoDELab/FLOATSense);
this repository contains the benchmark code, the physics baseline, the
20 learned models, the evaluation harness, and the scripts to reproduce
the paper results.

<p align="center">
  <img src="docs/figures/overview.png" alt="FLOATSense overview" width="800"/>
</p>


## FLOATSense Paper

**FLOATSense** is presented in the paper *FLOATSense: A Time-Series
Dataset and Benchmark for Fatigue Load Reconstruction on Floating
Offshore Wind Turbine Towers*, which fully describes the dataset, the
task, the evaluation protocols and the results.

Twenty learned models in eight families and a field-validated physics
baseline are evaluated within tower, across towers zero-shot and
few-shot, and under sensor ablation. **Sensor-level leaderboards
mislead**: the base of the tower does not separate the models and its
leaders collapse at the top, where at the declared budget only three
learned models beat a one-gain floor; longer training recovers the top
for convolutional models, not for attention and state-space ones; five
simulations of a new tower recover its base, not its top; and a second
accelerometer axis improves the top at every budget.


## What FLOATSense Provides

- **Dataset.** 6,468 simulations per tower (1,078 operating points on
  a $22 \times 7 \times 7$ wind/wave envelope $\times$ 6 turbulence
  seeds) on three 22 MW FOWT towers (`ref`, `opt1`, `opt2`), 37
  channels at 10 Hz over 1,000 s: tower-top accelerations, SCADA,
  platform motions, wave elevation, and fore-aft and side-side bending
  moments at 11 heights. Plus 22 parked runs per tower and per-window
  channel statistics. Every file is keyed by the FLOATBench `sim_id`.
- **Task.** From the gravity-corrected fore-aft tower-top acceleration,
  rotor speed, blade pitch, hub wind speed and a height $z/H$,
  reconstruct the fore-aft moment at that height over 400 to 1,000 s.
  One model serves the whole tower.
- **Damage-based scoring.** True and reconstructed moments pass through
  the pipeline that generated the FLOATBench labels (3 Hz low-pass,
  rainflow, DNV-RP-C203 S-N curve, Miner's rule). Five metrics: $R^2$
  of $\log_{10}$ damage, median damage ratio, fraction within a factor
  of two, mean relative error, and within-condition correlation over
  the six realizations of an operating point.
- **Benchmark protocols.** Within tower on the regime-aware partition
  of FLOATBench (In-train / Interpolate / Extrapolate on the wind and
  wave axes), cross tower (zero-shot and few-shot), and sensor ablation.
- **Baselines and models.** The band-gain physics baseline of Pimenta
  et al. (2024) with parked calibration and a height factor, a one-gain
  floor, and 20 learned models in eight families (convolutional,
  recurrent, attention, state-space, spectral, period-folding,
  pretrained encoders, physics-anchored), all trained with one recipe.
- **Reproducible harness.** CLI scripts for training, physics
  calibration and the results table with cluster-bootstrap confidence
  intervals, all driven by `--flagfile` configs.

> **Metrics.** Throughout, $D$ is the Miner fatigue damage of one
> simulation at one height. The headline metric is $R^2$ of
> $\log_{10} D$, reported per height, per regime cell and per tower.

## What's in this repo

```
floatsense/        Python package (data reader, models, training, physics baseline, metrics)
scripts/           Pipeline entry points — see "Scripts" below
splits/            Model-selection and few-shot splits (lists of sim_id, same on every tower)
towers/            ElastoDyn tower mass density of each tower (physics height factor)
docs/              Figures used in this README
environment.yml    Conda environment (Python 3.11 + GPU PyTorch)
requirements.txt   Pinned runtime dependencies
```

Inside `floatsense/`:

```
release.py     reader of the released Parquet dataset (shards, metadata, sections, splits, parked runs)
data.py        torch dataset of (inputs, condition, target) windows
models.py      the floor and the 20 learned models
trainer.py     training recipe and damage evaluation
physics.py     band-gain physics baseline and its parked calibration
heights.py     physics height factor along the tower
fatigue.py     rainflow, S-N curve and Miner's rule
metrics.py     damage metrics and cluster bootstrap
```

## Scripts

Each folder under [`scripts/`](./scripts) is a pipeline stage; the
three benchmark stages have their own `run.py` and a `--flagfile`
`config.cfg`. Run every command from the repository root (the scripts
add it to the Python path; for the Python snippets below, run them from
the root or `export PYTHONPATH=$PWD`):

- [`scripts/physics/`](./scripts/physics) — calibrate the physics
  baseline on a tower (C1 from the parked runs, the other five constants
  from the training split) and score it at the 11 gauges; zero-shot with
  `--source`, few-shot with `--train_split=fewshot/...`.
- [`scripts/train/`](./scripts/train) — train one or more learned
  models on a tower and score them on its test split, and zero-shot on
  other towers (`--eval_towers`); few-shot adaptation from a checkpoint
  (`--init_checkpoint_dir`); sensor ablation (`--input_channels`).
- [`scripts/benchmark/`](./scripts/benchmark) — score every run under
  an output tree: one long table per file, gauge and regime cell, with
  cluster-bootstrap 95% intervals. The scored tower is read from the path
  (`ref`, `opt1` or `opt2` in a directory name, `_zs_<tower>` or
  `<source>_to_<target>`); files without one are skipped with a warning.
- [`scripts/data/build_labels.py`](./scripts/data/build_labels.py) —
  build the label files of the release (`metadata`, `sections`,
  `damage`) from the series and the FLOATBench dataset (maintainers
  only; the released files already contain them).

## Install

**Recommended (conda, GPU):**

```bash
git clone https://github.com/Joao97ribeiro/FLOATSense
cd FLOATSense
conda env create -f environment.yml
conda activate floatsense
```

This installs Python 3.11, PyTorch 2.7.1 with CUDA 12.8, and the
dependencies of `requirements.txt`, including the pretrained encoders
Chronos and MOMENT. TimesFM is not included: `--models=timesfm` and
`timesfm_ft` need the TimesFM 2.5 PyTorch package installed separately
from [google-research/timesfm](https://github.com/google-research/timesfm)
(it provides `timesfm.timesfm_2p5_torch`).

**Alternative (pip, CPU or existing venv):**

```bash
git clone https://github.com/Joao97ribeiro/FLOATSense
cd FLOATSense
pip install torch  # torch==2.7.1 for the paper setting
pip install -r requirements.txt
```

## Dataset

The released Parquet files, schema, channel conventions and per-tower
layout are documented in the dataset README on Hugging Face:
[`DeCoDELab/FLOATSense`](https://huggingface.co/datasets/DeCoDELab/FLOATSense).

The three towers are:

- `ref` — IEA-22-MW reference tower (baseline)
- `opt1` — first redesign iterate (relaxed damage budget, $D \le 1.0$)
- `opt2` — final iterate ($D \approx 0.9$, targeting $D \le 0.9$)

The `opt1` and `opt2` geometries were produced by
[**FLOAT**](https://github.com/Joao97ribeiro/FLOAT), the fatigue-aware
tower design-optimization framework that the `ref` tower is redesigned with.

<p align="center">
  <img src="docs/figures/channels.png" alt="The 37 channels of one simulation" width="800"/>
</p>

```
FLOATSense/                          24.0 GB
├── ref/
│   ├── series-<k>-of-00017.parquet   sim_id, time_s, 37 channels; 400 simulations per shard, one row group each
│   ├── series_stats.parquet          mean, std, min, max of every channel over 400-1,000 s
│   ├── metadata.parquet              sim_id, wind_speed_id, wind_speed, mean_wind_speed, std_wind_speed,
│   │                                 wave_hs_id, wave_hs, wave_tp_id, wave_tp, wind_seed_id,
│   │                                 split, wind_group, wave_group, damage_weight
│   ├── sections.parquet              section_id, section_height_m, section_radius_m, section_thickness_m,
│   │                                 channel, gauge_height_m, z_over_h (the 11 scored sections)
│   └── damage.parquet                sim_id, section_id, damage (reference fore-aft damage)
├── opt1/                             same files
├── opt2/                             same files
├── parked.parquet                    the 22 parked runs (waves only) of each tower
├── assets/                           figures of the dataset card
└── README.md                         dataset card (schema, channels, OpenFAST mapping)
```

Column names and order follow FLOATBench, so a run joins its FLOATBench
rows on `sim_id` (and `section_id`). The 11 gauges are FLOATBench
sections 1, 3, 6, ..., 27, 30; each is scored with the mean outer radius
and wall thickness of that section.

### Download

```bash
# Option A: download with the HF CLI (one-time)
hf download DeCoDELab/FLOATSense --repo-type=dataset --local-dir=data/FLOATSense

# Option B: read one simulation from Python
python -c "from floatsense import load_tower; \
  t = load_tower('data/FLOATSense', 'opt2'); print(t.load(1).shape, t.channels[:4])"
```

The configs expect the dataset at `data/FLOATSense`; change
`--dataset_dir` otherwise.

**Lifetime weights.** `damage_weight` is the expected number of 600 s
windows a simulation represents over the 25-year service life; the
weights of the 6,468 simulations of a tower sum to 1,314,000, as in
FLOATBench. Lifetime damage at a section is
$\sum_i$ `damage_i * damage_weight_i`. The benchmark metrics use the
unweighted per-simulation damage.

## Quickstart

```bash
# Physics baseline on one tower (CPU, ~3 min): calibrate on train, score test
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=opt2

# One learned model on one tower, scored zero-shot on the other two (~10 min on one GPU)
python scripts/train/run.py --flagfile=scripts/train/config.cfg \
    --tower=opt2 --models=tcn --eval_towers=ref,opt1

# Results table of every run under outputs/
python scripts/benchmark/run.py --flagfile=scripts/benchmark/config.cfg
```

Model names (`--models`): `naive` (floor), `dlinear`, `tcn`, `unet`,
`lstm`, `transformer` (PatchTST), `itransformer`, `s4` (S4D), `mamba`,
`spectral`, `fits`, `fno`, `timesnet`, `chronos`, `moment`, `moment_ft`,
`timesfm`, `timesfm_ft`, `hybrid`, `hybrid_tcn`, `prob_tcn`.

**Training recipe**, identical for every model: 50 epochs of Adam at
$10^{-3}$, batch 16, mean squared error on the standardized moment,
4,096-sample crops (full window for the length-fixed models), no weight
decay, scheduler or early stopping, last epoch scored. Exceptions:
Mamba uses `--learning_rate=3e-4`; Prob-TCN trains on the Gaussian
likelihood. The hybrid models read the physics calibration of their
tower (`outputs/physics/<tower>`), so run the physics first.

### Hardware & runtime

The benchmarks were run on a single workstation; nothing in the
pipeline assumes a cluster.

| Resource | Paper setting | Notes |
| --- | --- | --- |
| GPU | 2 × NVIDIA RTX 4090 (24 GB), one run per GPU | Small models fit in a few GB; the pretrained encoders need more. |
| CPU | Intel Core i9-14900K (24 cores) | Rainflow counting of the evaluation runs in a process pool. |
| RAM | 128 GB available | One run reads one simulation at a time from the shards. |
| Disk | 24.0 GB dataset + a few MB per run | One checkpoint per model, tower and seed. |
| Wall-clock | 7–9 min per run for the small models (training and scoring), ~10 min with zero-shot on two towers | 10–60 min for the pretrained encoders, 142 min for Mamba; physics ~2–3 min per tower on CPU. |

### What lands in `outputs/`

```
outputs/
├── physics/<tower>[_<tag>]/
│   ├── calibration_fa.json            C1..C5, C_d and the transition frequency
│   ├── estimates_fa.csv               per-simulation estimates behind the medians
│   ├── profile.json                   height factor at the 11 gauges and the parked profile
│   └── damage_heights.csv             true and reconstructed damage at the 11 gauges, per test simulation
├── within/<tower>/seed<k>/
│   ├── <model>_fa.pt                  weights, normalization statistics, settings
│   ├── history_<model>_fa.json        training loss per epoch
│   ├── damage_comparison_<model>_fa.csv          true and reconstructed damage (and variance ratio) at the 11 gauges
│   ├── damage_comparison_<model>_fa_zs_<t>.csv   the same, zero-shot on tower <t>
│   └── summary_<model>_fa[_zs_<t>].json          base-height metrics
├── fewshot/<source>_to_<target>/draw<k>/    adaptation runs (same files as within/)
├── ablation/<input set>/<tower>/           sensor-ablation runs (same files as within/)
├── val_select/<tower>/seed<k>/             model-selection runs scored on val/val
└── tables/results.csv                 every file x gauge x regime cell: metrics and 95% intervals
```

Zero-shot physics runs (`<tower>_zs_<source>`) reuse the calibration of
the source tower, so they hold only `profile.json` and
`damage_heights.csv`.

### Protocols

```bash
# Within tower, three seeds
for s in 0 1 2; do
  python scripts/train/run.py --flagfile=scripts/train/config.cfg \
      --tower=opt1 --models=tcn --seed=$s --eval_towers=ref,opt2
done

# Cross tower, ten-shot: adapt the opt2 model on 10 ref simulations (100 steps)
python scripts/train/run.py --flagfile=scripts/train/config.cfg \
    --tower=ref --models=tcn --batch_size=4 \
    --train_split=fewshot/train_10_draw0 \
    --init_checkpoint_dir=outputs/within/opt2/seed0 \
    --output_dir=outputs/fewshot/opt2_to_ref/draw0

# Physics zero-shot (constants and height profile of opt2 on ref) and ten-shot
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=ref --source=opt2
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=ref \
    --train_split=fewshot/train_10_draw0 --tag=fs10_draw0

# Sensor ablation: both accelerometer axes and SCADA
python scripts/train/run.py --flagfile=scripts/train/config.cfg --tower=opt2 --models=tcn \
    --input_channels=tower_top_afa_mod,tower_top_ass_mod,rotor_speed,blade_pitch,wind_speed \
    --output_dir=outputs/ablation/twoaxis/opt2

# Model selection on the held-out validation split (scored on val/val, which
# has no regime cells: the benchmark reports it under the 'all' group only)
python scripts/train/run.py --flagfile=scripts/train/config.cfg --tower=opt2 --models=tcn \
    --train_split=val/train --test_split=val/val --output_root=outputs/val_select
```

`--max_train_sims` / `--max_eval_sims` cap a run for a quick test; they
take the first simulations by `sim_id`, one realization per operating
point, so the within-condition correlation of such a run is undefined.

Field-style SCADA (the ten-minute statistics a turbine logs) is an
input set too: `stat:<channel>:<mean|std|min|max>` reads
`series_stats.parquet` and feeds the value as a constant channel.

### Bootstrap confidence intervals

The confidence intervals are percentile-bootstrap intervals
($B = 1{,}000$, 95%) whose resampling unit is the **operating
condition** (wind speed, $H_s$, $T_p$): each replicate draws the test
conditions with replacement and keeps all six realizations of each
drawn condition. The six realizations share their met-ocean state, so
resampling simulations one by one would understate the spread.

```python
import pandas as pd
from floatsense import cluster_bootstrap, load_tower
from floatsense.metrics import condition_key

df = pd.read_csv("outputs/within/opt2/seed0/damage_comparison_tcn_fa.csv")
tower = load_tower("data/FLOATSense", "opt2")
meta = tower.metadata.loc[df.sim_id]
ci = cluster_bootstrap(df.damage_true_tower_top.values,
                       df.damage_rec_tower_top.values,
                       condition_key(meta).values)
```

## Reproducibility

- The reference damage in `damage.parquet` is computed on the even
  6,000-sample window 400.0–999.9 s, as in the evaluation of the learned
  models, and matches it to $10^{-6}$. The physics baseline scores its
  own truth on the inclusive 400–1,000 s window (6,001 samples), as in
  the paper; the two true damages differ by a median of $2 \times 10^{-4}$
  (99th percentile about 1%, up to about 10% on a few low-damage
  simulations).
- The parked constant C1 is recomputed from `parked.parquet` and
  rounded to 0.1 MN s², as in the paper; the physics baseline reproduces
  the paper to $10^{-7}$.
- Training on GPU is not bit-for-bit deterministic (cuDNN), so rerunning
  the same seed does not give the same numbers; the spread of reruns is
  that of the seeds. TCN on `opt2`, base $R^2$: 0.984, 0.991 and 0.992 in
  three reruns of seed 0, 0.987 to 0.992 over the three seeds of the
  paper; top $R^2$: 0.754 to 0.785 in reruns, 0.713 to 0.866 over the
  seeds. Compare a model with the paper through its three-seed median.

## Headline findings

**Within tower: the base does not separate the models, the top does.**
Ten of the 22 entries reach $R^2 \ge 0.95$ at the base and the first
seven lie within 0.023. At the top the same entries spread from $-2.3$
to 0.84 and the base ranking does not predict it: PatchTST and Mamba,
first at the base, fall below zero at the top, while TCN keeps
0.73–0.78 and Prob-TCN 0.82–0.86. Only three learned models beat the
one-gain floor at the top at the declared budget.

<p align="center">
  <img src="docs/figures/results.png" alt="Base vs top, regimes, few-shot and sensors" width="800"/>
</p>

**Over the design life the top governs the redesigns, and it is where
reconstruction fails.** On `opt1` the top carries 1.5 times the
lifetime damage of the base; PatchTST and Mamba keep 1 to 8% of it,
TCN 61 to 66%, and Prob-TCN over-predicts by 44 to 74%.

<p align="center">
  <img src="docs/figures/lifetime_damage.png" alt="Lifetime damage along the tower" width="800"/>
</p>

**Cross tower: zero-shot keeps the ranking, not the level.** The
within-condition correlation stays at 0.84 to 0.95, so realizations are
ranked right and placed at the wrong level. Five target simulations
recover the base on every pair, into `ref` included; no adaptation
measured recovers the top.

**Sensors: the second accelerometer axis helps the top at every
budget**, while platform motions add little beyond it.

## License

Code released under the [MIT License](LICENSE.txt). Dataset on
Hugging Face is released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).


## Citation

If you use **FLOATSense** in your work, please cite the paper (reference
to be added on publication) and FLOATBench, whose simulation campaign it
builds on:

> *FLOATBench: A Dataset and Benchmark for Floating Offshore Wind
> Turbine Tower Fatigue.*
> João Alves Ribeiro, Bruno Alves Ribeiro, Francisco Pimenta,
> Sérgio M. O. Tavares, Faez Ahmed. arXiv:2605.25717, 2026.
> https://arxiv.org/abs/2605.25717


## Maintenance & Support

For issues, questions, or feature requests related to FLOATSense:
[FLOATSense Issues](https://github.com/Joao97ribeiro/FLOATSense/issues).


## Acknowledgements

The high-fidelity simulations underlying FLOATSense were produced
with [OpenFAST](https://github.com/OpenFAST/openfast) on the
[IEA-22-280-RWT](https://github.com/IEAWindSystems/IEA-22-280-RWT)
reference floating wind turbine. The physics baseline follows
Pimenta et al. (2024), *Renewable Energy* 223, 119981. We thank these
communities for keeping the underlying tools open.
