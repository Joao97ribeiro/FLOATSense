<p align="center">
  <img src="docs/figures/logo.png" alt="FLOATSense" width="500"/>
</p>

# FLOATSense: A Time-Series Dataset and Benchmark for Fatigue Load Reconstruction on Floating Offshore Wind Turbine Towers
<p align="center">
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
simulations** of FLOATBench (cited in the paper)
on three 22 MW tower geometries: 37 channels per simulation, with the
accelerometer and SCADA signals as inputs and the fore-aft bending
moment at **eleven heights** as target. A reconstruction is scored by
the fatigue damage it implies, not by waveform error. For review, a fixed
subset (47 simulations per tower, one per wind speed plus all six
realizations of five operating points, the complete tabular files and the
checkpoints of 14 of the 20 learned models: all but the foundation models,
left out for size, and the two hybrids, which need the physics calibration)
is available through an anonymized link,
[https://osf.io/h54t6/?view_only=73f55c8d86214fe1b943ede5260ddf4e](https://osf.io/h54t6/?view_only=73f55c8d86214fe1b943ede5260ddf4e); the complete dataset (24.0 GB) is released on publication.
This repository contains the benchmark code, the physics baseline, the
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

- [`scripts/download/`](./scripts/download) — download the review subset
  into `data/FLOATSense` from its anonymized link, with a checksum check.
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
# download this anonymized repository and enter it
conda env create -f environment.yml
conda activate floatsense
pip install --no-deps momentfm==0.1.4   # MOMENT, only for --models=moment*
```

This installs Python 3.11, PyTorch 2.7.1 with CUDA 12.8, and the
dependencies of `requirements.txt`, including the pretrained encoders
Chronos. MOMENT is installed without its dependencies, whose pins clash
with the rest (`pip install --no-deps momentfm==0.1.4`, the last line
above). TimesFM is not included: `--models=timesfm` and
`timesfm_ft` need the TimesFM 2.5 PyTorch package installed separately
from [google-research/timesfm](https://github.com/google-research/timesfm)
(it provides `timesfm.timesfm_2p5_torch`). On first use the pretrained
encoders download their weights from the Hugging Face Hub
(`amazon/chronos-t5-small`, `AutonLab/MOMENT-1-small`,
`google/timesfm-2.5-200m-pytorch`), so that run needs internet access.

**Alternative (pip, CPU or existing venv):**

```bash
# download this anonymized repository and enter it
pip install torch  # torch==2.7.1 for the paper setting
pip install -r requirements.txt
pip install --no-deps momentfm==0.1.4   # MOMENT, only for --models=moment*
```

## Dataset

The released Parquet files, schema, channel conventions and per-tower
layout are documented in the paper and below; the review subset ships
with its own README.

The three towers are:

- `ref` — IEA-22-MW reference tower (baseline)
- `opt1` — first redesign iterate (relaxed damage budget, $D \le 1.0$)
- `opt2` — final iterate ($D \approx 0.9$, targeting $D \le 0.9$)

The `opt1` and `opt2` geometries were produced by
**FLOAT** (cited in the paper), the fatigue-aware
tower design-optimization framework that the `ref` tower is redesigned with.

<p align="center">
  <img src="docs/figures/channels.png" alt="The 37 channels of one simulation" width="800"/>
</p>

```
FLOATSense/                          24.0 GB
├── ref/
│   ├── series-<k>-of-00017.parquet   sim_id, time_s, 37 channels; 400 simulations per shard, one row group each
│   │                                 (the review subset has one shard, series-00000-of-00001.parquet)
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

### Download (review subset)

One command downloads the review subset from its anonymized link, checks
its SHA-256 and unpacks it into `data/FLOATSense` (release layout, 47
simulations per tower: one per wind speed, `sim_id = 25 + 294 k`,
k = 0..21, and the realizations 2 to 6 of the operating points at 5.5,
10.5, 14.5, 21.5 and 24.5 m/s; `conditions.csv` lists them):

```bash
python scripts/download/run.py --flagfile=scripts/download/config.cfg

# read one simulation from Python
python -c "from floatsense import load_tower; \
  t = load_tower('data/FLOATSense', 'opt2'); print(t.load(25).shape, t.channels[:4])"
```

(Manual alternative: download `FLOATSense-review.zip` from
[https://osf.io/h54t6/?view_only=73f55c8d86214fe1b943ede5260ddf4e](https://osf.io/h54t6/?view_only=73f55c8d86214fe1b943ede5260ddf4e), unzip it, and
`mkdir -p data && mv FLOATSense-review data/FLOATSense`.) The subset also holds the
trained checkpoints, the reference per-simulation results of the paper,
`compare.py`, which compares an evaluation run with them, and `rho_wc.py`,
which computes the within-condition correlation on the five operating
points with six realizations.

Check the paper's per-simulation results with the released checkpoints
(evaluation only, CPU, a few minutes):

```bash
mkdir -p outputs/review/opt2 && cp data/FLOATSense/checkpoints/opt2/seed0/*.pt outputs/review/opt2/
python scripts/train/run.py --flagfile=scripts/train/config.cfg \
    --tower=opt2 --test_split=review/test --run_training=False \
    --models=tcn,mamba,naive --output_dir=outputs/review/opt2
python data/FLOATSense/compare.py outputs/review/opt2 opt2
python data/FLOATSense/rho_wc.py outputs/review/opt2 opt2

# the metrics table of these runs, with condition-level bootstrap intervals
python scripts/benchmark/run.py --flagfile=scripts/benchmark/config.cfg \
    --output_root=outputs/review --test_split=review/test
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

The commands below need the full dataset: the physics calibration and
the training read the train split, which the review subset does not ship
(its simulations are listed in `splits/review/test`; see *Download (review
subset)* for what runs on it).

```bash
# Physics baseline on one tower (CPU, ~3-4 min): calibrate on train, score test;
# the log ends with base and top R2, all heights come from scripts/benchmark/run.py
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=opt2

# One learned model on one tower, scored zero-shot on the other two (~10–12 min on one GPU)
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
likelihood. The hybrid models read the physics calibrated on the same
split (`outputs/physics/<tower>` for `train`, `<tower>_fs10_draw0` for
`fewshot/train_10_draw0`), so run the physics on that split first.

Training is deterministic by default (`--deterministic=True`): a rerun
of a seed on the same GPU type gives identical numbers. It makes cuDNN
training about three times slower (TCN: ~7 instead of ~2.5 min of
training); `--deterministic=False` is faster and is how the paper runs
were trained (see [Reproducibility](#reproducibility)).

### Hardware & runtime

The benchmarks were run on a single workstation; nothing in the
pipeline assumes a cluster.

| Resource | Paper setting | Notes |
| --- | --- | --- |
| GPU | 2 × NVIDIA RTX 4090 (24 GB), one run per GPU | Small models fit in a few GB; the pretrained encoders need more. |
| CPU | Intel Core i9-14900K (24 cores) | Rainflow counting of the evaluation runs in a process pool. |
| RAM | 128 GB available | One run reads one simulation at a time from the shards. |
| Disk | 24.0 GB dataset + a few MB per run | One checkpoint per model, tower and seed. |
| Wall-clock | 4–9 min per run for the small models (training and scoring), ~10–12 min with zero-shot on two towers | 10–60 min for the pretrained encoders, 142 min for Mamba; physics ~3–4 min per tower on CPU. |

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
│   └── summary_<model>_fa[_zs_<t>].json          R², median ratio and within-2 at the base (tower_bottom)
├── fewshot/<source>_to_<target>/draw<k>/seed<s>/   adaptation runs (same files as within/)
├── ablation/<input set>/<tower>/seed<k>/           sensor-ablation runs (same files as within/)
├── val_select/<tower>/seed<k>/             model-selection runs scored on val/val
└── tables/results.csv                 every file x gauge x regime cell: metrics, 95% intervals,
                                       direction and num_dropped (simulations without a positive
                                       damage, left out of the metrics as in the paper)
```

Zero-shot physics runs (`<tower>_zs_<source>`) reuse the calibration of
the source tower, so they hold only `profile.json` and
`damage_heights.csv`.

### Protocols

Full dataset only, like the Quickstart.

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
    --output_dir=outputs/fewshot/opt2_to_ref/draw0/seed0

# Physics ten-shot (constants recalibrated on 10 ref simulations; the paper's
# cross-tower physics) -> outputs/physics/ref_fs10_draw0
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=ref \
    --train_split=fewshot/train_10_draw0

# Physics zero-shot, all constants (C1 included) and height profile of opt2
# applied to ref -> outputs/physics/ref_zs_opt2 (a diagnostic, not in the paper)
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=ref --source=opt2

# Sensor ablation: both accelerometer axes and SCADA
python scripts/train/run.py --flagfile=scripts/train/config.cfg --tower=opt2 --models=tcn \
    --input_channels=tower_top_afa_mod,tower_top_ass_mod,rotor_speed,blade_pitch,wind_speed \
    --output_root=outputs/ablation/twoaxis

# Model selection on the held-out validation split (scored on val/val, which
# has no regime cells: the benchmark reports it under the 'all' group only)
python scripts/train/run.py --flagfile=scripts/train/config.cfg --tower=opt2 --models=tcn \
    --train_split=val/train --test_split=val/val --output_root=outputs/val_select
```

`--max_train_sims` / `--max_eval_sims` cap a run for a quick test; they
take the first simulations by `sim_id`, one realization per operating
point, so the within-condition correlation of such a run is undefined.
Write quick tests to a separate `--output_root` (or delete them): the
benchmark table scores every run it finds under `outputs/`.

Field-style SCADA (the ten-minute statistics a turbine logs) is an
input set too: `stat:<channel>:<mean|std|min|max>` reads
`series_stats.parquet` and feeds the value as a constant channel.

### Flags of every experiment in the paper

All runs add `--flagfile=scripts/train/config.cfg --tower=<tower> --models=<model> --seed=<k>`
(seeds 0, 1, 2) to the flags below; Mamba always adds `--learning_rate=3e-4`.
`$SCADA` is `rotor_speed,blade_pitch,wind_speed`. Each experiment writes
to its own folder, `<output root>/<tower>/seed<k>` (the last column), so
runs never overwrite each other and the benchmark tells them apart.

| Experiment | Extra flags | Output |
| --- | --- | --- |
| Within tower (+ zero-shot) | `--eval_towers=<the other two>` | default `outputs/within` |
| Longer budgets | `--num_epochs=150` (or 300) | `--output_root=outputs/epochs150` |
| Ten-shot | `--train_split=fewshot/train_10_draw<k> --batch_size=4 --init_checkpoint_dir=outputs/within/<source>/seed0` | `--output_dir=outputs/fewshot/<source>_to_<tower>/draw<k>/seed<s>` |
| Few-shot budget curve (5 to 100 simulations: 50 to 1,250 steps) | as ten-shot with `--train_split=fewshot/train_<n>_draw0`, n = 5, 10, 25, 50, 100 | `--output_dir=outputs/budget/<source>_to_<tower>/<n>/seed<s>` |
| Accelerometer alone | `--input_channels=tower_top_afa_mod` | `--output_root=outputs/ablation/accel` |
| SCADA alone | `--input_channels=$SCADA` | `--output_root=outputs/ablation/scada` |
| Two accelerometer axes | `--input_channels=tower_top_afa_mod,tower_top_ass_mod,$SCADA` | `--output_root=outputs/ablation/twoaxis` |
| + platform motions | `--input_channels=tower_top_afa_mod,$SCADA,plat_surge,plat_sway,plat_heave,plat_roll,plat_pitch,plat_yaw` | `--output_root=outputs/ablation/platform` |
| + generator power | `--input_channels=tower_top_afa_mod,$SCADA,electrical_power` (two axes: add `tower_top_ass_mod` after the first) | `--output_root=outputs/ablation/power` |
| Field SCADA | `--input_channels=tower_top_afa_mod,stat:rotor_speed:mean,stat:rotor_speed:std,stat:blade_pitch:mean,stat:blade_pitch:std,stat:wind_speed:mean,stat:wind_speed:std` | `--output_root=outputs/ablation/fieldscada` |
| Model selection | `--train_split=val/train --test_split=val/val` (hybrids: physics run first with `--train_split=val/train`) | `--output_root=outputs/val_select` |
| Top only (diagnostic; not for the hybrids) | `--height_targets=False --target_channel=tower_top_mfa` | `--output_root=outputs/toponly` |
| Damage-aware loss (diagnostic) | `--loss=damage --damage_loss_weight=1.0` | `--output_root=outputs/damageloss` |

The hybrid models need the physics of the same split first
(`scripts/physics/run.py --tower=<tower> --train_split=<split>`); the
paper runs were trained with `--deterministic=False`.

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
# with the review subset: pd.read_csv("outputs/review/opt2/damage_comparison_tcn_fa.csv")
tower = load_tower("data/FLOATSense", "opt2")
meta = tower.metadata.loc[df.sim_id]
ci = cluster_bootstrap(df.damage_true_tower_top.values,
                       df.damage_rec_tower_top.values,
                       condition_key(meta).values)
print(ci)
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
- The paper runs were trained with `--deterministic=False`, where cuDNN
  convolutions are not bit-for-bit repeatable: in our reruns LSTM,
  PatchTST and the hybrid repeated the paper exactly, while TCN and
  Prob-TCN drifted within the spread of the seeds
  (Prob-TCN top $R^2$ on `opt1`: 0.80 and 0.84 in two reruns of seed 0,
  0.85 to 0.87 over the paper seeds). TCN on `opt2`, base $R^2$: 0.984, 0.991 and 0.992 in
  three reruns of seed 0, 0.987 to 0.992 over the three seeds of the
  paper; top $R^2$: 0.754 to 0.785 in reruns, 0.713 to 0.866 over the
  seeds. Compare a model with the paper through its three-seed median.
  With the default `--deterministic=True`, two runs of a seed are
  identical to each other (TCN on `opt2`: zero difference over the 4,740
  test simulations), though not to the paper's convolutional runs.

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

Code released under the [MIT License](LICENSE.txt). The dataset is
released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).


## Acknowledgements

The high-fidelity simulations underlying FLOATSense were produced
with [OpenFAST](https://github.com/OpenFAST/openfast) on the
[IEA-22-280-RWT](https://github.com/IEAWindSystems/IEA-22-280-RWT)
reference floating wind turbine. The physics baseline follows
Pimenta et al. (2024), *Renewable Energy* 223, 119981. We thank these
communities for keeping the underlying tools open.
