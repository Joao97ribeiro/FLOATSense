# FLOATSense: A Time-Series Dataset and Benchmark for Fatigue Load Reconstruction on Floating Offshore Wind Turbines

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/code--license-MIT-blue.svg">
  </a>
  <a href="https://creativecommons.org/licenses/by/4.0/">
    <img src="https://img.shields.io/badge/data--license-CC--BY--4.0-blue.svg">
  </a>
</p>

<p align="center"><strong>Reconstruct the tower bending moment of a 22 MW floating wind turbine from a tower-top accelerometer and SCADA, and score it by the fatigue damage it implies.</strong></p>

**FLOATSense** builds on the simulation campaign of
[FLOATBench](https://github.com/Joao97ribeiro/FLOATBench): 19,404 OpenFAST
simulations of the IEA 22 MW turbine on a semi-submersible, three tower
geometries (`ref`, `opt1`, `opt2`) with 6,468 simulations each (1,078
operating points x six realizations). FLOATBench released the fatigue labels;
FLOATSense releases the synchronized sensor and load time series, 37
channels at 10 Hz over 1,000 s, and a benchmark for load reconstruction.
Every file is keyed by the FLOATBench `sim_id`, the same on the three
towers.

This repository holds the benchmark code: data reader, the physics baseline
of Pimenta et al. (2024), the floor and 20 learned models in eight families,
training, damage scoring and the results tables.

## Task

Given the gravity-corrected fore-aft tower-top acceleration, three SCADA
channels (rotor speed, blade pitch, hub wind speed) and a requested height
`z/H`, reconstruct the fore-aft bending-moment history at that height over
the scored window, 400 to 1,000 s. One model serves the whole tower with
`z/H` as a constant input channel: training samples one of the 11 gauges per
example, testing scores all 11.

The score is fatigue damage, not waveform error. The true and the
reconstructed moment pass through the pipeline that generated the
FLOATBench labels: a 3 Hz low-pass, rainflow counting, the DNV-RP-C203
bilinear S-N curve (log10 intercepts 12.010 and 15.350, slopes 3 and 5,
thickness correction above 25 mm) and Miner's rule. Each gauge uses the
mean outer radius and wall thickness of the nearest of the 30 FLOATBench
sections (1, 3, 6, ..., 27, 30).

Metrics (`floatsense/metrics.py`): R^2 of log10 damage (ranking metric),
median ratio reconstructed/true, fraction within a factor of two, mean
relative error and within-condition correlation over the six realizations
of each operating point. Confidence intervals resample operating points,
not simulations (cluster bootstrap, B = 1,000).

## Protocols

| Protocol | What changes | How |
|---|---|---|
| Within tower | wind and wave regime | train on `train` (1,728 simulations, 288 operating points), score `test` (4,740) per regime cell (in-train / interpolate / extrapolate on each axis) |
| Cross tower | tower design | zero-shot (`--eval_towers`), ten-shot adaptation (`--train_split=fewshot/train_10_draw<k>`, `--init_checkpoint_dir`) |
| Sensor ablation | installed sensors | `--input_channels` (accelerometer alone, SCADA alone, both axes, platform motions) |

## What's in this repo

```
floatsense/
  release.py     reader of the released dataset (Parquet shards, splits, parked runs)
  data.py        torch dataset of (inputs, condition, target) windows
  models.py      the floor and the 20 learned models
  trainer.py     training recipe and damage evaluation
  physics.py     band-gain physics baseline and its parked calibration
  heights.py     physics height factor along the tower
  fatigue.py     rainflow, S-N curve and Miner's rule
  metrics.py     damage metrics and cluster bootstrap
scripts/
  train/         train and score learned models            run.py + config.cfg
  physics/       calibrate and score the physics baseline  run.py + config.cfg
  benchmark/     results table from every run              run.py + config.cfg
  data/          build the label files of the release      build_labels.py
splits/
  val/           model-selection split (1,380 train / 348 val simulations)
  fewshot/       adaptation sets, 5 to 100 simulations, three draws of 5 and 10
```

The `train` / `test` partition is the regime-aware partition of FLOATBench
and is read from `metadata.parquet`; the validation and few-shot splits are
lists of `sim_id`, the same on the three towers.

## Install

```bash
conda env create -f environment.yml
conda activate floatsense
```

## Dataset

The dataset (24.0 GB) has one folder per tower and one shared parked file.
Column names and order follow FLOATBench.

```
FLOATSense/
├── ref/
│   ├── series-<k>-of-00017.parquet   sim_id, time_s, 37 channels; 400 simulations per shard, one row group each
│   ├── series_stats.parquet          mean, std, min, max of every channel over 400-1,000 s
│   ├── metadata.parquet              sim_id, wind_speed_id, wind_speed, mean_wind_speed, std_wind_speed,
│   │                                 wave_hs_id, wave_hs, wave_tp_id, wave_tp, wind_seed_id,
│   │                                 split, wind_group, wave_group, damage_weight
│   ├── sections.parquet              section_id, section_height_m, section_radius_m, section_thickness_m,
│   │                                 channel, gauge_height_m, z_over_h (11 scored sections)
│   └── damage.parquet                sim_id, section_id, damage (reference fore-aft damage)
├── opt1/                             same files
├── opt2/                             same files
└── parked.parquet                    the 22 parked runs (waves only) of each tower
```

The scripts expect the dataset at `data/FLOATSense` and FLOATBench at
`data/FLOATBench` (the physics height factor reads the tower mass from the
30 FLOATBench sections); change `--dataset_dir` / `--floatbench_dir`
otherwise.

```bash
mkdir -p data
huggingface-cli download DeCoDELab/FLOATBench --repo-type dataset --local-dir data/FLOATBench
# FLOATSense: see the dataset link in the paper
```

Quick look:

```python
import pandas as pd
from floatsense import load_tower

tower = load_tower("data/FLOATSense", "opt2")
series = tower.load(1)                       # (10001, 37) float32, tower.channels
meta = tower.metadata.loc[tower.split_ids("test")]
damage = tower.damage()                      # sim_id x section_id
```

## Quickstart

```bash
# Physics baseline on one tower (CPU, ~10 min): calibrate on train, score test
python scripts/physics/run.py --flagfile=scripts/physics/config.cfg --tower=opt2

# One learned model on one tower, scored zero-shot on the other two (~10 min on one GPU)
python scripts/train/run.py --flagfile=scripts/train/config.cfg \
    --tower=opt2 --models=tcn --eval_towers=ref,opt1

# Results table of every run under outputs/
python scripts/benchmark/run.py --flagfile=scripts/benchmark/config.cfg
```

Model names (`--models`): `naive`, `dlinear`, `tcn`, `unet`, `lstm`,
`transformer` (PatchTST), `itransformer`, `s4`, `mamba`, `spectral`, `fits`,
`fno`, `timesnet`, `chronos`, `moment`, `moment_ft`, `timesfm`,
`timesfm_ft`, `hybrid`, `hybrid_tcn`, `prob_tcn`.

Recipe, identical for every model: 50 epochs of Adam at 1e-3, batch 16,
mean squared error on the standardized moment, 4,096-sample crops (full
window for the length-fixed models), no weight decay, scheduler or early
stopping, last epoch scored. Exceptions: Mamba uses `--learning_rate=3e-4`;
Prob-TCN trains on the Gaussian likelihood. The hybrid models need the
physics calibration of their tower (`outputs/physics/<tower>`), so run the
physics first.

Outputs of a training run (`outputs/within/<tower>/seed<k>/`):
`<model>_fa.pt` (weights and normalization), `history_<model>_fa.json`,
`damage_comparison_<model>_fa[_zs_<target>].csv` (true and reconstructed
damage at the 11 gauges per test simulation) and `summary_*.json`.

## Reproducibility notes

- Training on GPU is not bit-for-bit deterministic (cuDNN); reruns of a
  seed move the scores by far less than the spread between seeds.
- The reference damage is computed on the even 6,000-sample window
  400.0-999.9 s, as in the evaluation, and matches the evaluation to 1e-6.
- The parked constant C1 is recomputed from `parked.parquet` and matches
  the paper to 0.1 kN s^2.

## License

Code: MIT (`LICENSE.txt`). Data: CC-BY-4.0.
