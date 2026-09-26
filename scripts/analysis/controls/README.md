# Controls for the metric and transfer analyses

Scripts behind the appendix on controls. They read the per-simulation damage
files written by the benchmark runs and write to `outputs/controls/`. Only
`scale/infer_designed.py` and `waveform/waveform.py` run a model (inference
of the stored checkpoints, GPU optional); nothing is trained. The
surrogate needs `xgboost` and `scikit-learn`, the Markdown tables `tabulate`
(all in `requirements.txt`).

| Script | Analysis |
|---|---|
| `a123_cpu.py` | (1) condition-only oracle (mean of the true damage over the six realizations of each operating point, and leave-one-out); (2) within-condition calibration slope (OLS slope of predicted on true within-condition deviation of log10 damage); (3) lifetime damage ratio. Condition-level cluster bootstrap, B = 2000. |
| `a123_extras.py` | Same metrics for the protocol variants (two axes, SCADA only, top only). |
| `a1b_surrogate.py` | Condition-only XGBoost surrogate (and a GP) on (wind, Hs, Tp), trained on the training split. |
| `scale/infer_designed.py` | Inference of the source-tower checkpoints on the target tower's few-shot training simulations (`fewshot/train_designed`, `fewshot/train_5_draw0`) or, with `check`, on its first three test simulations. |
| `scale/check.py` | Checks that re-run inference (`check`) reproduces the stored zero-shot damage. |
| `scale/analyze.py`, `scale/boot.py` | One scale factor per height, median(true/predicted damage) over the calibration simulations, applied to the stored zero-shot predictions; cluster-bootstrap intervals. |
| `waveform/waveform.py`, `waveform/summarize.py` | Pearson correlation and normalized RMSE of the reconstructed moment history at base, mid and top on 200 fixed test simulations per tower. |
| `make_tables.py` | Tables of the appendix (after `a123_cpu.py`). |

## Paths

Set by environment variables (`layout.py`); relative values are taken from
`FLOATSENSE_ROOT`.

| Variable | Default | Content |
|---|---|---|
| `FLOATSENSE_ROOT` | `.` | repository root |
| `FLOATSENSE_DATA` | `data/FLOATSense` | released dataset (`load_tower`) |
| `FLOATSENSE_RUNS` | `outputs/within` | within-tower runs, `<runs>/<tower>/seed<k>/` |
| `FLOATSENSE_OUTPUTS` | `outputs` | root of the other experiments (table below) |
| `FLOATSENSE_CONTROLS` | `outputs/controls` | where the controls write |
| `FLOATSENSE_LAYOUT` | `release` | `release` or `paper` (table below) |
| `FLOATSENSE_PAPER_TABLES` | unset | folder of the paper's `curve_top.tex` and `curve_base.tex`; fills the `paper_FT_*` columns of `scale/analyze.py` |

Files read (`<t>` tower, `<k>` seed, `<m>` model: `naive`, `tcn`,
`prob_tcn`, `transformer`, `mamba`):

| Input | Written by | Release layout | Paper layout |
|---|---|---|---|
| within tower, seeds 0-2 | `scripts/train/run.py` (`--eval_towers` for the zero-shot files) | `<runs>/<t>/seed<k>/damage_comparison_<m>_fa[_zs_<target>].csv`, `<m>_fa.pt` | same, with `FLOATSENSE_RUNS=<outputs>/heights` |
| two axes, seed 0 | `--output_root=outputs/ablation/twoaxis` | `<outputs>/ablation/twoaxis/<t>/seed0/` | `<outputs>/heights/ablation/twoaxis/<t>/` |
| SCADA only, seed 0 | `--output_root=outputs/ablation/scada` | `<outputs>/ablation/scada/<t>/seed0/` | `<outputs>/heights/ablation/scada/<t>/` |
| top only, seed 0 | `--output_root=outputs/toponly` | `<outputs>/toponly/<t>/seed0/` | `<outputs>/diag/toponly/<t>/` |
| physics, seed 0 | `scripts/physics/run.py` | `<outputs>/physics/<t>/damage_heights.csv` | `<outputs>/physics_heights/<t>_E2/damage_heights.csv` |
| Prob-TCN mean head, seed 0 (optional) | research code only | `<outputs>/noise/<t>/damage_comparison_prob_mean_fa.csv` | `<outputs>/heights/noise/<t>/...` |

`a123_cpu.py` skips (and prints) a missing file, e.g. the mean-head row, which
no released script writes. The few-shot fine-tuning runs are not read: the
paper's fine-tuning numbers enter `scale/analyze.py` only through
`FLOATSENSE_PAPER_TABLES`.

Outputs, under `<controls>/`: `a123_per_seed.csv`, `a123_seed_median.csv`,
`a123_seed0.csv`, `tables_a123.md`, `a123_extras_seed0.csv`,
`tables_extras.md`, `a1b_surrogate.csv`; `a4_scale/infer/<src>/seed<k>/`
(re-inferred damage), `a4_scale/scale_recal_*.csv`; `a5_waveform/`.

## Order

```bash
export FLOATSENSE_DATA=data/FLOATSense FLOATSENSE_RUNS=outputs/within
C=scripts/analysis/controls
python $C/a123_cpu.py && python $C/make_tables.py
python $C/a123_extras.py
python $C/a1b_surrogate.py

# scale: inference (GPU optional), then the CPU analyses
for k in 0 1 2; do for s in ref opt1 opt2; do for t in ref opt1 opt2; do
  [ $s = $t ] && continue
  for m in tcn prob_tcn naive; do
    [ -f $FLOATSENSE_RUNS/$s/seed$k/${m}_fa.pt ] || continue
    for w in check designed rand5; do
      python $C/scale/infer_designed.py $s $t $m $k $w
    done
    python $C/scale/check.py $s $t $m $k
  done
done; done; done
python $C/scale/analyze.py && python $C/scale/boot.py

# waveform (seed 0 also for PatchTST and Mamba); an optional last argument
# scores only the first n of the 200 simulations
for t in ref opt1 opt2; do
  python $C/waveform/waveform.py $t 0 tcn,prob_tcn,transformer,mamba
  for k in 1 2; do python $C/waveform/waveform.py $t $k; done
done
python $C/waveform/summarize.py
```

Prob-TCN samples its variance head, so `check.py` matches its true damage
only; TCN and the floor match the stored zero-shot damage to rounding on the
same device (inference on the CPU differs from the GPU runs by up to a few
parts in a thousand in the reconstructed damage).
