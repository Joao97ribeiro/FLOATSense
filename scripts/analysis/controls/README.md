# Controls for the metric and transfer analyses

Scripts behind the appendix on controls. They read the per-simulation damage
files written by the benchmark runs (`outputs/heights/<tower>/seed<k>/damage_comparison_<model>_fa*.csv`)
and write to `outputs/controls/`. Set `FLOATSENSE_ROOT` to the repository root
(default: current directory). No model is trained.

| Script | Analysis |
|---|---|
| `a123_cpu.py` | (1) condition-only oracle (mean of the true damage over the six realizations of each operating point, and leave-one-out); (2) within-condition calibration slope (OLS slope of predicted on true within-condition deviation of log10 damage); (3) lifetime damage ratio. Condition-level cluster bootstrap, B = 2000. |
| `a123_extras.py` | Same metrics for the protocol variants (two axes, SCADA only, top only, Prob-TCN mean head). |
| `a1b_surrogate.py` | Condition-only XGBoost surrogate on (wind, Hs, Tp), trained on the E2 training split. |
| `scale/infer_designed.py` | Inference of the source-tower checkpoints on the target tower's few-shot training simulations (GPU optional). |
| `scale/analyze.py`, `scale/boot.py` | One scale factor per height, median(true/predicted damage) over the calibration simulations, applied to the zero-shot predictions. |
| `scale/check.py` | Checks that re-run inference reproduces the stored zero-shot damage. |
| `waveform/waveform.py`, `waveform/summarize.py` | Pearson correlation and normalized RMSE of the reconstructed moment history at base, mid and top on 200 fixed test simulations per tower. |
| `make_tables.py` | Tables of the appendix. |
