# pylint: disable=wrong-import-position
"""Status of the validation-tuned track, from its files only.

    python hpo/status.py --root=outputs/hpo              # advance + status
    python hpo/status.py --root=outputs/hpo --no_advance  # read only

Writes <root>/status.json, status.txt and status.html: per study the
counted, pruned, failed and running trials, the best validation score so
far, the phase, the confirmation and final units done, the hours left and
an ETA (from the measured trial durations of the study, else from the
`cost` given per model, in hours per trial); the workers and their units;
the parked units and studies; the recent alerts. No test output is read.
With advance (the default), the phases are moved forward first
(pick.advance).
"""

import argparse
import datetime
import glob
import html
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo import common
from hpo import pick
from hpo import constants as C
from hpo import search_space as S

MAX_ALERTS_SHOWN = 20
COLUMNS = ("site", "study", "phase", "counted", "target", "pruned", "failed",
           "running", "best", "confirm", "winner", "final", "hours_left", "eta")


def eta(hours: Optional[float], workers: int = 1) -> Optional[str]:
    """Time stamp `hours` of work from now on `workers` GPUs."""
    if not hours or hours <= 0:
        return None
    stamp = datetime.datetime.now() + datetime.timedelta(hours=hours /
                                                         max(1, workers))
    return stamp.isoformat(timespec="minutes")


def hours_left(state: Dict,
               cost: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Hours left in a study (one GPU): its trial-equivalents left times
    the measured hours per trial (else `cost`, hours per trial)."""
    per_trial = state["hours_per_trial"] or (cost or {}).get(state["model"])
    if per_trial is None:
        return None
    # Trial-equivalents left (work_left is in units of the model's cost,
    # which may be 0).
    trials = pick.work_left({**state, "cost": 1.0})
    return trials * per_trial


def study_rows(states: List[Dict],
               site: str = "",
               cost: Optional[Dict[str, float]] = None) -> List[Dict]:
    """One status row per study."""
    rows = []
    for state in states:
        hours = hours_left(state, cost)
        rows.append({
            "site": site,
            "study": f"{state['model']}_{state['tower']}",
            "model": state["model"],
            "phase": state["phase"],
            "counted": state["counted"],
            "target": state["target"],
            "pruned": state["pruned"],
            "failed": state["failed"],
            "running": state["running"],
            "best": state["best"],
            "best_trial": state["best_trial"],
            "confirm":
                (f"{state['confirm_done']}/"
                 f"{len(state['confirm_units'])}" if state["plan"] else "-"),
            "winner": state["winner"],
            "final": (f"{state['final_done']}/{len(state['final_units'])}"
                      if state["final_units"] else "-"),
            "units_parked": state["units_parked"],
            "hours_left": None if hours is None else round(hours, 1),
            "eta": eta(hours),
        })
    return rows


def read_workers(root: str) -> List[Dict]:
    """Worker records with the age of their last change (minutes)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "workers", "*.json"))):
        record = common.read_json(path) or {}
        record["age_min"] = round((time.time() - os.path.getmtime(path)) / 60,
                                  1)
        rows.append(record)
    return rows


def read_events(root: str) -> List[Dict]:
    """Every event line of the workers."""
    events = []
    for path in glob.glob(os.path.join(root, "events", "*.jsonl")):
        with open(path, encoding="utf-8") as file:
            for line in file:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line being written
    return events


def parked_units(root: str) -> List[str]:
    """Run directories with a PARKED marker, and parked studies."""
    found = []
    for pattern in ("trials/*/*/PARKED", "phase2/*/*/PARKED",
                    "final/*/*/PARKED", "parked/*.json"):
        found += [
            os.path.relpath(p, root)
            for p in glob.glob(os.path.join(root, pattern))
        ]
    return sorted(found)


def alert_files(root: str) -> List[str]:
    """Alert records, oldest first."""
    return sorted(glob.glob(os.path.join(root, "alerts", "*.json")))


def collect(root: str,
            models,
            n_trials: int = C.N_TRIALS,
            cost: Optional[Dict[str, float]] = None,
            site: str = "") -> Dict:
    """The status of the track under `root`."""
    states = pick.all_states(root, models, n_trials, cost)
    ends = [e["end"] for e in read_events(root) if "end" in e]
    usage = shutil.disk_usage(root)
    return {
        "time": common.now(),
        "root": root,
        "studies": study_rows(states, site, cost),
        "workers": read_workers(root),
        "units_completed": len(ends),
        "last_completion": max(ends) if ends else None,
        "parked": parked_units(root),
        "alerts": [
            os.path.basename(p) for p in alert_files(root)[-MAX_ALERTS_SHOWN:]
        ],
        "disk_used_fraction": round(usage.used / usage.total, 3),
        "ready_for_test": os.path.exists(pick.ready_path(root)),
    }


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def summary(status: Dict) -> str:
    """One line: completions, disk, readiness."""
    return (f"units completed {status['units_completed']}, last "
            f"{status['last_completion']}; disk "
            f"{100 * status['disk_used_fraction']:.0f}% used"
            f"{'; READY FOR TEST' if status['ready_for_test'] else ''}")


def as_text(status: Dict) -> str:
    """Plain-text status."""
    rows = [[_cell(r[c]) for c in COLUMNS] for r in status["studies"]]
    widths = [
        max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(COLUMNS)
    ]
    lines = [
        f"Validation-tuned track, {status['time']} ({status['root']})",
        summary(status), "",
        "  ".join(c.ljust(w) for c, w in zip(COLUMNS, widths))
    ]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    lines += ["", "workers:"]
    lines += [
        f"  {w.get('owner')} {w.get('tag') or ''} {w.get('host')} "
        f"{w.get('gpu') or ''} unit={w.get('unit')} ({w['age_min']} min ago)"
        for w in status["workers"]
    ]
    lines += ["", "parked:"] + [f"  {p}" for p in status["parked"]]
    lines += ["", "recent alerts:"] + [f"  {a}" for a in status["alerts"]]
    return "\n".join(lines) + "\n"


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>Tuned track status</title>
<style>
:root {{ --bg: #fbfbfa; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e2dc;
  --accent: #2f5d8a; --bad: #b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #171716; --fg: #ececea;
  --muted: #9a9a94; --line: #34342f; --accent: #8db4dc; --bad: #f2b8b5; }} }}
body {{ background: var(--bg); color: var(--fg); margin: 0 16px;
  font: 14px/1.45 system-ui, sans-serif; }}
h1 {{ font-size: 20px; margin: 20px 0 4px; }}
h2 {{ font-size: 16px; margin: 24px 0 6px; }}
p {{ color: var(--muted); margin: 4px 0; }}
.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; border-bottom: 1px solid var(--line);
  text-align: left; white-space: nowrap; }}
th {{ color: var(--muted); font-weight: 600; }}
td.num {{ text-align: right; }}
tr.parked td {{ color: var(--bad); }} tr.done td {{ color: var(--accent); }}
</style></head><body>
<h1>Validation-tuned track</h1>
<p>{stamp}. {summary}</p>
<h2>Studies</h2><div class="wrap"><table><tr>{head}</tr>{rows}</table></div>
<h2>Workers</h2><div class="wrap"><table>{workers}</table></div>
<h2>Parked</h2><p>{parked}</p>
<h2>Recent alerts</h2><p>{alerts}</p>
</body></html>
"""


def as_html(status: Dict) -> str:
    """Static status page (reloads every 10 minutes)."""
    esc = html.escape
    rows = "".join(
        f"<tr class=\"{esc(r['phase'])}\">" +
        "".join(f"<td{' class=num' if isinstance(r[c], (int, float)) else ''}>"
                f"{esc(_cell(r[c]))}</td>"
                for c in COLUMNS) + "</tr>"
        for r in status["studies"])
    workers = "".join(
        f"<tr><td>{esc(str(w.get('owner')))}</td><td>"
        f"{esc(str(w.get('tag') or ''))}</td><td>"
        f"{esc(str(w.get('gpu') or ''))}</td><td>{esc(str(w.get('unit')))}"
        f"</td><td>{w['age_min']} min</td></tr>" for w in status["workers"])
    return PAGE.format(stamp=esc(status["time"]),
                       summary=esc(summary(status)),
                       head="".join(f"<th>{c}</th>" for c in COLUMNS),
                       rows=rows,
                       workers=workers or "<tr><td>none</td></tr>",
                       parked=esc(", ".join(status["parked"]) or "none"),
                       alerts="<br>".join(esc(a) for a in status["alerts"]) or
                       "none")


def write_status(root: str, status: Dict) -> None:
    """status.json, status.txt and status.html, atomically."""
    common.write_json(os.path.join(root, "status.json"), status)
    for name, text in (("status.txt", as_text(status)), ("status.html",
                                                         as_html(status))):
        path = os.path.join(root, name)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(tmp, path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="outputs/hpo")
    parser.add_argument("--models",
                        default="all",
                        help="Models shown (comma list or all).")
    parser.add_argument("--n_trials", type=int, default=C.N_TRIALS)
    parser.add_argument("--no_advance",
                        action="store_true",
                        help="Only read (do not advance the phases).")
    args = parser.parse_args(argv)
    args.models = (list(S.LEARNED)
                   if args.models == "all" else args.models.split(","))
    return args


def main(argv: Optional[List[str]] = None,
         cost: Optional[Dict[str, float]] = None) -> Dict:
    """Advances the phases (unless --no_advance) and writes the status."""
    args = parse_args(argv)
    if not args.no_advance:
        pick.advance(args.root, args.models, args.n_trials)
    status = collect(args.root, args.models, args.n_trials, cost)
    write_status(args.root, status)
    print(as_text(status), flush=True)
    return status


if __name__ == "__main__":
    main()
