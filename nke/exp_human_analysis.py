"""Turn returned human-study responses into the empirical human curve (Sec 3.3).

The harness (see `human.export_stimuli_and_harness`) has each participant
classify a between-subjects sample of the stimulus set and export a
`responses_<pid>.csv`. This module ingests a directory of such CSVs, applies an
attention-check filter (participants must classify the clean control images
correctly above a threshold), and computes, per perturbation level `eps_l` and
condition (NKE vs matched-Gaussian control):

  * human accuracy (response == true_label),
  * "cannot tell" rate,
  * mean reaction time,
  * number of judgements and participants,
  * bootstrap 95% CI on accuracy.

It writes `results/human_real_<dataset>_<arch>.json` and, if the computational
proxy file (`human_<dataset>_<arch>.json`) is present, an overlay figure putting
the real human curve next to the model line and the proxies.

Usage:
  ./.venv/bin/python -m nke.exp_human_analysis --dataset cifar10 --arch resnet18 \
      --responses /path/to/collected_csvs/
"""
import os, csv, glob, json, argparse
from collections import defaultdict
import numpy as np
from . import config as C

REQUIRED = {"file", "true_label", "condition", "eps_l", "response"}

# Attention-check thresholds are dataset-dependent: the clean-image human ceiling
# on 32x32 CIFAR-10 is far below MNIST's (tiny, ambiguous images -> genuine
# ship/automobile-type confusions even when attentive), so a single 0.8 gate
# wrongly excludes attentive CIFAR raters. These are sensible defaults; override
# with --attn once you have enough clean-trial data to estimate the true ceiling.
DEFAULT_ATTN = {"mnist": 0.8, "cifar10": 0.55}


def _load_rows(resp_dir):
    rows = []
    paths = sorted(glob.glob(os.path.join(resp_dir, "*.csv")))
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not REQUIRED.issubset(set(reader.fieldnames)):
                print(f"[human-analysis] skipping {os.path.basename(path)}: "
                      f"missing columns {REQUIRED - set(reader.fieldnames or [])}")
                continue
            for r in reader:
                r["_srcfile"] = os.path.basename(path)
                rows.append(r)
    return rows, paths


def _pid(r):
    return r.get("participant_id") or r["_srcfile"]


def _is_clean(r):
    try:
        return r["condition"] == "clean" or float(r["eps_l"]) == 0.0
    except (ValueError, KeyError):
        return r.get("condition") == "clean"


def _boot_ci(correct, n_boot=2000, seed=C.SEED):
    """Bootstrap 95% CI for a mean of a 0/1 array."""
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return (float("nan"), float("nan"))
    g = np.random.RandomState(seed)
    means = [correct[g.randint(0, len(correct), len(correct))].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run(dataset, arch, responses_dir, attn_threshold=None, make_figure=True):
    if attn_threshold is None:
        attn_threshold = DEFAULT_ATTN.get(dataset, 0.8)
    rows, paths = _load_rows(responses_dir)
    if not rows:
        print(f"[human-analysis] no usable responses.csv in {responses_dir}")
        return None

    # --- attention check: per-participant accuracy on clean trials ---
    clean = defaultdict(lambda: [0, 0])  # pid -> [correct, total]
    for r in rows:
        if _is_clean(r):
            c = clean[_pid(r)]
            c[1] += 1
            c[0] += int(r["response"] == r["true_label"])
    have_clean = any(tot > 0 for _, tot in clean.values())
    if have_clean:
        passed = {p for p, (ok, tot) in clean.items() if tot > 0 and ok / tot >= attn_threshold}
        excluded = {p for p in clean if p not in passed}
    else:
        passed = {_pid(r) for r in rows}          # no attention trials -> keep everyone
        excluded = set()
        print("[human-analysis] WARNING: no clean/attention trials found; keeping all participants.")

    kept = [r for r in rows if _pid(r) in passed]
    n_participants = len({_pid(r) for r in rows})
    print(f"[human-analysis] {n_participants} participants, "
          f"{len(passed)} passed attention (>= {attn_threshold:.0%} on clean), "
          f"{len(excluded)} excluded; {len(kept)}/{len(rows)} judgements retained.")

    # --- aggregate per (eps_l, condition) over retained participants ---
    cells = defaultdict(lambda: {"correct": [], "cant": 0, "rt": [], "pids": set()})
    for r in kept:
        try:
            eps = float(r["eps_l"])
        except ValueError:
            continue
        cond = "clean" if _is_clean(r) else r["condition"]
        key = (eps, cond)
        cell = cells[key]
        cell["correct"].append(int(r["response"] == r["true_label"]))
        cell["cant"] += int(r["response"] == "cannot_tell")
        if r.get("rt_ms", "").strip().isdigit():
            cell["rt"].append(int(r["rt_ms"]))
        cell["pids"].add(_pid(r))

    levels = []
    for (eps, cond) in sorted(cells.keys()):
        cell = cells[(eps, cond)]
        n = len(cell["correct"])
        acc = float(np.mean(cell["correct"])) if n else float("nan")
        lo, hi = _boot_ci(cell["correct"])
        levels.append({
            "eps_l": eps, "condition": cond, "n_judgements": n,
            "n_participants": len(cell["pids"]),
            "human_accuracy": acc, "acc_ci95": [lo, hi],
            "cannot_tell_rate": (cell["cant"] / n) if n else float("nan"),
            "mean_rt_ms": (float(np.mean(cell["rt"])) if cell["rt"] else None),
            "model_accuracy": 1.0,                     # 1.0 by NKE construction
            "human_model_gap": (1.0 - acc) if n else float("nan"),
        })

    out = {"dataset": dataset, "arch": arch, "responses_dir": responses_dir,
           "n_csv_files": len(paths), "n_participants": n_participants,
           "n_passed_attention": len(passed), "attn_threshold": attn_threshold,
           "levels": levels}
    path = os.path.join(C.RESULTS_DIR, f"human_real_{dataset}_{arch}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")

    # log a compact NKE-condition summary
    print("  eps_l | human_acc (NKE) [95% CI] | cannot-tell | n")
    for lvl in levels:
        if lvl["condition"] == "nke":
            ci = lvl["acc_ci95"]
            print(f"  {lvl['eps_l']:5.1f} | {lvl['human_accuracy']:.3f} "
                  f"[{ci[0]:.2f},{ci[1]:.2f}] | {lvl['cannot_tell_rate']:.2f} | {lvl['n_judgements']}")

    if make_figure:
        _figure(dataset, arch, out)
    return out


def _figure(dataset, arch, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(cond):
        pts = sorted([l for l in out["levels"] if l["condition"] in (cond, "clean")],
                     key=lambda d: d["eps_l"])
        # clean row belongs to both conditions (eps_l=0 baseline)
        xs = [p["eps_l"] for p in pts]
        ys = [p["human_accuracy"] for p in pts]
        lo = [p["acc_ci95"][0] for p in pts]
        hi = [p["acc_ci95"][1] for p in pts]
        return xs, ys, lo, hi

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    xn, yn, ln, hn = curve("nke")
    ax.plot(xn, yn, "-o", color="tab:red", lw=2, label="Human accuracy (NKE)")
    if not any(np.isnan(ln)):
        ax.fill_between(xn, ln, hn, color="tab:red", alpha=0.15)
    xg, yg, lg, hg = curve("gauss")
    if len(xg) > 1:
        ax.plot(xg, yg, "--s", color="gray", lw=1.6, label="Human accuracy (Gaussian ctrl)")
    ax.axhline(1.0, color="tab:blue", lw=2, ls="-", label="Model accuracy (=1.0)")

    # overlay computational proxies if available
    prox = os.path.join(C.RESULTS_DIR, f"human_{dataset}_{arch}.json")
    if os.path.exists(prox):
        p = json.load(open(prox))
        pe = [r["eps_l"] for r in p["levels"]]
        ax.plot(pe, [r["generic_proxy_nke"] for r in p["levels"]], ":^",
                color="tab:green", alpha=0.8, label="Generic-recognizer proxy")
        ax.plot(pe, [r["shape_proxy_nke"] for r in p["levels"]], ":d",
                color="tab:purple", alpha=0.8, label="Shape proxy")

    ax.axhline(1.0 / C.NUM_CLASSES[dataset], color="k", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel(r"Perturbation magnitude $\epsilon_l$ (L2)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Empirical human curve vs. model & proxies — {dataset.upper()}/{arch}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.FIG_DIR, f"human_real_{dataset}_{arch}.png")
    fig.savefig(fp); plt.close(fig)
    print(f"[fig] {fp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--responses", required=True, help="directory of returned responses_*.csv")
    ap.add_argument("--attn", type=float, default=None,
                    help="min clean-trial accuracy to keep a participant "
                         "(default: dataset-aware — 0.8 MNIST, 0.55 CIFAR-10)")
    a = ap.parse_args()
    run(a.dataset, a.arch, a.responses, attn_threshold=a.attn)
