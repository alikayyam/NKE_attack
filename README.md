# NKE — "A New Kind of Adversarial Example": experiment suite

Implementation of the three evaluation tracks proposed in the paper (human
psychometric study, OOD/calibration analysis, robustness-intervention
evaluation) plus the mechanistic shape-vs-texture analysis and the reduced
baseline replication, run on **MNIST** and **CIFAR-10**.

NKE examples are *large, label-preserving* perturbations: we minimise the
true-label loss (keep the model correct) while a lower-bound projection holds
the perturbation at distance `>= eps_l` from the clean image (Eq. 1 / Nie Eq. 7).

## Environment note
The box's PyTorch (`cu130`) dropped Pascal (sm_61) support, so the GTX 1080 Ti
GPUs were unusable with it. A local venv with `torch==2.5.1+cu121` (which
includes sm_61) was created at `.venv/` inheriting the base env's numpy/sklearn/
cv2/skimage. **Run everything with `./.venv/bin/python`.**

## Layout
```
nke/
  config.py        paths, device (auto-selects free GPU), dataset stats
  data.py          MNIST/CIFAR loaders ([0,1] space), OOD ref sets, correct-sample helper
  models.py        LeNet/MLP (MNIST), ResNet18/VGG11/ViT (CIFAR) + NormalizedModel wrapper
  attack.py        NKE attacks (ifgsm/nifgm/nmifgsm/nmifgm) + classical PGD
  attackset.py     shared stimulus generator across an eps_l grid (single-pipeline design)
  metrics.py       ECE, MSP/energy/Mahalanobis OOD scores, detection-rate@FPR, AUROC
  mechanism.py     Canny edge-preservation, GLCM + Gram texture-similarity
  human.py         generic-recognizer + shape/edge recognizer proxies; stimuli + HTML harness export
  defenses.py      JPEG / blur / random-resize-pad / randomized-smoothing wrappers (BPDA where needed)
  exp_*.py         one driver per track; run_all.py orchestrates
  figures.py       overlay / mechanism / calibration / defense figures
  train_models.py  clean + adversarial (PGD, Madry) training
results/           JSON results + cached stimulus tensors
figures/           PNG figures
stimuli/<ds>_<arch>/  images/, manifest.json, index.html  (ready-to-run crowd study)
checkpoints/       trained models
```

## Reproduce
```bash
./.venv/bin/python -m nke.train_models                 # clean models (all 4)
./.venv/bin/python -m nke.train_models --advtrain      # + adversarially-trained CIFAR resnet
./.venv/bin/python -m nke.run_all --datasets mnist cifar10 --n 200
./.venv/bin/python -m nke.figures
```

## Mapping to paper sections
| Paper section | Module | Output |
|---|---|---|
| 3.2 Baseline replication (cross-model matrix, ablations) | `exp_baseline` | `results/baseline.json` |
| 3.3 Human psychometric study | `exp_human` + `human` | `results/human_*.json`, `stimuli/*` |
| 3.3 OOD + calibration | `exp_ood` | `results/ood_*.json` |
| 3.3 Adaptive attack vs. Mahalanobis | `exp_adaptive` | `results/adaptive_*.json` |
| 3.4 Shape vs texture mechanism | `exp_mechanism` | `results/mechanism_*.json` |
| 3.5 Robustness interventions | `exp_defenses` | `results/defenses_*.json` |
| 3.x ViT shape-bias probe (architecture generality) | `exp_vit_shapebias` | `results/vit_shapebias_cifar10.json` |
| Fig. 4 overlay | `figures.overlay_figure` | `figures/overlay_*.png` |

## Human study — what is and isn't measured
Real crowd participants cannot be recruited in this environment. The suite
therefore (1) exports a complete, ready-to-run stimulus set + forced-choice HTML
task (with a "cannot tell" option) and (2) reports **two clearly-labelled
computational recognizability proxies** — an independent generic recognizer and
a shape/edge-map recognizer — as stand-ins for the human curve. These are
proxies, **not** human data; the empirical human curve requires running the
exported harness with participants.

### Running the human study (pilot → full)
The exported harness (`stimuli/<ds>_<arch>/`) is a self-contained static web app:
`index.html` (forced-choice task) + `manifest.json` (stimuli with ground truth,
condition, ε_l) + `images/`. It assigns each participant a between-subjects
sample (**no participant sees the same base image at two ε_l levels**), always
shows the clean images as **attention checks**, and exports a
`responses_<pid>.csv`.

```bash
# 1. Pilot locally — MUST be served over HTTP (fetch() is blocked on file://)
cd stimuli/cifar10_resnet18 && python -m http.server 8000   # open http://localhost:8000/

# 2. For a real run, host the folder as static files (Netlify/S3/GH-Pages) and
#    recruit via Prolific/MTurk. Optionally set human.HARNESS_POST_URL before
#    export to auto-POST responses instead of manual CSV download.

# 3. Collect the returned responses_*.csv into one directory, then:
./.venv/bin/python -m nke.exp_human_analysis --dataset cifar10 --arch resnet18 \
    --responses /path/to/collected_csvs/
#    -> results/human_real_cifar10_resnet18.json  (human acc + 95% CI, cannot-tell,
#       RT per ε_l, NKE vs Gaussian control, attention-filtered)
#    -> figures/human_real_cifar10_resnet18.png   (human curve vs model line + proxies)
```
Participants failing the attention check (clean-trial accuracy `< --attn`, default
0.8) are excluded automatically.
