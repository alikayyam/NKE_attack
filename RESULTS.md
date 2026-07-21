# NKE experiments — results & findings

Experiments for the three evaluation tracks proposed in *"A New Kind of
Adversarial Example"* (human psychometric study, OOD/calibration, robustness
interventions) plus the shape-vs-texture mechanism analysis and a reduced
baseline replication. Run on **MNIST** and **CIFAR-10**.

All raw numbers are in `results/*.json`; figures in `figures/`. Every downstream
track operates on one shared stimulus set per model (single-pipeline design).

## Setup
- **Env fix.** The box's `torch cu130` had dropped Pascal (sm_61) support, making
  the GTX 1080 Ti GPUs unusable. Installed `torch 2.5.1+cu121` in `.venv/`.
- **Models (clean test acc):** LeNet 99.3%, MLP 98.5% (MNIST); ResNet18 93.5%,
  VGG11 90.2% (CIFAR-10). Adversarially-trained (Madry PGD): LeNet 97.2%,
  ResNet18 74.0%.
- **Attack:** NMI-FGSM/NMI-FGM family, `min J(x',y) s.t. ||x'-x||_2 >= eps_l`,
  60 steps, exterior L2 projection at target `eps_l`.
- **Core NKE property holds everywhere:** in every condition below, source-model
  accuracy on the perturbed images is **100%** with **~1.0 confidence**, out to
  perturbations as large as the image content itself (MNIST L2≈11, CIFAR L2≈16).

---

## Track 0 — Baseline replication (`baseline.json`, Sec 3.2)

**Cross-model label-kept matrix** (fraction of source-generated NKE images the
target model still labels correctly; `*` = white-box):

| Source → | MNIST LeNet | MNIST MLP | | CIFAR RN18 | CIFAR VGG11 |
|---|---|---|---|---|---|
| **LeNet** | 1.00* | 0.98 | **RN18** | 1.00* | 0.50 |
| **MLP** | 0.95 | 1.00* | **VGG11** | 0.23 | 1.00* |

- White-box success ≈**100%** everywhere → replicates the high white-box rates of
  Nie et al. (80–99%).
- **White-box ≫ black-box asymmetry replicated on CIFAR** (1.00 vs 0.23–0.50).
- **Novel synthesis — transferability falls with task complexity:**
  MNIST ≈0.96 → CIFAR ≈0.37 → ImageNet <0.05 (Nie et al.). Borji's simple-dataset
  regime and Nie's ImageNet regime are two ends of one trend; on simple data the
  NKE perturbation preserves class-defining structure that generalizes across
  models, on complex data it becomes model-specific.

**Ablations (white-box success):** flat at 1.00 across perturbation size,
momentum μ∈[0,1], and iterations ≥10 (5 iters → 0.875 on CIFAR). The attack is
robust and cheap to run — consistent with prior work.

---

## Track 1 — Human psychometric study (`human_*.json`, `human_real_*.json`, Sec 3.3)

Deliverables: (a) exported stimulus set + forced-choice HTML harness with a
"cannot tell" option (`stimuli/<ds>_<arch>/`); (b) **two labelled computational
proxies** — an independent *generic recognizer* (the other architecture, never
exposed to the attack gradients) and a *shape/edge-map recognizer*; (c) a **real
human pilot (N=5)** collected with the harness (see "Empirical human curve"
below). The proxies are proxies, **not** human data; the pilot is real but
small-sample.

**CIFAR-10 (ResNet18 source; model accuracy = 1.00 at every ε_l):**

| ε_l | model | generic proxy (NKE) | generic (Gaussian ctrl) | shape proxy (NKE) |
|----:|:----:|:----:|:----:|:----:|
| 0 | 1.00 | 0.93 | 0.93 | 0.60 |
| 2.5 | 1.00 | 0.965 | 0.54 | 0.40 |
| 5 | 1.00 | 0.80 | 0.20 | 0.19 |
| 8 | 1.00 | 0.64 | 0.15 | 0.13 |
| 16 | 1.00 | **0.49** | 0.15 | 0.11 |

**Findings.**
1. **A real (proxied) human-model gap exists on CIFAR:** the independent
   recognizer falls to **~49%** while the source model stays at 100% — a ~50-point
   gap, the first quantitative evidence for the gap the paper's premise assumed
   (Borji Fig. 3 was only schematic).
2. **The gap is *not* mere signal removal.** A matched-L2 **Gaussian control**
   degrades the generic recognizer *faster* (to ~0.15) than the structured NKE
   perturbation (to ~0.49). This directly answers Borji (2022)'s confound
   concern: the effect is specific to the adversarial structure, not to the
   magnitude of signal removed.
3. **Dataset dependence:** on MNIST the generic proxy stays high (0.99→0.94) — NKE
   digits remain broadly recognizable (mirrors the high MNIST transfer above), so
   the human-model gap is small there. The gap is a natural-image phenomenon.

### Empirical human curve — real pilot (`human_real_cifar10_resnet18.json`, N=5)

Five participants ran the exported forced-choice harness; **all 5 passed the
dataset-calibrated attention check** (≥55% on clean trials — the CIFAR clean
ceiling is well below MNIST's), **260/260 judgements retained**. Model accuracy =
1.00 at every ε_l by construction.

| ε_l | human acc (NKE) | 95% CI | cannot-tell | human acc (Gauss ctrl) | gap | n (NKE) |
|----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 0 (clean) | 0.85 | [.72,.95] | 0.07 | 0.85 | 0.15 | 40 |
| 1 | 0.88 | [.69,1.0] | 0.06 | 0.73 | 0.12 | 16 |
| 2.5 | 0.88 | [.71,1.0] | 0.00 | 0.81 | 0.12 | 17 |
| 5 | 0.63 | [.42,.84] | 0.11 | 0.50 | 0.37 | 19 |
| 8 | 0.37 | [.16,.58] | 0.37 | 0.44 | 0.63 | 19 |
| 12 | 0.09 | [.00,.22] | 0.78 | 0.08 | 0.91 | 23 |
| 16 | 0.08 | [.00,.25] | 0.83 | 0.05 | **0.92** | 12 |

**Findings (pilot — small sample, wide CIs).**
4. **The human-model gap is confirmed on real humans, not just proxied.** Human
   recognition of NKE images collapses from ~0.88 at low ε_l to **~0.08 at ε_l ≥
   12**, with an **83% "cannot tell" rate**, while the model stays at 100% — a
   **~90-point gap**. First direct human evidence for the gap the premise assumed.
5. **Real humans fall *further* than the proxy predicted.** The generic proxy
   plateaued at ~0.49 (ε_l=16); real humans reach ~0.08. The suite *under*-estimated
   the gap.
6. **The "not signal loss" dissociation does *not* clearly replicate at N=5.**
   Real humans find NKE slightly more recognizable than the matched-L2 Gaussian
   control at low–mid ε_l (0.63 vs 0.50 at ε_l=5; same direction as the proxy), but
   the two conditions are **not statistically separable** (CIs overlap at every
   level) and the ordering flips at ε_l=8. The signal-vs-structure claim rests on
   the proxy/CLIP evidence; confirming it on humans needs a properly-powered sample.
   *(N=5 is the full available pilot — no more participants can be run.)*
   → `figures/human_real_cifar10_resnet18.png` (human curve vs. model line vs. proxies).

---

## Track 2 — OOD detection & calibration (`ood_*.json`, Sec 3.3)

Detectors thresholded at **5% FPR** on clean validation data.

**CIFAR-10 ResNet18 (detection rate of NKE images):**

| ε_l | model acc | conf | ECE | MSP | Energy | **Mahalanobis** |
|----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| 8 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| 16 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | **1.00** |

Reference: genuine OOD (SVHN) → Maha 0.97 / MSP 0.52; classical PGD adv → Maha
1.00 / MSP 0.00. MNIST shows the same pattern more softly (MSP/Energy ≈0;
Mahalanobis 0.07→0.68).

**Findings.**
- **Sharp detector dissociation.** Confidence/logit-based detectors (MSP, energy)
  are **completely blind** (0% detection) to NKE images, because the model is
  confident *and* correct — so **ECE ≈ 0** (the calibration paradox: perfectly
  "calibrated" on inputs no human could label).
- **Feature-space distance catches them perfectly** (Mahalanobis = 100% on CIFAR
  from ε_l=1). NKE images are, as their construction implies, far off-manifold in
  representation space while indistinguishable at the output layer.
- **Takeaway:** NKE is best understood as an **OOD phenomenon invisible to
  confidence-based monitoring** — a concrete argument for feature-distance OOD
  detection over softmax/energy/temperature methods in deployment. This is the
  reframing neither source paper performed.

---

## Track 3 — Shape vs texture mechanism (`mechanism_*.json`, Sec 3.4)

**CIFAR-10 ResNet18 (clean vs perturbed):**

| ε_l | model acc | edge preservation (Canny F1) | texture sim (GLCM) |
|----:|:----:|:----:|:----:|
| 1 | 1.00 | 0.76 | 0.83 |
| 2.5 | 1.00 | 0.57 | 0.30 |
| 5 | 1.00 | 0.45 | **0.00** |
| 16 | 1.00 | 0.31 | 0.00 |

**Finding.** A clean **channel dissociation**: GLCM **texture similarity collapses
to ~0 by ε_l=5**, while **edge/shape structure survives far longer** (F1 plateaus
~0.31–0.45), with model accuracy pinned at 1.00 throughout. The perturbation
destroys low-level texture much faster than shape, and the model is invariant to
both. This is consistent with a texture/shape account of the human-model gap
(Geirhos et al.): the surviving edge skeleton is not enough for the shape
recognizer or generic recognizer to keep up, yet the source model retains its
decision regardless. (Gram-matrix cosine stayed ~1.0 — it is dominated by global
colour energy and is uninformative here; GLCM is the meaningful texture metric.)

---

## Track 4 — Robustness interventions (`defenses_*.json`, Sec 3.5)

White-box NKE re-run against each defended CIFAR model:

| Model | clean acc | classical robust acc (PGD) | NKE success @ max ε_l |
|---|:--:|:--:|:--:|
| undefended | 1.00 | 0.00 | 1.00 |
| JPEG (q40) | 0.78 | 0.11 | 0.99 |
| random resize+pad | 0.85 | 0.00 | 1.00 |
| adversarial training | 0.74 | **0.45** | **1.00** |

**Findings.**
- **No classical defense stops NKE:** every model, including one with **45% PGD
  robust accuracy**, is **~100% NKE-vulnerable** at large ε_l.
- **The two vulnerabilities are orthogonal.** Across models, correlation between
  classical robust accuracy and NKE resistance is **r ≈ −0.02** (MNIST: NKE
  resistance is uniformly 0 across a 0→86% robust-acc range — zero variance, the
  strongest possible statement of orthogonality). Robustness to small-ε attacks
  does *not* transfer to large-ε_l, same-label attacks.
- *Caveat:* post-hoc blur/randomized-smoothing crater clean accuracy on this
  non-robust base model (0.35 / 0.10), confounding their defense numbers; the
  clean adv-trained-vs-undefended contrast is the load-bearing comparison.

---

## Track 5 — Adaptive attack vs. the Mahalanobis detector (`adaptive_cifar10_resnet18.json`, Sec 3.3)

Track 2 champions feature-distance (Mahalanobis) detection as the one thing that
catches NKE. But that assumes an attacker who doesn't know the detector is there.
Since Mahalanobis distance is differentiable, the adaptive attacker minimises
`CE(f(x'),y) + λ·Maha(x')` — keep the label *and* pull the penultimate feature
back toward the clean class manifold — while still projecting the perturbation to
`L2 >= eps_l`. To rule out white-box overfitting, we score with **two** deployed
detectors: a *white-box* one (the exact detector the attacker differentiates
through) and a *held-out* one (same estimator, fit on a disjoint data split the
attacker never sees). Swept at a large `eps_l` (L2≈12) where oblivious NKE is 100%
caught:

| λ | NKE success | L2 | **Maha (white-box)** | **Maha (held-out)** | MSP |
|----:|:----:|:----:|:----:|:----:|:----:|
| 0 (oblivious) | 1.00 | 12.02 | **1.00** | **1.00** | 0.00 |
| 0.001 | 1.00 | 12.05 | **0.00** | **0.00** | 0.00 |
| 0.01 | 1.00 | 12.06 | **0.00** | **0.00** | 0.02 |
| 0.02 | 1.00 | 12.06 | **0.00** | **0.00** | 0.08 |
| 0.03 | 0.97 | 12.06 | 0.00 | 0.00 | 0.20 |
| 0.05 | 0.72 | 12.06 | 0.00 | 0.00 | 0.34 |
| 0.1 | 0.17 | 12.07 | 0.00 | 0.00 | 0.08 |
| 1.0 | 0.11 | 12.06 | 0.00 | 0.00 | 0.02 |

**Findings.**
1. There is a wide λ band (`10⁻³–2·10⁻²`) that is **simultaneously 100%
   successful, large-perturbation (L2≈12.06, essentially unchanged), and fully
   evasive** — Mahalanobis detection drops from 100% to **0%** at no cost. The max
   evasive λ with success=1.0 is **0.02** (still 0.97 at 0.03). Only for λ≥0.03,
   where feature-pullback trades off against the large-L2 projection, does success
   degrade (→0.11 by λ=1). MSP stays ≤0.08 across the evasive band (it only rises
   to 0.20–0.34 in the mid-λ region where success is already collapsing).
2. **The evasion is not white-box overfitting.** The held-out detector — fit on
   disjoint data the attacker never optimised against — is evaded *identically*
   (0% wherever the attacked detector is 0%). The attacker needs to know only that
   a feature-distance detector is in use, not its exact parameters.

**Feature-distance detection defeats a *static* NKE adversary but is defeated for
free by an *adaptive* one.** The honest reframing is therefore not "deploy
Mahalanobis" but: NKE is an off-manifold phenomenon that *only* off-manifold
detectors can see at all — and even those are not adaptively robust. See
`figures/adaptive_cifar10_resnet18.png`.

### 5a — Does evasion transfer across detector *design*? (`adaptive_detectors_cifar10_resnet18.json`)

Attacker penalises only the **tied-covariance penultimate** Mahalanobis detector,
then we score a battery of detectors it may/may not target (CIFAR-10 RN18, L2≈12,
5% FPR):

| Scenario | success | tied-penult | untied-penult | Maha layer-3 | cosine-kNN |
|---|:--:|:--:|:--:|:--:|:--:|
| Standard NKE (no adaptation) | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** |
| Adapt vs. tied-penult (λ=0.02) | 0.99 | **0.00** | 1.00 | 0.81 | 0.03 |
| Adapt vs. tied ensemble (penult+layer-3, λ=0.02) | 1.00 | **0.00** | 1.00 | **0.00** | 0.01 |

**Findings.**
1. **kNN is blind to NKE even un-adapted** (0.00): NKE features keep high cosine
   similarity to training features, so only *whitened*-distance (Mahalanobis-type)
   detectors catch NKE at all — detector *choice* matters before adaptation.
2. **Evasion is metric-specific.** Evading tied-cov penultimate does *not* evade a
   **per-class (untied) covariance** detector on the same features (stays 100%),
   nor a different layer (81%). A two-layer tied ensemble attacker evades both tied
   layers at 100% success — but untied-cov **still flags 100%**.
3. **Takeaway:** feature-distance detection is evadable only for the *specific
   metric the attacker targets and differentiates through*. The defender's residual
   advantage is detector **diversity/secrecy**, not any single detector. This
   sharpens *and bounds* the Track-5 negative result.

### 5b — Adaptive attack vs. an adversarially-trained model (`adaptive_cifar10_resnet18_advtrain.json`)

Same attack against the Madry-PGD ResNet18 (45% PGD robust acc): no-cost evasion
**reproduces** at small λ — λ=0.001 gives success 1.00, Maha 100%→0% (white-box &
held-out), L2≈12.5. At larger λ success degrades more gracefully and plateaus
~0.53 (vs 0.11 for the clean model) — the robust feature space is somewhat easier
to pull back toward the manifold while keeping the label — but small-λ evasion is
unaffected, consistent with the NKE/robustness orthogonality (Track 4).

---

## Track 6 — Vision Transformer shape-bias probe (`vit_shapebias_cifar10.json`, `*_cifar10_vit.json`)

Every finding above was measured on CNNs. We added a **from-scratch Vision
Transformer** (patch 4, dim 192, depth 6, heads 3; 1.8 M params; **clean test acc
83.2 %**) as a third CIFAR-10 architecture — a deliberately *different family*
(attention, no conv stack beyond patchify). It serves two purposes: an
**architecture-generality check** and a **falsifiable shape-bias test**. ViTs are
markedly more shape-biased than CNNs (Naseer et al. 2021; Tuli et al. 2021), so
Track 3 (texture is destroyed, shape survives) makes a concrete prediction: a
shape-biased recognizer should read the surviving edge skeleton *better* than a
CNN does.

**3×3 white/black-box label-kept matrix** (CIFAR-10, ε_l = 12; `*` = white-box):

| Source → | ResNet18 | VGG11 | ViT |
|---|:--:|:--:|:--:|
| **ResNet18** | 1.00* | 0.50 | 0.25 |
| **VGG11** | 0.23 | 1.00* | 0.22 |
| **ViT** | 0.13 | 0.16 | 1.00* |

**Finding 1 — the mechanism is architecture-general (not a CNN artifact).**
ViT-sourced NKE reproduces the Track-3 channel dissociation exactly: GLCM texture
similarity collapses to ~0 by ε_l = 5–8 while edge/shape F1 plateaus at ~0.32–0.44,
with **model accuracy pinned at 1.00** throughout. The texture-before-shape
signature holds for an attention model.

**Finding 2 — the shape-bias prediction is *refuted* (an honest negative).**
Raw cross-recognizer accuracy is confounded by the ViT's lower clean accuracy, so
we report **retention** = (accuracy on NKE) / (accuracy on that source's *clean*
samples). For **CNN-sourced** NKE, the ViT recognizer retains **~0.08–0.11 *less*
than the other CNN** across every ε_l ≥ 2.5 (contrast −0.083 → −0.109), i.e. the
shape-biased model is **not** a better reader — if anything slightly worse (driven
by the ResNet18 source; VGG11 source is ≈neutral). **Shape survival is therefore
necessary-looking but not sufficient for recognizability**: even a shape-biased
network cannot recover the label from the edge skeleton NKE leaves behind. This
*sharpens* Track 3 — the human–model gap is not closed by shape bias.

**Finding 3 — novel: attention models produce the *most model-specific* NKE.**
Source-specificity = mean cross-model retention of the NKE a source generates
(lower ⇒ more off-manifold / less shared). The ViT's NKE transfers **worst and
collapses earliest**:

| ε_l | ResNet18 source | VGG11 source | **ViT source** |
|----:|:--:|:--:|:--:|
| 5 | 0.75 | 0.66 | **0.24** |
| 16 | 0.41 | 0.21 | **0.15** |

Corroborated in the human track (generic recognizer = ResNet18): on **ViT**-sourced
NKE the recognizer falls to **0.12–0.14**, barely above the matched **Gaussian
control (0.10)** — ViT NKE is nearly as cross-model-destructive as pure noise,
whereas ResNet18 NKE stayed far above its control (0.49 vs 0.15, Track 1). The
**human–model gap is correspondingly the largest for the ViT source (~86 points:
model 1.00 vs external reader ~0.14)**. Attention-based off-manifold directions
are the least shared across architectures.

**Finding 4 — feature-distance detection is architecture-dependent and weaker on
the ViT.** On the ViT penultimate (the CLS token), **Mahalanobis detection of
oblivious NKE only reaches 0.43 → 0.50** across ε_l (vs **1.00** on ResNet18 from
ε_l = 1). MSP/energy stay blind (0.00) and ECE ≈ 0.02 — the *same* calibration
paradox — but the transformer's global CLS representation gives a far weaker
feature-distance signal. The adaptive attacker still drives Mahalanobis to **0 at
λ = 0.001** (white-box *and* held-out); success collapses faster than on the CNN
(0.66 at λ = 0.02 vs ~1.0), so the no-cost evasion band is narrower but present.
**Track 2's "Mahalanobis catches static NKE" is thus a CNN-favourable result** —
it is much weaker on a transformer and remains adaptively defeatable either way.

**Takeaway.** Adding a ViT (i) confirms the texture-≫-shape mechanism is
architecture-general; (ii) *refutes* its natural corollary — a shape-biased model
is not a better NKE reader, so shape survival ≠ recognizability; (iii) reveals
attention models generate the most model-specific NKE and the largest human–model
gap; and (iv) shows the paper's feature-distance detection result is favourable to
CNNs and degrades on transformers. See `figures/vit_shapebias_cifar10.png` and
`figures/{mechanism,overlay,calibration}_cifar10_vit.png`.

---

## Bottom line vs the paper's three questions
1. **Is there a human-model gap?** Yes on natural images — a ~50-point gap on the
   proxy at large ε_l, now **corroborated by a real N=5 human pilot** (recognition
   →~0.08, gap ~0.92); small/absent on MNIST. The *specific to adversarial
   structure, not signal loss* part holds on the proxy/CLIP but the pilot is
   underpowered to separate NKE from the Gaussian control (Track 1). A
   properly-powered human study would strengthen the signal-vs-structure claim.
2. **Is NKE an OOD/calibration phenomenon?** Yes, but only for *feature-distance*
   detectors (Mahalanobis 100%); confidence/energy/ECE are structurally blind. And
   feature-distance detection is **not adaptively robust** — an attacker who
   penalises Mahalanobis distance drives detection 100%→0% at no cost to success
   or perturbation size (Track 5).
3. **Do existing defenses help?** No — orthogonal to classical robustness (r≈0).

## Limitations
- Primary human curve is a **computational proxy**; a real **N=5 pilot**
  corroborates the gap but is small-sample (wide CIs, can't separate NKE from the
  Gaussian control). No further participants available.
- ImageNet-scale replication infeasible (no ImageNet data); complexity trend uses
  Nie et al.'s reported ImageNet transfer as the third point.
- Gram texture metric uninformative (see Track 3); GLCM used as primary.
- Input-transform defenses evaluated on a non-robustly-trained base (clean-acc
  confound noted).
- The ViT (Track 6) is a small from-scratch model (83.2% clean acc); its lower
  base accuracy is handled by the **retention** normalisation, but a larger
  pretrained ViT would strengthen the shape-bias conclusion. The shape-bias
  effect is a single-architecture negative result on CIFAR-10.
