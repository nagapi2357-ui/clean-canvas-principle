# The Clean Canvas Principle
### A Three-Experiment Study of Training and Inference Canvas Effects on OOD Generalisation

**Authors:** Nagaπ + Adrian Tusk  
**Status:** Adversarially reviewed (Grok + Thoth GND) — ready for community review  
**Date:** 2026-10-07

---

## Summary

We trained SmallResNet models on CIFAR-10 under seven different noise "canvas" conditions — from white Gaussian noise to structured activations from a second neural network (B-model "exhaust") — and measured out-of-distribution (OOD) generalisation under rotation.

Three findings emerged across 15 seeds, 360+ evaluations, and three experimental phases:

1. **The Clean Canvas Effect (Exp 3a):** Training on white noise (maximum entropy, no imposed structure) produces significantly better OOD generalisation than training on structured exhaust. **Δ = −2.30 pp, t = 5.83, CI [−3.39, −1.20]** for CROSS-EXHAUST vs WHITE. Exhaust training *hurts*.

2. **Stochastic Resonance at Inference (Exp 3b):** At inference time, structured exhaust *helps* OOD performance. White noise at inference *hurts* (Δ = −0.60 pp, t = 24.39, CI [−0.67, −0.53]). CROSS-EXHAUST: +0.55 pp, t = 10.45.

3. **Canvas Dependency — Exploratory (Exp 3c):** Models trained with structured exhaust show asymmetric behaviour at inference: their own training canvas helps (SAME+SAME: +0.63 pp, t = 17.02) while white noise hurts. Clean-canvas-trained models show the opposite.

> *Build in silence. Perform with signal.*

---

## Metric Note

All primary results use **Mean OOD across three rotation angles (30°, 45°, 60°)**. This was the pre-registered metric (`final_ood` in seed JSON files). An earlier version of the synthesis mislabeled this as "OOD at 45°" — corrected after adversarial review.

---

## Repository Structure

```
exp3a_exhaust_canvas.py          # Exp 3a — training canvas study
exp3b_inference_canvas_corrected.py  # Exp 3b — inference canvas (corrected v1.1)
exp3c_canvas_swap.py             # Exp 3c — canvas-swap matrix
SYNTHESIS_3a_3b_3c.md           # Full paper (Grok + Thoth reviewed)
EXP4_AI_IMPLICATIONS.md         # Exp 4 proposal: dynamic exhaust feedback
GROK_REVIEW.md                  # Grok adversarial review
results/
  VERDICT.json                  # Pre-registered kill tests, regenerated from seed files
  exp3b_corrected_results.csv   # Exp 3b inference canvas results
  exp3c_results.csv             # Exp 3c canvas-swap results
  *_seed*.json                  # Exp 3a per-arm per-seed JSON files (35 files)
```

---

## Reproducing the Experiments

All scripts are designed for Google Colab with A100 GPU. Estimated runtimes:

| Experiment | Phases | A100 time |
|---|---|---|
| Exp 3a | Train 7 arms × 5 seeds | ~3.5 hrs |
| Exp 3b corrected | Train WHITE × 5 seeds + inference eval | ~3 hrs |
| Exp 3c | Train 3 arms × 5 seeds + inference eval | ~6-7 hrs |

Model weights (~4.7 MB × 10 trained models) are not committed to this repo. Run the scripts to reproduce them.

---

## Architecture

- **Model A:** SmallResNet (1.17M params) — subject model trained on CIFAR-10
- **Model B:** Same architecture, trained on a different task/dataset — provides "exhaust" activations
- **Exhaust:** B's layer-2 activations → frozen 1×1 projection → normalised to σ=0.05 → added to A's input
- **OOD measure:** Accuracy on rotated CIFAR-10 test set (30°, 45°, 60°)

---

## Adversarial Review

This paper has passed two rounds of adversarial review before publication:

- **Grok (xAI, 2026-10-07):** Identified mislabeling of mean OOD as "OOD at 45°". Confirmed from seed JSONs. Full review: `GROK_REVIEW.md`.
- **Thoth GND (Omarchy/Hermes, 2026-10-07):** Independent recomputation from all 40 result files. Confirmed effects on mean metric. Raised framing corrections (no post-hoc Laws, PINK explanation post-hoc). All corrections applied. Stamp embedded in `VERDICT.json`.

---

## Future Work

See `EXP4_AI_IMPLICATIONS.md` for the proposed Exp 4: **Dynamic Exhaust Feedback Architecture** — testing three candidates for continual-learning and sleep/wake architectures motivated by the three findings.

---

## Citation

```
Nagaπ & Tusk, A. (2026). The Clean Canvas Principle: Training and Inference Canvas Effects 
on OOD Generalisation. collab-thoth/exp3-exhaust-canvas.
```
