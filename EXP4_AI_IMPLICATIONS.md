# Exp 4 — Dynamic Exhaust Feedback: AI Architecture Implications
## From the Clean Canvas Principle to Continual Learning

**Authors:** Nagaπ + Adrian Tusk  
**Motivation:** Direct implication of Exp 3a/3b/3c findings  
**Status:** PROPOSED — pending Thoth adversarial review of Exp 3 synthesis  
**Date:** 2026-10-07

---

## The Unasked Question

Exp 1–3 tested static exhaust: the canvas was fixed for training or fixed at inference. 
The three laws that emerged immediately suggest a fourth question:

> *What if the inference-time exhaust is not static but live — continuously updated from 
> real B-model activations as they process the same incoming stream?*

Current LLMs and CNNs freeze weights after training. Exp 3b/3c show this is leaving 
performance on the table: structured inference exhaust provides a measurable boost even 
to frozen weights. But the weights never adapt to that exhaust.

---

## What the Three Laws Predict for AI Architecture

**From Law 1 (Clean Canvas):**  
Pre-training should use maximally unstructured noise — not curated datasets alone, 
but datasets augmented with white noise canvas. Structured data augmentation (cutmix, 
mixup) may be introducing premature structured exhaust, fossilising representations 
before they generalise.

**From Law 2 (Stochastic Resonance):**  
Fixed-weight models already benefit from structured inference context — this is 
exactly what RAG, chain-of-thought, and few-shot prompting provide. 
The data predicts this benefit is real and measurable, which aligns with empirical 
results across LLMs. The new prediction: the benefit is canvas-specific (Law 3) — 
the optimal inference context is structurally similar to the model's training corpus.

**From Finding 3 (Canvas Dependency — exploratory):**  
Models may develop implicit exhaust dependency matching their training corpus. 
A model trained primarily on code may perform better when its inference context 
contains code exhaust (related activations), not natural language exhaust. 
This would explain why domain-specific fine-tuning helps even when the task 
is held constant.

---

## Three Architecture Candidates for Exp 4

### Architecture A — Exhaust-Gated Continual Learning

**Idea:** Weights update at inference, but only when the incoming exhaust signal 
exceeds a structure threshold (not white noise).

**Mechanism:**
- B processes the incoming stream → exhaust score = mean activation magnitude
- If exhaust_score > threshold: small Δw update (Hebbian or gradient step)
- If exhaust_score ≤ threshold: weights frozen (clean canvas mode)

**Testable prediction from Law 1+3:** The gating threshold should match the 
training noise level (σ = 0.05 in our experiments). Below this, the canvas is 
effectively clean and the model should not update.

**Neural parallel:** Synaptic long-term potentiation (LTP) requires a minimum 
stimulation threshold. Below-threshold activity leaves synaptic weights unchanged.

---

### Architecture B — Coupled Exhaust Inference (Static Weights, Live B)

**Idea:** Weights frozen (as in our Exp 3b/3c), but B processes the SAME input 
stream as A in real time. B's activations are projected and added to A's input 
for every inference batch.

**This is the corrected version of our Exp 3b setup** (where we erroneously used 
test images for B). The corrected 3b showed inference benefits even with B 
processing independent data; live coupling should amplify this.

**Testable prediction:** Live-coupled B (B sees same input as A) vs independent B 
(B sees its own training data, as in corrected 3b). If live coupling produces 
correlated exhaust, and correlated exhaust helps, live coupling should outperform 
independent exhaust.

**Caution from Exp 3b bug:** The original Grok 3b used live-coupled B and produced 
confounded results. Architecture B must carefully control for input correlation 
in the evaluation. The corrected 3b showed live coupling HURT vs independent — 
the correlation confound is real.

---

### Architecture C — Sleep/Wake Architecture (Alternating Clean and Exhaust Phases)

**Idea:** Explicitly separate training (clean canvas, weight update) from inference 
(exhaust present, weights frozen) with periodic reconsolidation cycles.

**Phase 1 — Wake (inference):** Fixed weights, structured exhaust from live B.
**Phase 2 — Sleep (consolidation):** Small dataset of OOD examples, white noise canvas, 
gradient update. NO exhaust during sleep.
**Cycle:** Repeat.

**Prediction from Law 1+3:** The sleep phase must use white noise (clean canvas) 
to maintain OOD generalisation. If sleep uses the same structured exhaust from 
inference, Law 3 predicts the model will fossilise further into its current 
inference canvas — the opposite of generalisation.

**This is a direct testable prediction of the canvas dependency finding.**

---

## Minimal Viable Exp 4 (SmallResNet, CIFAR-10)

**Design:** 3-arm comparison using SmallResNet architecture

| Arm | Training | Inference | Update |
|-----|----------|-----------|--------|
| CONTROL | WHITE (clean canvas) | NO-NOISE | None (frozen) |
| 3b-REPLICATE | WHITE (clean canvas) | SAME-EXHAUST (live B on A's input) | None (frozen) |
| SLEEP-WAKE | WHITE (clean canvas) → Sleep cycles | SAME-EXHAUST (live) | Periodic: white noise phase |

**Primary measure:** OOD accuracy at 45° over time (n inference batches)
**Prediction:** SLEEP-WAKE > 3b-REPLICATE > CONTROL at all timepoints
**Kill criterion:** SLEEP-WAKE's OOD advantage grows over time (vs degradation for others)

**Resource estimate:** ~3 hrs on A100 per cycle. 10 sleep/wake cycles = ~30 hrs.

---

## Connection to Current AI Practice

| Our finding | Current practice | Gap |
|---|---|---|
| Law 1: Train clean | Pre-training on raw internet text | Web text IS structured exhaust — may be fossilising representations |
| Law 2: Infer with exhaust | RAG, few-shot, CoT prompting | Already using inference exhaust — consistent with our finding |
| Finding 3: Canvas dependency | Domain-specific fine-tuning | Matches: fine-tuning = adding canvas-specific structure |
| Sleep/Wake architecture | RLHF fine-tuning cycles | RLHF uses human feedback (structured) — but does the fine-tuning phase use clean canvas? |

**The key open question for LLMs:** Does structured pre-training data (code, math, books) 
foster canvas-specific OOD dependency? If a model pre-trained on code OODs poorly on 
natural language tasks, and structured inference context (code in the prompt) helps — 
this is Law 3 in production. We may already be seeing it and not naming it.

---

## Pre-Registration Notes

- Primary hypothesis: Architecture C (Sleep-Wake) maintains OOD generalisation better 
  than Architecture B (static exhaust) over multiple inference/update cycles
- Null: No significant difference after 5 sleep/wake cycles
- Architecture A (exhaust-gated continual learning) is exploratory in Exp 4; 
  promote to primary hypothesis in Exp 5 if signal appears
- B-model quality to be equalised across arms (addressing Grok's confound critique)

---

*Pending Thoth adversarial review of Exp 3 synthesis before committing to Exp 4 protocol.*
