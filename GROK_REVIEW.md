# Grok Adversarial Review
**Date:** 2026-10-07
**Model:** xai/grok-4.7
**Status:** done
**Document reviewed:** SYNTHESIS_3a_3b_3c.md

---

The recomputation is done. The headline claims do not survive the seed-level numbers.

# Adversarial review: The Clean Canvas Principle

This is not a paper that found three laws. It is a small, single-architecture, single-shift probe that found one real training effect, one tiny inference perturbation, and then narrated both into ontology. The numbers below are recomputed from `results/*_seed*.json`, `exp3b_corrected_results.csv`, and `exp3c_results.csv`, not from the writeup.

## 1. FATAL FLAWS

**1. The 3a table is mislabeled, and the primary kill test fails at the angle the paper claims to report.**

The table is titled “OOD at 45°.” The numbers are the three-angle mean (`final_ood`). Actual 45° means:

| Canvas | Claimed “45°” | Actual 45° | Rank at true 45° |
|---|---|---|---|
| WHITE | 0.3033 | 0.3032 | 2 |
| SHUFFLED | 0.2956 | **0.3087** | **1** |
| RANDOM-LABEL | 0.2866 | 0.2989 | 4 |
| PINK | 0.2870 | 0.2992 | 3 |
| SAME | 0.2809 | 0.2917 | 7 |
| DEAD | 0.2805 | 0.2951 | 5 |
| CROSS | 0.2803 | 0.2940 | 6 |

WHITE vs CROSS at true 45°: Δ = −0.0092, t = −1.13, p ≈ 0.32, signs `---++`. Not significant. Two of five seeds go the wrong way. The reported Δ = 0.023, t = 6.12, p = 0.004 is the three-angle mean, and even that t is not 6.12 — recomputed paired t is −5.83, and the archived `VERDICT.json` CI is [−0.034, −0.012], not the paper’s [−0.031, −0.015]. The appendix and the results section do not even agree with each other.

The 0.023 effect is almost entirely a 30° effect (WHITE−CROSS Δ = −0.046, t = −6.41). At 45° the ranking the law depends on does not exist. SHUFFLED beats WHITE. Calling 45° “the optimal discriminator” is inherited from `EXP3A_RESULTS.md` and is false for the ranking the synthesis uses.

**2. “Complete reversal” is an artifact of comparing different quantities, and it is not a reversal.**

3a ranks are training-condition differences on clean evaluation. 3b ranks are inference-condition differences on one fixed WHITE-trained model. Those are not the same ranking inverted. They are two different experimental factors.

Within 3b at 45°, the order is CROSS > SHUFFLED > SAME > RANDOM = DEAD > NO-NOISE > PINK > WHITE. The 3a three-angle order was WHITE > SHUFFLED > RANDOM > PINK > SAME > DEAD > CROSS. SHUFFLED is 2nd in both. DEAD is not inverted relative to RANDOM. PINK does not swap with a matched partner. This is a partial reordering of 4–5 arms, not a complete inversion. The slogan is doing the statistical work.

**3. n = 5 cannot support a 3×8 interaction, 7-arm ranking, or three named laws.**

Five paired observations, df = 4. A t of 6 on a 0.3-point accuracy delta is easy when the within-seed SD of the delta is 0.001. That is precision of a deterministic perturbation, not evidence of a general phenomenon.

The interaction was pre-registered as exploratory. The paper then promotes it to Law 3. Per seed, the “best” inference canvas at 45° is not the starred cell in a single seed:

- WHITE-trained: SHUFFLED, CROSS, DEAD, DEAD, SAME. CROSS is the mean winner and the per-seed winner once.
- CROSS-trained: RANDOM, DEAD, DEAD, DEAD, RANDOM. SAME is the mean winner and the per-seed winner never.
- SAME-trained: RANDOM, CROSS, DEAD, DEAD, RANDOM. The matched diagonal wins no seed.

DEAD, which the narrative says has “zero learned content,” is the modal per-seed winner. Mean-rank storytelling with n = 5 is how you manufacture a law the data do not contain.

**4. The “laws,” Laplacian identity, and biological confirmations are not results. They are untested metaphors asserted as mechanisms.**

∇²φ = 0 is not satisfied by adding pixelwise Gaussian noise to a natural image. The image already has structure. White noise does not make each pixel the average of its neighbourhood. A frozen random 1×1 projection of bilinear-upsampled layer-2 activations is not a gravitational potential, not a tissue niche, and not Jupiter. None of Laplace’s equation, the Smith chart, prime orbital periods, somaclonal variation, scRNA-seq stress programmes, terroir, or sleep consolidation was operationalised or tested. “CONFIRM…