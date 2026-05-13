# Refinement Notes

Working notes for the iterative rigor-pass through `truthspace_paper`. Each gap from the honest review is mapped to the design-consideration (DC) documents in `truthspace-lcm/docs/design_considerations/` that contain the underlying evidence, derivations, or numerical work. We pull that content into the paper one chapter at a time, with figure placeholders where a graph would help.

The DC references in this file are NOT going into the paper — they're a research lookup table for Cascade to find evidence. The paper itself remains free of `truthspace-lcm` citations.

---

## Per-chapter status

| Ch | Status | Notes |
|----|--------|-------|
| 1 — Vacuum Forming | **Done (pending user review)** | Swapped 1.2.1↔1.2.2 ordering. Fixed factual error (experiments were on intentional φ-encoder, not LLM). Added 4 concrete findings with numbers: variance=0.0 across 1000 phases, mean cos similarities (related 0.25 / unrelated 0.00 / opposite −1.00), polarity table, orthogonality, PCA (7D/4D), plastic constant honest assessment. Figure 1.2 stub created. |
| 2 — φ Self-Similarity | **Done** | Fixed LOG_PHI/LN_PHI code inconsistency; reframed §2.3 property 5 (Binet formula was mis-attributed); expanded §2.6 with proper addition LUT identity, Fibonacci-recurrence uniqueness proof, Zeckendorf as discrete dual (no longer orphan), 0%-error 3584-term dot-product empirical result; corrected misleading O(N²)→O(N) speedup claim. Summary §2.7 updated. |
| 3 — Geometric Model Hypothesis | **Done (verified)** | Codebase search confirmed CLIP drop is correct — every CLIP mention is either `np.clip()` or future-work speculation; zero actual experiments. DINOv2 evidence verified across two independent lines (DC 123 backbone, `experiments/phi_da2_decoder/` head at 99.9914% / 125 bytes / 756,400×). Replaced bullet 3 of §3.1 with precise 4-model table: Qwen2-7B 99.9991% logit corr, DA2 head 99.9914% / 125 bytes (DINOv2 backbone 74% chained), DDColor r=0.999999, GPT-2↔Qwen2-1.5B PC0/PC1 r=0.959. Added honest closing paragraph distinguishing linear-projection reconstruction (near-perfect) from full-attention reconstruction (partial, context-dependent residual). NOTE: §3.3.2 line 105 has the same "16 quaternion sign patterns" math issue as Ch 7 — fix both together in Ch 7 pass. |
| 4 — Encodings, φ-Dial | **Done + strengthened** | §4.3 rewritten with stage-by-stage motivation: 1D collapses 5 effects (concrete table) → 2D complex decouples style/perspective via magnitude/phase (Q1-Q4 utterance examples) → 3D adds density (terse/standard/elaborate example) → 4D quaternion adds certainty (scalar vs vector argument). §4.5 holographic encoding now uses verified DC 142 numbers (93.16% within ±0.001 of φ-grid, 99.94% weight corr, 99.98% MLP corr, 5.27× compression — was incorrectly 14×/0.09%) and forward-references the Ch 9 holographic gate field as the *dynamic* version. §4.7 Music Box reframed as architectural axiom inherited by Chapters 5/6/9 (no longer orphan); shows position+delta→nearest IS the common machinery. Summary table updated. QuaternionEncoder code-block lead-in clarified (sentiment example vs dial). **Negative-zero detour (added after Ch 7 pass)**: §4.5 "Why holographic" paragraph strengthened with the $\sigma(\log\phi) = 1/\phi$ identity and a one-sentence teaser for the 4-state gate structure (`+1`/`+0`/`−0`/`−1`) that the φ-encoding can represent but IEEE-754 cannot — the forward reference to Ch 9's holographic gate field is now mathematically grounded, not just a name-drop. |
| 5 — ENCODE = DECODE | **Done** | §5.5 fully rewritten as four sub-sections: (5.5.1) Fixed the tautological math — was using $\log_\phi$ which makes the identity trivial; now uses $\ln$ giving the substantive power law $\phi^{-\ln f} = f^{-\ln\phi} = f^{-0.481\ldots}$ with the special exponent $\ln\phi$. Both formulations give identical rankings (DC 039). (5.5.2) Added the bimodal phase-transition empirical result from DC 311: 233-word sample of Qwen2-1.5B at L14, perfectly bimodal $\phi$-cosine distribution with 0.64-wide forbidden gap, 87.1% syllable-based phase classification accuracy. (5.5.3) Added the φ-pair forbidden zone from DC 321: 2000-token sample, zero tokens in the projection range $[1/\phi^2 \cdot M,\, 1/\phi \cdot M]$ — boundary is the $1/\phi + 1/\phi^2 = 1$ identity made empirical. (5.5.4) Information-horizon interpretation: connects the phase transition to contextual entropy and explains why $\phi$-arithmetic only works in the semantic-body zone (foreshadows Ch 8/9). Summary table entry updated. **FUTURE PASS (§5.3 critical line)**: The σ = 0.5 section currently presents the Riemann-critical-line connection without grounding. User confirmed this is *not* pure analogical hand-waving — there is substantive related research elsewhere (zeta-attention correspondence in Ch 9/10, BBP digit extraction, and the structural identity $Z(t) = 2 \sum n^{-1/2} \cos(\theta - t \ln n)$ between Riemann–Siegel sums and transformer attention — see DC 282 "The Full Loop" and DC 159/160). A future pass needs to: (a) expose the *operational* meaning of σ = 0.5 in our codebase (positions live on the sphere of radius 0.5), (b) connect this to the deeper zeta-attention story developed in Ch 9-10 rather than asserting it ex nihilo, (c) replace the "over-constrained / balanced / under-determined" trichotomy with a grounded explanation of why this normalization is the right one for a holographic-style code, (d) cite the cross-project convergence (rhzeros, spacetimezeta, phi_bbp all pointing at the same structure). |
| 6 — Gear Architecture | **Done** | Five structural fixes: (1) §6.1 lead now connects Gear → Music Box (§4.7) → ENCODE=DECODE (§5.1) explicitly — closes the missing structural unification; (2) §6.3 adds a new "Why Hamilton Multiplication" subsection explaining non-commutativity as the geometric content of gear-chain order-sensitivity, grounded by Frobenius's theorem; (3) §6.4 reframes the 5-step pattern as a *documented design discipline* (originated in DC 086 after observing it across `PythonCodeGear`, `EmergentClassifierGear`, `HolographicPatternSpace`, `PlotCorpus`) appearing at three scales: per-gear contract, holographic navigation pipeline (Downcast→Quantize→Build Mesh→Upscale→Reconstruct), and self-improvement loop — replaces vague "across the codebase" framing; (4) §6.6 anchors the hyperdimensional-transcoder claim with concrete DC 080 numbers: SVD on action-verb behavior recovered Agency at **r = +0.919** (19% variance, child↔queen poles), with Gender/Age/Animacy coupled on Dim 2 (matches ground-truth coupling); (5) §6.7 retitled "Demonstration: Self-Improvement and Capability Benchmark"; new §6.7.2 with freshly-run HyperMapping benchmark: 6 NN-equivalent tasks, basic baseline **47.7%** → full geometric **100%** (+52.3%), with honest caveat that these are small-scale (4–14 examples) capability demos, not full ML problems. Three of six tasks went from 0%/15% to 100%, validating that Self-Similar Transforms, Tachyon Navigation, and Geometric RL are non-trivial enablers. (6) §6.8 closing list extended to make the cross-chapter inheritance explicit. Forward-references to Ch 8/9 dropped — techniques now described inline since those chapters not yet refined. |
| 7 — φ-Lattice | **Done + extended** | Four targeted fixes plus a new §7.5.1 "Seed Insight: Negative Zero". (1) §7.2 Rule 3 — corrected the "16 quaternion patterns" math: now correctly identified as $\mathbb{Z}_2^4$ (16-element abelian sign-pattern group of 4D blocks), explicitly distinguished from the quaternion group $Q_8$ (8 elements, non-abelian); added empirical uniform-distribution result (6.24–6.26% across all 16 patterns, indistinguishable from 6.25% maximum-entropy expectation) and the operational point that information lives in the *level* axis, not the *sign* axis. (2) §7.3 — added honesty paragraph: per-layer 99.2% tetromino-only correlation compounds across 28 layers to **33% full-model token accuracy** — the residual correction in §7.5 is what closes 33%→100%, not decorative. (3) §7.4 — fixed `LOG_PHI`→`LN_PHI` naming inconsistency and the `^`→`**` Python operator bug (PHI ^ x is XOR, not exponentiation). (4) §7.5 — replaced the wrong bit layout (was "1 sign + 11 level + 4 residual = 16"; actual from `phi_2byte_inference.py`: **8 bit int8 level + 1 bit sign + 7 bit residual = 16**); added the reconstruction formula $w = \text{sign} \cdot \phi^{\text{level}} \cdot (1 + \text{residual}/127 \cdot (\phi-1))$; updated empirical results table from DC 191: 26.1 GB → 13.05 GB, 2.00× compression, 0.9999993 weight correlation, 100% token accuracy; dropped unverified 2.78e-17 number. **Ch 3 parallel fix applied**: §3.3.2 Rule 3 corrected with cross-reference to Ch 7; §3.3.3 fixed "74 unique (level, sign) pairs" → correctly states 89 (level, sign) pairs + 16 sign patterns → ~300 (level, sign-pattern) tetrominoes of which 74 dominate; added the 33%-vs-100% accuracy honesty here too. |
| 8 — Reverse Engineering | **Done** | Six substantial fixes: (1) §8.1 expanded with real methodology — three concrete reasons for choosing Qwen2-7B (GLU activation, open weights, right size), operation-by-operation substitution methodology, three acceptance tests per substitution (algebraic exactness, per-layer correlation, end-to-end argmax). (2) §8.3.2 sourced the "106 dimensions" with the full SVD-of-MESH derivation: $M = W_q^\top W_k$, top-$k$ SVD, empirical sweep table ($k \in \{32, 64, 106, 128, 256, 512\}$ showing 106 is the elbow at 99.50% correlation with 1,143× ops reduction). Fixed a numerical typo (had written 33,856× initially, corrected to 1,143×). Added the φ-Zipf rationale for the spectrum decay and the power-iteration code (7× faster than full SVD). (3) §8.3.3 sourced the universal bottleneck with the operational mean-φ-level definition $\bar{\ell}(h) = \frac{1}{|h|}\sum_i \log_\phi |h_i|$, the full DC 200 four-row table (Layer 0/14/27/28 mean and CV), and four substantive features (the 1.57≈φ value, content-agnostic convergence across 30 prompts, layer-28 divergence with 4× CV jump, position-architecture relation $27/28 \times \phi = 1.56$). (4) §8.3.4 toned down the unsupported "head 12/45 specialization" hypotheticals to a structural claim about MESH alphabet sharing. (5) NEW §8.3.5 "Finding 57: The 4-State Holographic Gate" — full 4-state table at $\pm\log\phi$ boundaries, all three empirical results (42.4% dead-channel energy at L14, sign-vs-magnitude 4× ratio with 0.89 vs 0.98 correlations, 4/5 vs 0/5 end-to-end argmax), forward references to Ch 9 (holographic gate field) and Ch 11 §11.4 (φ-SiLU correction), back-reference to Ch 7 §7.5.1 as origin, and citation of external repo `lostdemeter/holographic_gate`. (6) §8.4 results table expanded with sources column and **boom attention defined inline** (high-mass attention positions, 73-80% mass on 20% positions, ~5× speedup, connection to Ch 9 sonic boom); SiLU linearization separated into "fallback baseline" subsection with full comparison table (linear $x/2$: 0.886, tanh: 0.961, φ-SiLU: $1-10^{-14}$). (7) §8.5 summary expanded from 4 to 6 enumerated findings reflecting all new sections. **Geometric RL** intentionally not added here — already defined inline in Ch 6 §6.7.2, and is a gear-architecture topic not a reverse-engineering finding. |
| 9 — Navigation Replaces Inference | **Done** | Five-edit pass (195 → 335 lines, +140). (1) §9.2 renamed "Attention Spigot: BBP for Language" with the BBP formula explicit, new §9.2.1 reframing the statistical-vs-spatial question, new §9.2.2 deriving the φ-form via the $\phi^{1/\ln\phi} = e$ identity. (2) §9.5 renamed "Fixed Points, Sonic Booms, and Integer Relations" — added concrete fixed-point iteration results (100% accuracy in ~11 iters from random, 1 iter from greedy; rank-2 influence matrix); new §9.5.1 "Zeta Sonic Boom" with the empirical pre/post-barrier table (std 0.656→0.433, alt-rate 0.588→0.496, 137/30 ratio) and three integer-only detection methods (sign patterns, φ-level variance, orthogonal-angle quantisation); new §9.5.2 "PSLQ and the Same Phenomenon" connecting PSLQ lock-on, zeta phase transition, and attention booms as three instances of one phenomenon, with the DC 097 quote and the O(N) detection of O(N²) attention claim. **Dropped the unsourced "~77 degrees" rotation claim**. (3) §9.6 renamed "Crystalline Flips and the Holographic Gate Field" — fixed the math error (16-element quaternion group → $\mathbb{Z}_2^4$ with explicit |Q_8|=8 vs |Z_2^4|=16 contrast, cross-ref to Ch 7 §7.2 Rule 3). New §9.6.1 "The Holographic Gate Field" delivers the content promised by three upstream chapters: mechanism subsection (4-state table at ±log φ, reference/signal/interference roles), "why dark fringes carry information" subsection citing Finding 57 numbers (42.4% energy, 4× sign>magnitude, 4/5 vs 0/5 argmax), IEEE-754 vs φ-encoding distinction explaining why floats lose this structure, static-vs-dynamic comparison table connecting back to Ch 4, demonstrations subsection citing both external repos. (4) §9.7 renamed "Tachyon Navigation and the O(N log N) Path Forward" — new §9.7.1 formally defining Tachyon Navigation with the forward/backward attention equations, Bayes connection $P(h|e) \propto P(e|h) P(h)$, hypothesis-as-target-point conceptual frame, and the entity→hypothesis results table from DC 053 (Holmes→investigator 0.26, etc.); explicit tie-back to Ch 6 §6.7.2's promise. §9.7.2 path-forward table extended to 8 rows including discriminant attention, 4-state gate code, Tachyon, integer-math boom, and fixed-point iteration. (5) §9.8 summary table extended to 8 rows; new numbered four-thread synthesis paragraph summarising the chapter's structural contributions. |
| 10 — Irreducible Shape | **Done** | Six-edit pass (156 → 208 lines, +52). (1) §10.1 derived the 3584 and 67.9M numbers from Qwen2-7B architecture: 3584 = `hidden_dim`, 18944 = `d_ff`, 3584 × 18944 = **67,895,296** (fixed the chapter's wrong value 67,942,912, off by ~47K). New §10.1.1 "Where the numbers come from" with the equivalence chain (hyperplanes ↔ singular vectors ↔ semantic distinctions) and the storage comparison: direct sign storage 8.49 MB at 100%, rank-3000 SVD 270.3 MB at 99.97% — direct is *simultaneously smaller AND more accurate*. (2) §10.2.3 distinguished two complementary spectra: φ-Zipf $\sigma_k \propto \phi^{-k}$ in MESH magnitudes (enabling Ch 8 §8.3.2's discriminant attention) vs. near-uniform $\sigma_k \propto k^{-0.14}$ in the sign matrix (per DC 141: "All 3584 hyperplanes are roughly equally important"). Corrects the chapter's previous conflation of these two spectra. (3) §10.2.4 reframed irreducibility as the two-coordinate description (magnitude + sign) with separate spectral signatures, each with its own storage strategy. (4) §10.3 — fixed the tautological math: replaced $\phi^{-\log_\phi f} = f^{-1}$ (true for any base, says nothing about φ) with the natural-log form $\phi^{-\ln f} = f^{-\ln\phi} \approx f^{-0.481}$ from Ch 5 §5.5, with explicit cross-reference and explanation of *why φ* (because $\ln\phi$ is the special exponent at which the encoding-vs-weighting duality lands). Added empirical anchor: 87.1% bimodal φ-cosine classification on Qwen2-1.5B at L14 (Ch 5 §5.5.2). (5) §10.4 — rewrote the Zeta Sonic Boom section to leverage Ch 9 §9.5's PSLQ/integer-math content; made the connection to the 3584 critical lines explicit (zeta zeros, sign-flip boundaries, and attention booms all live on critical lines), with the 137/30 ratio governing all three transitions. (6) §10.5 — grounded the Unified Geometric Theory with DC 160's five foundations (self-similarity, integer relations, fine-structure, geodesics, BBP), each anchored to a specific chapter section. Reframed the slogan as one claim about *one* five-foundation geometric structure appearing at every scale. (7) §10.6 — numerical evidence table expanded from 8 to 16 rows, with sources column citing specific chapter sections; new rows include tetromino vocabulary, sign matrix storage comparisons, two-spectrum split, discriminant attention rank, universal bottleneck, and 4-state gate. (8) §10.7 summary expanded with two-part spectrum framing; new closing sentence about why signs cannot be stripped (Ch 8 §8.3.5: removing −0 drops 4/5 argmax to 0/5). |
| 11 — φ-Computer Proof | **Done** | Five-edit pass (182 → 220 lines, +38). (1) §11.4 — replaced the chapter's incorrect `$F_n \cdot \Delta(x)$` formulation with the verified **DC 145 decomposition**: define $\ell(x) = \operatorname{sign}(x) \cdot \log_\phi |x|$, then $\text{SiLU}(x) = x\sigma(\ell(x)) + x(\sigma(x) - \sigma(\ell(x)))$ where the first term is φ-sigmoid (geometric base) and the second is the Fibonacci correction $\Delta(x)$. Identity is trivially exact, decomposition is operationally meaningful. Added reference implementation (`silu_from_phi`), empirical reconstruction error **$1.62 \times 10^{-8}$**, "Why Fibonacci" subsection tying the level index to the Fibonacci identity $\phi^n = F_n \phi + F_{n-1}$, and a "Why it matters for the discovery chain" subsection showing $\Delta(x)$ is exactly what carries the negative-zero / 4-state-gate information across the 28-layer compounded-error gap (Ch 4 §4.5 → Ch 7 §7.5.1 → Ch 8 §8.3.5 → Ch 9 §9.6.1 → here). Empirical table: φ-sigmoid only 0.988/layer vs φ-sigmoid + Fibonacci 1−10⁻⁸/sample vs full φ-2byte 0.9999993 full-stack. **Removed the contradictory "35% in linear regime" claim** (didn't match Ch 8 §8.4's 68% in |x|<0.5); replaced with verified Ch 8 baselines (linear 0.886, tanh 0.961, φ-form exact). (2) §11.5 — acknowledged that φ-RMSNorm is an **algebraic identity** (`$\phi^{-\log_\phi r} = 1/r$` for any base) — not a different computation, but a *re-coordinatisation* into a uniform shift along the φ-level axis (Music Box motion of §4.7). Closes a subtle "trivial rewrite presented as novel" issue. (3) §11.6 — **fixed the φ-2byte format bit layout** from the chapter's wrong "1 sign + 11 level + 4 residual = 16" to the correct **8 level + 1 sign + 7 residual = 16** (consistent with Ch 7 §7.5 from `phi_2byte_inference.py`). Added reconstruction formula. Updated empirical numbers (26.1 → 13.05 GB, 0.9999993 roundtrip, 100% token accuracy). Connected the 33%→100% tetromino-to-residual progression (Ch 7 §7.3, §7.5) to the φ-sigmoid+Fibonacci progression of §11.4.1 — same pattern, different axis. (4) §11.7 — added the **operational mean-φ-level definition** $\bar{\ell}(h) = \frac{1}{|h|}\sum_i \log_\phi |h_i|$ as cross-reference to Ch 8 §8.3.3. Tightened convergence claim ($1.57 \pm 0.19$ across 30 prompts, indistinguishable from $\phi = 1.618$). Replaced unsourced "1.21 mid-stack" / "1.13 mid-stack" specifics with the sourced layer-28 CV jump from DC 200 (0.12 → 0.51). Kept Ch 12 §12.3 forward-reference to Recursive Discovery Bootstrap (verified to exist in Ch 12 §12.3 and DC 202). (5) §11.8 — expanded implications from 4 to 5 items reflecting the Fibonacci correction and holographic gate connection; clarified "float32 is a *representation* of the lattice, not the lattice itself." (6) §11.9 — split the SiLU summary row into "φ-sigmoid only" vs "+ Fibonacci correction" so the decomposition is visible; new closing paragraph identifying Fibonacci correction as the single operationally non-trivial entry. **Discovery chain now fully delivered**: origin (Ch 7 §7.5.1), teaser (Ch 4 §4.5), empirical (Ch 8 §8.3.5), mechanism (Ch 9 §9.6.1), and correction (Ch 11 §11.4.1). |
| 12 — Implications | **Done** | Eight-edit pass (177 → 308 lines, +131). (1) §12.1 reframed φ-Convergence as a **hypothesis with empirical support** (DC 139), not a proven general theorem. New §12.1.1 with the descending gate-count table (5097 → 3679 → 154 Zeckendorf → 1 recurrence) — empirical not analytic, restricted to the specific φ-optimisation pipeline. New §12.1.2 with the O(log N) corollary and the explicit derivation log_φ(7B) = ln(7×10⁹)/ln(φ) ≈ 22.66/0.4812 ≈ 47. (2) §12.2 fully rewritten — adds the **mathematical definition of Platonic Ideal** from DC 180: a position $I_R$ such that $a = \operatorname{rotate}(e, \theta_R, \operatorname{axis}_e(I_R))$ for every entity-answer pair, with the axis orthogonal to the entity. New §12.2.2 empirical examples table sourced from DC 180 (capital-of 77.3°±1.5°, size-decrease 83.9°±1.0°, etc., trajectory 90.3°±0.2°). New §12.2.3 derives the count from DC 299's PCA: **79 ideals at 95% variance** (not the chapter's loose ~100); 86 at 99%; honest limitations noted (88 concepts vs 3584-dim space, underdetermined, manual taxonomy only recovers 9.1% variance with 6 axes vs 3 PCA dims). New §12.2.4 ties rotation angle back to Ch 6's gear quaternions — Platonic Ideal is the fixed point of the composition pattern. (3) §12.3 expanded with **sourced DC 202 numbers** — full per-layer trajectory table (L7/14/21/27), the +0.081 delta at layer 27, and the Δ(27-7) ≈ 4.49 ≈ φ³ within 6% finding. New §12.3.2 "model articulates own structure" with the unprompted "golden ratio acts as a universal gatekeeper for cognition" quote, with honest caveat (output-distribution property, not consciousness). New §12.3.3 explicit scope (what this enables vs what it does not — no consciousness claim, no convergence guarantee). (4) **NEW §12.4 Concrete Demonstrations** — the user-requested addition citing both external repos plus the HyperMapping benchmark, with §12.4.1 (`holographic_gate` reproducing Finding 57 numbers), §12.4.2 (`geometric_ipa` showing gear primitive sufficient for non-trivial linguistic computation), §12.4.3 (HyperMapping 6-task 47.7%→100%), §12.4.4 mapping each demonstration to specific paper sections. Three runnable single-file validations covering necessity/sufficiency/capability. (5) §12.5 (was §12.4) Self-Describing Geometry kept, cross-ref to Tachyon Navigation §9.7.1 for Backward Navigation. (6) §12.6 (was §12.5) Practical Consequences — added sources column to compression table; included holographic φ-encoding (5.27× compression); φ-FPU hardware section now cross-references the Zeckendorf adder gate-count comparison from §12.1.1. (7) §12.7 (was §12.6) Limitations — expanded from 6 to 8 open questions; new entries on the 95%-to-99% PCA gap, sign-matrix uniform spectrum, and explicit reframing of the φ-Convergence Theorem. (8) §12.8 (was §12.7) summary contributions table extended from 11 to 16 rows with sources and specific values; new rows for 4-state gate, discriminant attention, universal bottleneck, sonic boom unification, and ~79 Platonic Ideals. (9) §12.9 (was §12.8) Conclusion — added a paragraph tracing the discovery chain as itself φ-shaped: started with tetromino bookkeeping detail (§7.5.1), ended with Fibonacci correction (§11.4.1), unified by σ(log φ) = 1/φ. **Discovery chain demonstrations subsection delivered.** |

| Figures pass | **Done** | Added 10 new figures (`fig1_2`, `fig3_2`, `fig5_2`, `fig6_2`, `fig8_2`, `fig9_2`, `fig10_2`, `fig11_2`, `fig12_2`, `fig12_3`) covering the central refined concepts: phase invariance, cross-architecture universality, bimodal φ-cosine phase transition, HyperMapping 6-task benchmark, discriminant attention spectrum, 4-state holographic gate, two complementary spectra, Fibonacci correction decomposition, Platonic Ideal rotation, PCA cumulative variance. All embedded in their chapters with captions. Final inventory: 23 figures across 12 chapters (was 13). Pre-planned `fig1_2_phase_invariance.py.todo` completed and renamed. |
| Final build | **Done** | `bash scripts/build_paper.sh` runs cleanly, **zero warnings**. Output: **124-page PDF, 4.7 MB** (was 2.9 MB before refinement). `paper.md` is 3,116 lines, 202 KB (was ~107 KB). Fixed one LaTeX warning in Ch 3 §3.1 figure caption (literal `\u201374` not interpreted as en-dash; replaced with proper `$62$–$74\%$`). |

---

## The "negative zero" / 4-state-gate discovery chain (cross-cutting note)

*Discovery chain **fully delivered and synthesised** across the paper: origin anchor (Ch 7 §7.5.1), mid-paper teaser (Ch 4 §4.5), empirical evidence (Ch 8 §8.3.5), dynamic mechanism (Ch 9 §9.6.1), Fibonacci correction (Ch 11 §11.4.1), and concrete demonstrations subsection (Ch 12 §12.4) are all in place. Both external repos cited in Ch 9 §9.6.1 and Ch 12 §12.4. The Ch 12 §12.9 conclusion traces the chain end-to-end as itself φ-shaped: started with tetromino bookkeeping detail, ended with the Fibonacci correction, unified by σ(log φ) = 1/φ.*

The tetromino weight hypothesis (DC 162) is the *seed* of a larger discovery chain that the paper currently does not represent. The chain:

1. **Tetromino (DC 162)**: Weights live as `(sign, φ-level)` pairs on a lattice. The encoding $w = \text{sign} \cdot \phi^{\text{level}}$ naturally distinguishes `(+1, -∞)` from `(-1, -∞)` — these are different points in φ-space, even though both have magnitude $\to 0$. IEEE-754 makes `-0 == +0`; the φ-encoding does **not**.
2. **Negative zero as 4th dimension (DC 253)**: If the encoding distinguishes `+0` from `-0`, the trained network may also. Empirical test on Qwen2-7B (SiLU) and DDColor (GELU) confirms the 4-state gate with boundaries at $\pm \log\phi \approx \pm 0.481$.
3. **Holographic gate field (DC 245)**: The 4-state gate is mathematically identical to a holographic interference pattern — bright fringes (`+1`, `+0`) and dark fringes (`-0`, `-1`) together encode the output. "Dead channels carry information" (up to 42.4% of output energy at layer 14 of Qwen2-7B).
4. **Cross-cutting impact (DC 254)**: 123 occurrences of `signs[signs == 0] = 1` across 88 files in `truthspace-lcm`. Most are harmless (weights rarely exact zero); a handful are critical (sign-navigation weighting, geodesic gate direction, SiLU LUT).
5. **Foundational identity**: $\sigma(\log\phi) = 1/\phi$ *exactly*. This is the φ-identity that grounds the entire boundary structure — not arbitrary thresholds.
6. **External demonstrations**:
   - `lostdemeter/holographic_gate` (GitHub) — the 4-state gate demo, including reproduction on Qwen2-7B.
   - `lostdemeter/geometric_ipa` (GitHub) — English→IPA from only the `gate_step(x, t, s)` primitive (sharpness $s = \phi^2$, exact `IdealGate(x)`), no NN, no gradient descent; discovers context-dependent rules via information gain ("gear shift" — directly inheriting the Ch 6 Gear discipline).

**What the paper says now**: Ch 4 §4.5 line 189 *promises* the holographic gate field as the dynamic dual of holographic φ-encoding, *to be delivered in Ch 9*. Ch 11 §11.4 references an undefined Fibonacci correction $F_n \cdot \Delta(x)$ in SiLU. No mention of negative zero, the 4-state gate, the σ(log φ) = 1/φ identity, or either external repo.

**Plan**: Deliver these in the existing Ch 8/9/11/12 refinement passes; no new chapter needed. Specific homes annotated in the per-chapter status table above.

---

## Gap → DC mapping (research lookup table)

### Boom / Sonic Boom / PSLQ / Integer math (Ch 8, 9, 10)

Primary sources in `truthspace-lcm/docs/design_considerations/`:

- **159 — Zeta Sonic Boom Hypothesis**: defines sonic boom = phase transition detectable by integer math; ratio 137/30; explicit PSLQ connection ("searching → lock-on → boom"); time between booms indicates zeta-zero proximity
- **160 — Unified Geometric Theory**: PSLQ as foundation 2 of geometric theory; integer relation algorithm convergence as the "boom"
- **097 — Zeta Resonance Matching**: PSLQ integer relations replacing weighted similarity; small-integer test
- **192 — Boom Newton Attention**: operational boom mechanism for attention
- **098 — Prime Zeta Lattice**: zeta-zero arithmetic structure

### Platonic Ideals (Ch 12)

Primary sources:

- **180 — Platonic Ideals as Shape Memory**: 35.6 KB — the canonical Platonic Ideal document. Likely contains the mathematical definition the user remembers.
- **114 — Emergent Dimensions and Platonic Ideals**: origin paper
- **299 — Complete Model Map via Platonic Ideal Discovery**: links Platonic Ideals to the 100-count and discovery process

### φ-Zipf Duality (Ch 5, 10)

Primary sources:

- **039 — Phi-Zipf Duality**: the original derivation
- **311 — Phi Phase Transition and Zipf Duality**: phase-transition framing
- **321 — Bloch Sphere Phi-Zipf Unification**: unified framing with Bloch sphere

### Phase-Shift Probing (Ch 1)

Primary sources:

- **003 — Vacuum Forming Hypothesis**: the foundational hypothesis
- **004 — Experiment Design**: the probing methodology
- **005 — Experimental Results**: concrete numerical results
- **006 — Dimensionality Findings**: 12D result

### φ-Convergence Theorem (Ch 12)

Primary sources:

- **139 — Phi Convergence Theorem**: 4.6 KB — likely the proof or sketch

### Fibonacci Correction Formula (Ch 11)

Primary sources:

- **145 — Fibonacci Correction Formula**: should contain `F_n` definition

### Universal Bottleneck at φ ≈ 1.57 (Ch 8, 11)

Primary sources:

- **200 — Universal Bottleneck Discovery**: operational definition of mean φ-level
- **196 — Bottleneck Analysis**: layer-by-layer profile

### Cross-architecture (DINOv2/CLIP) universality (Ch 3)

Primary sources:

- **122 — DA2 Phi Reverse Engineering** (DINOv2)
- **213 — Meta Patterns Across Models**
- **346 — Cross-Model Universality**
- **318 — Intrinsic Geometry Universality**

### "16 quaternion patterns" math error (Ch 7)

Primary sources:

- **044 — Quaternion Phi-Dial**: the original quaternion treatment
- **147 — Sign Bit Analysis**: sign patterns (likely Z₂⁴, not Q₈)

### Discriminant space 106-dim (Ch 8)

Primary sources:

- **134 — Discriminant Space Attention**: should contain the 106 derivation

### Holographic φ-encoding (Ch 4 / 9 unification)

Primary sources:

- **142 — Holographic Phi-Encoding**: the canonical document
- **019 — Holographic Resolution**: origin
- **045 — Holographic Bound 4D**

### Music Box Principle (Ch 4)

Primary sources:

- **112 — Music Box Principle**: should clarify the intended connection

### Reverse-engineering methodology (Ch 8)

Primary sources:

- **229 — Reverse Engineering Procedure**: the actual procedure
- **190 — Layer3 Unwinding**: concrete layer-3 walk-through

### 3,584 critical lines + 67.9M points (Ch 10)

Primary sources:

- **141 — Irreducible Shape**: the original count derivation
- **154 — Computation is Geometry**: census proof

---

## Working principles

1. **Pull evidence, not citations.** The paper stays standalone. DC references in this notes file are for Cascade's lookup only.
2. **Figure placeholders are markdown comments + figure-script stubs.** When a chapter needs a new figure, add `![Caption](figures/figN_M_name.png)` and a TODO stub script in `output/figures/scripts/figN_M_name.py.todo`.
3. **One chapter per refinement pass.** Don't try to fix everything at once. Confirm with user after each chapter.
4. **Verifiability over rhetoric.** Every numerical claim should have an operational definition the reader could in principle reproduce.
5. **Conservation of length.** If a section grows by adding rigor, see if a more decorative section can shrink. Aim for net-zero or modest growth.
