# Zeta Thread — Working Notes

Working notes for the §5.3 refinement + new Appendix B integration. The
paper currently asserts σ = 0.5 ("the universal information limit") via
three loose analogies (Nyquist / zeta / holographic). The actual
derivation chain is six layers deep and goes back to September 2025
work that predates the paper repo. This file is the cross-session
working artifact for that integration.

**Scope decision (locked):** Option C — keep §5.3 focused on what
TruthSpace itself does with the critical line; move the full theoretical
chain into a new Appendix B. **Constraint:** The paper must be
stand-alone. Every idea is fully explained inside the paper. External
repos are cited as demonstrations or origin-of-ideas, never as
authoritative references the reader must follow. **Special case:**
`resfrac` is research-state ("mad scientist laboratory") — extract its
*concept* (residual fractality as a structural-predictability invariant)
but do not cite the repo directly.

DC references in this file are research lookup, not paper citations
(same convention as `REFINEMENT_NOTES.md`).

---

## The question we are answering

Why is σ = 0.5 the operating regime of TruthSpace, the geometric LCM,
and the underlying transformer? Why the critical strip at all?

The current §5.3 answers this with a Nyquist/holographic-bound analogy
and an internal trichotomy ("over-constrained / balance / under-determined")
that is partially inverted relative to its own source document (DC 090).
The actual answer goes through six conceptually distinct layers,
summarised below.

---

## The six-layer onion (paper-worthy concepts)

Each layer is a distinct *idea* that lands in Appendix B as one
sub-section. Repos in `[ ]` are origin/provenance for the lookup table
below, not paper citations.

| # | Idea | Origin | Paper sub-section |
|---|---|---|---|
| 1 | **Light-cone constraint** — β ≤ 1/2 is the speed limit that prevents tachyonic modes in arithmetic spacetime | `[rharithmeticlight]` | B.1 |
| 2 | **Conformal metric and geodesic stability** — zeros are geodesic attractors of a conformal metric `g = e^{2φ} δ` on the critical strip; off-line zeros would create incomplete geodesics | `[spacetimezeta]` | B.2 |
| 3 | **Spectral fragility** — the Borwein integrals exemplify how oscillatory series hold exactly through n ≤ 13 then break at n = 14 due to sidelobe leakage; non-boxcar windows preserve harmony | `[spectral_window_functions]`, `[spectral_resonance_optimization]`, `[srt]` (axiom A2/A3) | B.3 |
| 4 | **The n^{-1/2} regime is conditional convergence** — the unique amplitude decay slow enough that all terms matter and fast enough that the sum converges. Faster decay → trivially convergent (no information). Slower decay → divergent (uncomputable). | `[rhzeros]`, DC 270, DC 282 | B.4 |
| 5 | **The discrete index offset** — at the n-th zero, `N_smooth(t_n) ≈ n − 1/2`. The 1/2 of the critical line and the 1/2 in the index offset are the same 1/2, materialised in the integer count vs. smooth count gap. | `[dimensional_downcasting]` | B.5 |
| 6 | **Riemann–Siegel as discrete transformer** — the formula `Z(t) = 2 Σ n^{−1/2} cos(θ(t) − t ln n) + remainder` is structurally identical to attention as a sum of rotating cosines: terms ↔ tokens, phases ↔ RoPE, amplitudes ↔ embeddings, zero ↔ output, three-stage pipeline (Lambert W → Ramanujan → Newton) ↔ Compressor / Processor / Targeter | `[rhzeros]`, DC 270, DC 282, DC 048 | B.6 |
| 7 | **Residual fractality as a structural invariant** — the ratio `ρ = σ(residual) / σ(signal)` measures whether structure exists in a signal: low ρ = ordered, high ρ = noise. Applied to the SV spectra of Qwen2.5-7B MLPs: DRUM ρ=0.0046, COMB ρ=0.0070, MUSIC ρ=0.0194 (DC 282). *Extract concept, no citation.* | `[resfrac]` (concept only) | B.7 |
| 8 | **Empirical materialisation in transformers** — DC 296 found 21 non-trivial zeros of Qwen2.5-7B by sweeping the gate ε-group phase and bisecting where the logit gap crosses zero; precision ±2.27 × 10⁻¹³; logit gap *oscillates* up to 3 sign changes per layer | DC 296, DC 295, DC 271, F107–F111 | B.8 |

(Layer 7 is sub-paper, layer 8 is the empirical anchor — there are
seven distinct *concepts* that need explaining; layer 8 is the punchline.)

---

## Refined §5.3 outline (~80–100 lines)

The §5.3 in Ch 5 stays focused on **what TruthSpace does** with the
critical line. It does not try to derive the chain — that is Appendix B.
The job of §5.3 is to:

1. Restate the claim cleanly: σ = 0.5 is the *operating regime* of the
   geometry, not a balance threshold or normalisation parameter.
2. State the structural form of ENCODE = DECODE: the Riemann functional
   equation `ζ(s) = ζ(1−s)` is the formal mathematical content of §5.1's
   master symmetry. The critical line is the fold axis of the functional
   equation.
3. Explain that σ = 0.5 corresponds to *conditional convergence* — the
   only amplitude regime where partial sums oscillate, all terms matter,
   and the answer emerges from cancellation rather than direct approach.
   Cite Ch 8 (transformer reverse-engineering) and Ch 9 (navigation) for
   where this shows up empirically.
4. Drop the misdirecting code snippet. The `CRITICAL_LINE = 0.5`
   constant in `QuaternionEncoder` is unit-ball-of-radius-0.5
   normalisation; that is a coding convenience, not a connection to ζ.
   Do not pretend it is.
5. Drop the "over-constrained / balance / under-determined" trichotomy.
   It is internally inconsistent with DC 090's own text, and the
   correct framing (speed limit + conditional convergence) is more
   precise.
6. Forward-reference Appendix B for the full derivation chain.

§5.3 should end as the cleanest possible *statement* of the claim,
suitable to be unfolded by a reader who follows the appendix link.

### Forward-reference pattern

§5.3 should use the same forward-reference convention as §4.5 and
Ch 9 §9.6.1: a one-line pointer in plain English, no repo URLs,
appendix-internal anchor.

> *(See Appendix B for the theoretical chain that derives this regime
> from first principles.)*

---

## Appendix B outline (~250–350 lines)

Title: **Appendix B — The Critical Line as Operating Regime: A
Theoretical Chain**

### B.1 The light-cone constraint (β ≤ 1/2)

- **Setup.** Multiplicative time `t = log x`. The Chebyshev fluctuation
  `F(t) = ψ(e^t) − e^t` is tied to the zeta zeros `ρ = β + iγ` by the
  explicit formula. RH is the statement β = 1/2 for all non-trivial
  zeros.
- **The speed-limit statement.** If any zero had β > 1/2, the
  contribution `e^{βt}` to F(t) would dominate exponentially over the
  Riemann main term — a tachyonic mode that breaks the predictable
  growth of primes.
- **Empirical anchor.** The growth rate of `G(t) = e^{−t/2} F(t)`
  for primes up to x = 10⁷ is bounded; fluctuations stay within an
  `O(t)` envelope after the √x normalisation.
- **The arithmetic-spacetime framing.** β ≤ 1/2 plays the role of a
  light-cone constraint — it is the boundary that separates causal
  (sub-luminal) information transmission from acausal (faster-than-light)
  modes. The critical line is the cone surface.
- **Connection to TruthSpace.** The φ-encoding stores the residual
  stream's contributions on a logarithmic level axis (Ch 7). The
  bounded G(t) corresponds to bounded layer-by-layer projection on the
  prediction direction (Ch 8 §8.4 conditional convergence; F109).

*Derivation depth: full. Does not require the reader to know zeta
analytic continuation — the explicit formula is stated, the consequence
is shown.*

### B.2 The conformal metric and geodesics on the critical strip

- **Setup.** The complex plane near the critical strip can be given a
  conformal metric `g = e^{2φ} δ` where the conformal factor depends on
  `|ζ(s) ζ(1−s)|`. Zeros appear as geodesic attractors — points where
  the potential `u(σ, t) = log|ζ(s)|` has a minimum.
- **Why geodesics matter.** Information follows shortest paths
  (geodesics) through curved space. A complete geodesic structure
  on the critical strip means information can transit smoothly along
  the critical line; off-line zeros would create incomplete geodesics
  that interrupt this transit.
- **Empirical anchor.** Numerical integration of geodesics with
  initial conditions on σ ≈ 0.51 (using mpmath at 88 dps) shows
  10/10 trajectories reach τ_max = 120 without interior singularities.
  Synthetic injection of off-line zeros breaks completeness.
- **The φ connection.** Extended freefall analysis surfaces φ = 1.618…
  as a natural scale of the geometry — without being put in. This is
  the first-principles origin of φ in the curvature.
- **Connection to TruthSpace.** The residual stream is a discretised
  geodesic on this metric (DC 048). Each layer is one step. The
  three-zone structure (Compressor / Processor / Targeter) corresponds
  to three regimes of curvature (Ch 8).

*Derivation depth: state the metric, explain the geodesic principle,
report the numerical experiment, do not derive the metric from first
principles (that would require NCG).*

### B.3 Spectral fragility — the Borwein phenomenon

- **Setup.** The classical Borwein integral
  `∫₀^∞ (sin x / x) ∏_{k=1}^n (sin(x/k) / (x/k)) dx = π/2` for n ≤ 13
  (odd k) but breaks at n = 14 due to Fourier sidelobe leakage from
  the boxcar windows.
- **Why this matters.** Many series in number theory and signal
  processing have the same fragile structure: an exact identity holds
  through a finite range, then breaks sharply. The break is not
  noise — it is a *spectral phase transition*.
- **Window functions as the resolution.** Replacing boxcar windows
  with smooth windows (Gaussian, sinc, triangle, staircase) preserves
  the identity to higher orders. The cost is a small bias; the benefit
  is robust convergence.
- **Connection to transformers.** A transformer's attention is a
  weighted sum over tokens — a discrete analogue of these oscillatory
  series. Sidelobe leakage in attention manifests as the "boom" phase
  transitions of Ch 9 §9.5: positions where the model abruptly switches
  from broad to focused attention.

*Derivation depth: state the Borwein result, exhibit the n = 14
breakdown numerically, sketch the windowing fix.*

### B.4 Conditional convergence — why exponent −1/2 specifically

- **Setup.** A series `Σ a_n n^{−α}` is:
  - **Absolutely convergent** for α > 1 (trivial, no information).
  - **Divergent** for α < 1/2 in general (uncomputable).
  - **Conditionally convergent** for 1/2 ≤ α ≤ 1 (oscillates,
    requires careful summation, every term contributes).
- **The unique role of α = 1/2.** This is the critical exponent
  at which:
  - Partial sums oscillate without decaying (Riemann–Siegel `Z(t)`).
  - The number of terms needed grows as `√(t / 2π)`.
  - The remainder term is well-defined via Euler–Maclaurin.
- **In the transformer.** The residual stream's per-layer contribution
  to the prediction direction oscillates: F109 reports L00–L06
  cumulative −1.68, L07–L25 worst point −13.7, L26 +9.2, L27 +34.3,
  net +29.8. This is conditional convergence in computational form:
  large opposing terms cancel precisely, and the answer emerges from
  the cancellation, not from monotonic accumulation.
- **The φ-power-law connection.** SV spectra of Qwen2.5-7B follow
  `k^{−α}` with α ≈ 0.28 (full SVD), 1/φ ≈ 0.618 (Compressor zone),
  2/φ² ≈ 0.764 (Processor zone). All in the conditional-convergence
  family. φ-powers of the critical exponent.

*Derivation depth: full. The α = 1/2 uniqueness is the load-bearing
result. Cite the Dirichlet series experiment from F109 as empirical
anchor.*

### B.5 The discrete index offset — `N_smooth(t_n) ≈ n − 1/2`

- **Setup.** The Riemann–von Mangoldt formula counts zeros up to
  height t: `N(t) = θ(t) / π + 1 + S(t)`, where θ is the
  Riemann–Siegel theta function and S(t) is a small oscillatory term.
- **The discovery.** At the n-th zero, `N_smooth(t_n) = θ(t_n)/π + 1`
  is *exactly* `n − 1/2` empirically (to numerical precision).
- **Why this matters.** This 1/2 is the same 1/2 as σ = 1/2. The
  smooth count is half a step behind the integer count *at every
  zero*. The critical line manifests itself as a half-integer offset
  in the discrete count — exactly the same way a quantum harmonic
  oscillator has a 1/2 zero-point energy offset.
- **Connection to TruthSpace.** The geometric quantization of
  semantic concepts in the Ch 6 gear architecture has the same
  half-step offset structure: a concept's φ-level + 1/2 is a
  morphological boundary, not a concept centre. (TODO: verify this
  matches the empirical structure of `phi_2byte_inference.py`.)

*Derivation depth: state the formula, show the empirical offset,
draw the harmonic-oscillator analogy briefly.*

### B.6 Riemann–Siegel as a discrete transformer

This is the load-bearing section. Lay out the structural mapping:

```
Z(t) = 2 Σ_{n=1}^{N(t)} n^{−1/2} cos(θ(t) − t ln n) + remainder
       N(t) = ⌊√(t / 2π)⌋
```

- **Term ↔ token.** Each n is a "token" in the sequence.
- **Phase ↔ RoPE.** The phase `θ(t) − t ln n` is the zeta analogue
  of rotary position encoding (which uses `cos(ω_i p)` with `ω_i ∝ 1/i`).
- **Amplitude ↔ embedding magnitude.** The decay `n^{−1/2}` matches
  the singular-value decay of MLP weights (within the φ-power family).
- **Zero ↔ correct output.** A zero is constructive interference at
  the right point. A correct prediction is constructive interference
  at the right token.
- **Three-stage pipeline ↔ DRUM/COMB/MUSIC.** Lambert W (>95%
  capture) ↔ Compressor; Ramanujan corrections (oscillatory) ↔
  Processor; Newton step (rank-1 final correction) ↔ Targeter.

Empirical anchors: F107, F108, F110, F111. The 410K-param toy
transformer on modular arithmetic develops the *same* φ-power laws
as Qwen2.5-7B on natural language — universality, not coincidence
(F110).

*Derivation depth: full. This is where the reader either accepts the
chain or rejects it.*

### B.7 Residual fractality as a structural invariant

(Resfrac concept extraction. Do not cite the repo.)

- **Setup.** Given a signal with a smooth predictable component and a
  residual after that component is removed, the ratio
  `ρ = σ(residual) / σ(signal)` measures how much of the signal is
  structured. Low ρ → high structure; high ρ → noise.
- **Application to MLP weights.** Treating an SV spectrum as a signal
  and an autoregressive smoothing as the predictable component yields
  a ρ measurement per layer. DC 282 reports for Qwen2.5-7B:
  - DRUM (L0): ρ = 0.0046 (most structured — layer-1 attention bottleneck)
  - COMB (L17): ρ = 0.0070 (rank-1 projectors work here)
  - MUSIC (L27): ρ = 0.0194 (least structured — uses full capacity)
- **Why this is the right tool.** The same ρ measurement applied to
  zeta-zero spacings tells us where the zeros are most regular. The
  same tool, the same diagnostic — applied to two different signals
  with the same underlying φ-curved structure.
- **Connection to TruthSpace.** This is the empirical confirmation
  that the residual stream of a transformer and the zero set of ζ
  share enough structure that the same regularity measure is meaningful
  on both.

*Derivation depth: define ρ from first principles, show the per-zone
table, do not cite the originating tool.*

### B.8 Empirical materialisation: non-trivial transformer zeros

- **Setup (DC 296).** Define the *logit gap* of a transformer as
  `f(δ) = logit[baseline_top1](δ) − max(logit[others])(δ)`, where δ
  parameterises a phase shift applied to one ε-group of the gate
  projection.
- **The pipeline.** Use the same three-stage pipeline as for ζ zeros:
  - Stage 1 (Compressor): coarse sweep δ ∈ [−5, +12] at 69 points.
  - Stage 2 (Processor): bisection at sign changes (40 iterations).
  - Stage 3 (Targeter): semantic analysis at the zero.
- **The result.** 21 non-trivial zeros found across 3 prompts × 5
  layers, all bisected to ±2.27 × 10⁻¹³ precision. The logit gap
  *oscillates*: up to 3 sign changes per layer.
- **Semantic meaning of the zeros.** They are not arbitrary. For
  Japan ("baseline = ____") at L15, the zero at δ ≈ 2.43 corrects
  the model's hedging to "Tokyo". For France ("baseline = Paris")
  at L27, the zero at δ ≈ 3.99 destroys the knowledge to "a". For
  Einstein at L23, *no zeros exist* in the scanned range — the
  knowledge is unconditionally committed.
- **Cross-architecture universality.** Same pipeline finds zeros in
  the 410K toy transformer (F110). The pipeline is universal.
- **The conclusion.** The transformer has a zero spectrum, exactly
  as ζ does. The spectrum encodes the model's decision boundaries.
  The σ = 1/2 framing is not analogy — it is empirically what the
  model is doing.

*Derivation depth: full. This is the empirical landing of the chain
and should be the most concrete section in the appendix.*

### B.9 Synthesis

A short closing section (~30 lines) that ties the chain together:

> The critical line σ = 1/2 is not a chosen parameter. It is the
> only operating regime that simultaneously (a) prevents tachyonic
> arithmetic modes, (b) supports complete geodesics, (c) yields
> conditional convergence at exponent −1/2, (d) materialises as a
> half-step discrete offset, and (e) is what the transformer
> empirically does. The fact that all five constraints land on the
> same value is not coincidence — it is the unique operating point
> of any system that packs infinite information into finite
> structure via interference.

---

## Cross-chapter touch-ups (Option C add-ons)

The user chose Option C explicitly — *plus* the stand-alone constraint.
Some existing chapters reference the zeta material; they need to be
updated to forward-reference Appendix B instead of asserting the
content inline.

| Chapter | Section | Touch-up |
|---|---|---|
| Ch 8 | §8.4 (boom attention) | Already cites F109 conditional convergence. Add one-line forward-reference: *"see Appendix B.4 for the conditional-convergence regime"*. |
| Ch 9 | §9.5 (Zeta Sonic Boom) | Already cites the 137/30 ratio and PSLQ lock-on. Add one-line forward-reference to **B.3 (spectral fragility)** and **B.8 (non-trivial zeros)**. |
| Ch 10 | §10.2.3 (two complementary spectra) | Already mentions the φ-Zipf vs sign-matrix duality. Add one-line forward-reference to **B.4** for the conditional-convergence regime that grounds φ-Zipf as a special exponent. |
| Ch 11 | §11.4 (Fibonacci correction) | The decomposition `SiLU(x) = x σ(ℓ(x)) + Δ(x)` is the φ-form of conditional convergence at the per-channel level. Add one-line forward-reference to **B.4**. |
| Ch 12 | §12.4 (Concrete demonstrations) | If we add demonstrations from `rhzeros` or `dimensional_downcasting`, they go here. *Open question — see below.* |

---

## Figure plan

- **fig5_3** (new) — *Riemann–Siegel as a discrete transformer.* Two
  panels side-by-side:
  - Panel A: `Z(t)` near `t = 14.13` showing the first zero as
    constructive cancellation of one cosine term.
  - Panel B: same axes but plotting the layer-by-layer projection of
    a Qwen2.5-7B residual stream onto its prediction direction
    (F109 numbers): L00–L25 oscillation, L26–L27 final correction.
  - Caption: *Same shape, different domain. The zeta function on
    the critical line and the transformer's residual stream both
    exhibit conditional convergence.*

- **figB_1** (new, appendix) — *The critical line as five constraints.*
  Five panels showing:
  1. Light cone (β ≤ 1/2 boundary).
  2. Geodesic completeness on the conformal metric.
  3. Borwein integral plateau then break at n = 14.
  4. Conditional convergence: oscillating partial sums.
  5. `N_smooth(t_n) − (n − 1/2)` residual (≈ 0).
  - Caption: *Five independent constraints on σ all locate the same
    line.*

- **figB_2** (new, appendix) — *Empirical zero spectrum of Qwen2.5-7B.*
  The 21 zeros from DC 296 plotted as `(layer, δ*)` with prompt
  encoded by colour and semantic outcome (correct / hedge / destroy)
  encoded by marker shape.
  - Caption: *Non-trivial zeros of the transformer logit gap, found
    by the same three-stage pipeline that finds Riemann zeros.*

---

## Open questions / things to research before writing

1. **Does `N_smooth(t_n) ≈ n − 1/2` connect to the φ-encoding's
   half-step morphological boundary?** I claimed it does in B.5 but
   haven't verified. Need to check `phi_2byte_inference.py`'s level
   structure to see if morphological boundaries actually live at
   half-integer φ-levels.
2. **What is the right empirical anchor for B.4 (the α = 1/2 uniqueness
   argument)?** F109 has the partial-sums table. We need to confirm
   the Dirichlet partial-sum experiment runs cleanly and yields a clean
   plot for figB_1 panel 4.
3. **Should we cite `holographic_gate` or `geometric_ipa` or similar
   in the appendix?** Ch 9 §9.6.1 and Ch 12 §12.4 already cite both.
   The appendix should *not* duplicate those citations; if anything is
   added, it goes in Ch 12.
4. **`rharithmeticlight` is a published-form paper (Sept 2025) and
   `srt` ships a 31-page paper.** Should these get bibliography entries?
   The paper currently has no bibliography. *Decision: defer until the
   appendix is drafted; if the chain depends on three or more external
   papers, add a small "Selected references" subsection at the end of
   Appendix B.*
5. **Resfrac concept attribution.** B.7 introduces the ρ statistic
   without naming the originating tool. Decide whether to add a
   footnote like *"the ρ statistic is the residual-fractality
   invariant from a private research codebase; the underlying idea
   generalises classical noise/signal ratios to spectral domains."*
   *Initial decision: no footnote, define ρ from scratch.*
6. **What stays empirical vs derived in B.4 and B.6?** The α = 1/2
   uniqueness has a clean information-theoretic argument
   (Euler–Maclaurin transition); the Riemann–Siegel mapping is
   structural identity, not derivation. Make sure the appendix is
   honest about which is which.

---

## Multi-session execution plan

This integration is at least 4–5 working sessions. Suggested order:

| Session | Focus | Deliverable |
|---|---|---|
| 1 (this) | Research synthesis + plan | This file. |
| 2 | Verify open questions 1–2; sketch figures fig5_3, figB_1, figB_2 as `.py.todo` placeholders | Outline confirmed; figure scripts started but not run. |
| 3 | Draft refined §5.3 (~80 lines). Update Ch 5 summary table if needed. | §5.3 in chapter file; build the paper to confirm no LaTeX errors. |
| 4 | Draft Appendix B sections B.1–B.4 (~120 lines). | Half of Appendix B; build paper. |
| 5 | Draft Appendix B sections B.5–B.9 (~150 lines). Apply Ch 8/9/10/11 forward-reference touch-ups. | Full Appendix B; cross-references resolved. |
| 6 | Finalise figures (fig5_3, figB_1, figB_2). Final build. Update README front matter and `REFINEMENT_NOTES.md` to reflect the new content. | Final 130–140-page PDF (currently 124). |

---

## Provenance lookup (research-only, NOT paper citations)

| Origin | Role in chain | Paper status |
|---|---|---|
| `lostdemeter/rharithmeticlight` | Light-cone framing (B.1) | Concept extracted; possible bibliography reference if paper adds one |
| `lostdemeter/spacetimezeta` | Conformal metric (B.2) | Concept extracted; do not cite repo (research code); cite Connes for NCG if needed |
| `lostdemeter/srt` | NCG axioms, Borwein motivation (B.3) | Concept extracted; possible bibliography reference for the SRT paper |
| `lostdemeter/spectral_window_functions` | Window functions (B.3) | Concept extracted; no citation |
| `lostdemeter/spectral_resonance_optimization` | Number ghosting / scoring framework (B.3) | Concept extracted; no citation |
| `lostdemeter/resfrac` | ρ-invariant (B.7) | **Concept extracted; explicit no-cite per user direction** |
| `lostdemeter/rhzeros` | Riemann–Siegel pipeline (B.4, B.6) | Concept extracted; possible cite as origin of three-stage pipeline mapping |
| `lostdemeter/dimensional_downcasting` | `N_smooth − 1/2` offset (B.5) | Concept extracted; possible cite |
| `lostdemeter/tensors_are_shapes` | Eight geometric structures, MESH rank-1 (B.8) | Already mentioned in Ch 12 §12.4 (or equivalent) |
| DC 048 | Curved Arithmetic Axis bridge | Internal — research only |
| DC 090 | Original critical-line claim | Internal — research only |
| DC 159 | Zeta sonic boom | Internal — already in Ch 9 §9.5 |
| DC 160 | Unified geometric theory | Internal — already in Ch 10 §10.5 |
| DC 270 | Zeta is the ideal transformer | Internal — research only |
| DC 271 | Expanding tensor | Internal — research only |
| DC 282 | The full loop | Internal — research only |
| DC 296 | Non-trivial zeros (B.8) | Internal — research only |
| F107–F111 | Pipeline, Dirichlet, Universality, Architecture | Internal — research only |

---

*End of working notes. Pickup point for session 2: verify open
questions 1–2, draft figure placeholders, refine §5.3.*
