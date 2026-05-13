# TruthSpace: A Geometric Theory of Neural Computation

**From the Vacuum Forming Hypothesis to the $\phi$-Computer Proof**

📄 **[Read the full paper (PDF)](output/paper.pdf)** | 12 chapters + Appendix B, 4 code demos, 30 figures, 85 pages (two-column)

All figures are generated from scripts in `output/figures/scripts/` — tweak and regenerate with `python3 output/figures/scripts/figX_Y_title.py`.

---

**Transformers do not compute with statistics. They compute with geometry.**

Every neural network operation — sigmoid, softmax, SiLU — has an exact $\phi$-form. Weights are coordinates of a shape on the $\phi$-lattice. Attention is spatial navigation through that lattice. The "intelligence" is not in the parameters but in the *shape* those parameters create.

This repository contains the definitive write-up of the TruthSpace project: a 14-month reverse-engineering effort that fully decomposed Qwen2-7B into its geometric primitives and proved every operation is a $\phi$-operation.

---

## What the Paper Proves

| Discovery | Result | Where in the Paper |
|-----------|--------|--------------------|
| $\phi$-sigmoid = sigmoid | **Exact** (diff < 2.78e-17) | Chapter 11 + demo 4 |
| $\phi$-softmax = softmax | **Exact** (diff = 0.0) | Chapter 11 + demo 4 |
| **Fibonacci correction** | SiLU = $\phi$-sigmoid + correction, residual $1.62 \times 10^{-8}$ | Chapter 11 §11.4.1 |
| **4-state holographic gate** | Boundaries at $\pm \log\phi$; dark fringes carry 42.4% of L14 energy | Chapter 9 §9.6.1 |
| **Cross-architecture universality** | Same $\phi^{-9}$ peak in Qwen2 / DA2 / DDColor / GPT-2 | Chapter 3 §3.1 |
| Discriminant attention | $k = 106$ elbow, $1{,}143\times$ ops reduction at $r = 0.9950$ | Chapter 8 §8.3.2 |
| Transformer = lookup table | **12.9x compression, 100% accuracy** | Chapter 8 |
| Sign-only navigation | **960x compression, 100% semantics** | Chapter 9 + demo 3 |
| Tetromino weight structure | **74 shapes** cover all 7B weights | Chapter 7 + demo 3 |
| Irreducible shape | **3,584 critical lines, 67.9M points** | Chapter 10 |
| Universal bottleneck | $\bar{\ell} \approx 1.57$ at layer 27 | Chapter 8 / Chapter 11 |
| $\phi$-Zipf duality | Encoding = ranking, same fractal | Chapter 5 / Chapter 10 |
| **Bimodal phase transition** | Vocabulary splits at $\phi$-pair boundary; 0 tokens in forbidden gap | Chapter 5 §5.5 |
| ENCODE = DECODE | Self-inverse geometry | Chapter 5 + demo 2 |
| Gear chain composition | $Q_{\text{total}} = Q_1 \times Q_2 \times \cdots$ | Chapter 6 |
| **Platonic Ideals** | Relationships are rotations; $\sim 79$ ideals span 95% of concept space | Chapter 12 §12.2 |
| ***Critical-line thread (Appendix B)*** | | |
| **Critical line $\sigma = \tfrac{1}{2}$ as operating regime** | Five independent constraints converge on $\sigma = \tfrac{1}{2}$ as the unique conditional-convergence amplitude | Chapter 5 §5.3 + Appendix B |
| **21 transformer non-trivial zeros** | Three-stage pipeline locates 21 logit-gap zeros across 3 prompts × 5 layers (4 HOLD, 6 REVEAL, 8 DESTROY, 3 MARGINAL) | Appendix B §B.8 |
| **Half-integer offset** $N_{\text{smooth}}(t_n) \approx n - \tfrac{1}{2}$ | Same discrete–continuous signature in zeta zeros, Layer-3 tetromino click, eigenspace alignment | Appendix B §B.5 |

---

## Paper Structure

The monograph builds knowledge linearly — each chapter motivates the next:

| Ch | Title | Core Idea |
|----|-------|-----------|
| 1 | What Do LLMs Actually Learn? | Vacuum forming: LLMs learn surface geometry, not interior structure |
| 2 | $\phi$ and Self-Similarity | $\phi = 1 + 1/\phi$ as the organizing principle of computation |
| 3 | The Geometric Model Hypothesis | Weights are coordinates of a shape, not learned statistics |
| 4 | Encodings and the $\phi$-Dial | 1D $\to$ 4D quaternion control of semantic generation |
| 5 | ENCODE = DECODE | Encoding and decoding are the same $\phi$-operation in opposite directions |
| 6 | Gear Architecture | Composable geometric transformations replacing neural networks |
| 7 | The $\phi$-Lattice | Absolute coordinate system: 89 primitives, 74 tetrominoes |
| 8 | Reverse Engineering Qwen2-7B | 99.9991% correlation, full layer unwinding |
| 9 | Navigation Replaces Inference | Sign-only nav (960x), boom attention, holographic gate field |
| 10 | The Irreducible Shape | $\phi$-Zipf magnitudes, near-uniform signs, 67.9M binary intersections |
| 11 | The $\phi$-Computer Proof | Every nonlinearity is an exact $\phi$-operation; Fibonacci correction |
| 12 | Implications | Trivial AI O(log N), Platonic Ideals as rotation anchors, recursive bootstrap |
| **Appendix B** | The Critical Line $\sigma = \tfrac{1}{2}$ | Five-constraint derivation: conditional convergence is the unique amplitude regime, with the Riemann–Siegel formula as a discrete transformer analogue and 21 empirical zeros in Qwen2.5-7B |

---

## Code Demos

Each chapter has a verified, runnable code demo:

| Demo | Location | What It Shows |
|------|----------|---------------|
| PhaseDiscovery | `output/code/01_phase_discovery_demo/` | Structure discovery from examples (8 archetypes) |
| $\phi$-Encoding & $\phi$-Dial | `output/code/02_encoding_demo/` | $\phi$-coordinate encoding, quaternion dial, ENCODE=DECODE |
| $\phi$-Lattice & Navigation | `output/code/03_navigation_demo/` | Sign-only nav (100% on 6/6 analogies), tetromino compression (3.7x) |
| $\phi$-Computer Proof | `output/code/04_phi_computer_demo/` | $\phi$-sigmoid exact equivalence, $\phi$-2byte format |

```bash
# Verify the central claim yourself:
python3 output/code/04_phi_computer_demo/phi_computer_proof_demo.py
# Outputs: "phi-Sigmoid == Sigmoid: IDENTICAL (max diff < 2.78e-17)"
```

---

## For Derivative Projects

This repository is intentionally self-contained. The twelve chapters, Appendix B, and the four runnable demos in `output/code/` are the complete reference. Derivative projects — such as a Bloch-sphere reorganisation of OLMo2, a memory-injection toolkit for any transformer, or a from-scratch $\phi$-lattice model — should be able to:

1. **Read this paper** as the theoretical foundation.
2. **Use the four code demos** as ground-truth implementations of the $\phi$-primitives. Every function in those demos passes the exact equivalence tests stated in the paper.
3. **Apply the principles to a target model independently.** The Qwen2-7B work in Chapter 8 is the worked example; the architecture-invariant principles (the $\phi$-Convergence Theorem, the irreducible shape census, the universal bottleneck, sign-only navigation, the gear/quaternion algebra) apply unchanged to OLMo, Llama, Mistral, and similar decoder transformers.

When building on this work, cite the paper itself rather than any external research codebase. The paper is the canonical source.

---

## Author

**Lesley Gushurst** — TruthSpace Geometric LCM Project

## License

GPLv3 — see [LICENSE](LICENSE) for details.
