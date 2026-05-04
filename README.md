# TruthSpace: A Geometric Theory of Neural Computation

**From the Vacuum Forming Hypothesis to the $\phi$-Computer Proof**

📄 **[Read the full paper (PDF)](output/paper.pdf)** | 12 chapters, 4 code demos, 12 figures

All figures are generated from scripts in `output/figures/scripts/` — tweak and regenerate with `python3 output/figures/scripts/figX_Y_title.py`.

---

**Transformers do not compute with statistics. They compute with geometry.**

Every neural network operation — sigmoid, softmax, SiLU — has an exact $\phi$-form. Weights are coordinates of a shape on the $\phi$-lattice. Attention is spatial navigation through that lattice. The "intelligence" is not in the parameters but in the *shape* those parameters create.

This repository contains the definitive write-up of the TruthSpace project: a 14-month reverse-engineering effort that fully decomposed Qwen2-7B into its geometric primitives and proved every operation is a $\phi$-operation.

---

## What the Paper Proves

| Discovery | Result | Source (Doc) | Code Verification |
|-----------|--------|-------------|-------------------|
| $\phi$-sigmoid = sigmoid | **Exact** (diff < 2.78e-17) | 191 | `phi_computer.py` |
| $\phi$-softmax = softmax | **Exact** (diff = 0.0) | 191 | `phi_components.py` |
| Transformer = lookup table | **12.9x compression, 100% accuracy** | 187 | `tetromino_*.py` |
| Sign-only navigation | **960x compression, 100% semantics** | 165 | `sign_only_navigation.py` |
| Tetromino weight structure | **74 shapes** cover all 7B weights | 162 | `tetromino_fast_inference.py` |
| Irreducible shape | **3,584 critical lines, 67.9M points** | 141 | `measure_complexity.py` |
| Universal bottleneck | $\phi \approx 1.57$ at layer 27 | 200 | `automated_discoveries.json` |
| $\phi$-Zipf duality | Encoding = ranking, same fractal | 039 | `phi_dial_experiment.py` |
| ENCODE = DECODE | Self-inverse geometry | 061 | `generation.py` (ReverseEngine) |
| Gear chain composition | $Q_{\text{total}} = Q_1 \times Q_2 \times \cdots$ | 086, 095 | `gear.py`, `hypermapping.py` |

---

## Paper Structure

The monograph builds knowledge linearly — each chapter motivates the next:

| Ch | Title | Core Idea | Key Sources |
|----|-------|-----------|-------------|
| 1 | What Do LLMs Actually Learn? | Vacuum forming: LLMs learn surface geometry, not interior structure | 001-006 |
| 2 | $\phi$ and Self-Similarity | $\phi = 1 + 1/\phi$ as the organizing principle of computation | 010, 124, 137 |
| 3 | The Geometric Model Hypothesis | Weights are coordinates of a shape, not learned statistics | 022, 127, 154 |
| 4 | Encodings and the $\phi$-Dial | 1D $\to$ 4D quaternion control of semantic generation | 041-044, 067 |
| 5 | ENCODE = DECODE | Encoding and decoding are the same $\phi$-operation in opposite directions | 061, 089-091 |
| 6 | Gear Architecture | Composable geometric transformations replacing neural networks | 033, 086, 095 |
| 7 | The $\phi$-Lattice | Absolute coordinate system: 89 primitives, 74 tetrominoes | 099, 162, 163 |
| 8 | Reverse Engineering Qwen2-7B | 99.9991% correlation, full layer unwinding | 129, 186, 190, 191 |
| 9 | Navigation Replaces Inference | Sign-only nav (960x), boom attention, fixed points | 161, 165-167, 175 |
| 10 | The Irreducible Shape | $\phi$-Zipf spectrum, 67.9M binary intersection points | 039, 141, 154, 159 |
| 11 | The $\phi$-Computer Proof | Every nonlinearity is an exact $\phi$-operation | 145, 191, 199, 200 |
| 12 | Implications | Trivial AI O(log N), recursive bootstrap, Platonic ideals | 140, 180, 202 |

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

## Source Codebase

The research implementation lives at **[github.com/lostdemeter/truthspace_lcm](https://github.com/lostdemeter/truthspace_lcm)** and includes:

- **`phi_geometric/`** — PhaseDiscovery, CascadeNavigator, ReverseEngine, $\phi$-encoder
- **`unwound_transformer/`** — Qwen2-7B reverse engineering, $\phi$-computer proof, tetromino analysis
- **`src/phi_navigator/`** — Sign-only, geometric, and zeta-based navigation engines
- **`hypermapping/`** — Pure position-based geometric knowledge store
- **`phi_adapter/`** — Universal geometric model reconstruction
- **`phi_chat/design_docs_workspace/`** — 207+ design documents charting 14 months of discoveries

> **Note on the state of `truthspace_lcm`:** That repository reflects 14 months of active, exploratory research and is currently in an organic, pre-reorganization state. It was not designed as a clean reference implementation — it is a working research environment that accumulated structure as ideas evolved. Some diagrams and modules are provisional, naming conventions are inconsistent across phases of the project, and not all components are expected to run out of the box without context.
>
> **This paper repository is the intended starting point.** It exists precisely to distill, clarify, and reorganize the ideas from `truthspace_lcm` into a coherent, linear narrative. If you want to understand the theory, start here. The source codebase is best approached *after* reading the paper, and with the expectation that it is a research artifact rather than a finished product.

---

## Author

**Lesley Gushurst** — TruthSpace Geometric LCM Project

## License

GPLv3 — see [LICENSE](LICENSE) for details.
