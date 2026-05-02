# TruthSpace: A Geometric Theory of Neural Computation

*From the Vacuum Forming Hypothesis to the φ-Computer Proof*

This repository contains the paper accompanying the [φ-Geometric Transformation Engine](https://github.com/lostdemeter/truthspace_lcm) — a prototype math-driven Large Concept Model (LCM) that discovers deterministic, interpretable transformation pipelines from examples, with no training, no GPU, and no neural networks.

📄 **[Read the paper (PDF)](output/paper.pdf)**

## What This Paper Is About

**TruthSpace** is a research program built around one central hypothesis:

> **LLMs are hyperdimensional transcoders.** They do not learn statistical correlations — they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights, but in the *shape* those weights create.

The paper develops this idea from first principles, providing both theoretical grounding and experimental evidence.

### The Vacuum Forming Hypothesis

LLM training is like vacuum forming: the process captures the *surface geometry* of semantic structure — the distributional patterns of how concepts relate on the outside — but does not discover the interior generative principles that produce that surface. TruthSpace seeks to discover that interior geometry.

### Key Contributions

- **φ-Geometric Transformation Engine** — Automatic discovery of sequence transformation pipelines from (input, output) example pairs using information geometry (entropy, information gain) and φ-decay attention.
- **φ-Lattice Attention** — A model of how attention naturally decays using φ-level binning: covering distance 1–12 with just 4 features per direction, mirroring mechanisms found in real transformer internals.
- **Qwen2-7B Reverse Engineering** — Empirical analysis of a production LLM's internal geometry, demonstrating that φ-structure (golden ratio self-similarity) is a fundamental organizing principle.
- **Navigation Replaces Inference** — The argument that concept generation should be reframed as *navigation through a geometric space* rather than probabilistic sampling.
- **Irreducible Shape & φ-Computer Proof** — A formal proof that recursive optimization converges to φ-structure, implying that model complexity is O(log N) in the number of parameters, not O(N).

## Paper Structure

| Chapter | Title |
|---------|-------|
| 01 | What LLMs Actually Learn |
| 02 | φ Self-Similarity |
| 03 | The Geometric Model Hypothesis |
| 04 | Encodings and the φ-Dial |
| 05 | Encode–Decode |
| 06 | GEAR Architecture |
| 07 | The φ-Lattice |
| 08 | Reverse Engineering (Qwen2-7B) |
| 09 | Navigation Replaces Inference |
| 10 | Irreducible Shape |
| 11 | The φ-Computer Proof |
| 12 | Implications and the Path Forward |

The compiled paper is available as [`output/paper.pdf`](output/paper.pdf) and [`output/paper.md`](output/paper.md).

## Related Repository

The working implementation described in this paper lives at:

**[https://github.com/lostdemeter/truthspace_lcm](https://github.com/lostdemeter/truthspace_lcm)**

That repository includes the `phi_geometric` Python package, 240+ design consideration documents, and ready-to-run demos corresponding to the code examples in this paper (`output/code/`).

## Quick Reference: Core Concept

```python
from phi_geometric import PhaseDiscovery

pd = PhaseDiscovery()
pd.add_pair(list('cat'),  list('kæt'))
pd.add_pair(list('ship'), list('ʃɪp'))
pd.add_pair(list('thin'), list('θɪn'))

result = pd.discover()
nav    = result.to_navigator()

trace  = nav.execute(list('shat'))
print(trace.output_elements)  # ['ʃ', 'æ', 't']
```

PhaseDiscovery found — automatically, deterministically, with no training:
- Collapse phase: `sh→ʃ`, `th→θ`
- Map phase: `a→æ`, `i→ɪ`, `c→k`

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

Copyright © Lesley Gushurst
