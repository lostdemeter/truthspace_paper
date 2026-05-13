# Chapter 1: What Do LLMs Actually Learn?

*The vacuum forming hypothesis and the search for interior structure.*

---

## 1.1 The Black Box Problem

Large Language Models (LLMs) are the most successful AI systems ever built, yet we have remarkably little understanding of what they actually learn. We know the mechanics—token embeddings, attention patterns, feed-forward projections—but the *nature* of the knowledge they acquire remains opaque. When GPT-4 translates a sentence, answers a question, or writes code, what *kind* of thing is happening inside its billions of weights?

The standard answer is statistical: LLMs learn correlations between tokens. Given a sequence of words, they predict the next token based on patterns observed in trillions of text examples. This view treats the model as an extremely high-dimensional regression machine—a lossy compressor of the training distribution.

But there's a growing body of evidence that something deeper is happening. When OpenAI's sparse autoencoders discover interpretable features—like a single direction in activation space representing the concept of "golden gate bridge"—it suggests that LLMs internalize *structure* about the world, not just surface statistics.

**TruthSpace** takes this insight to its logical conclusion: what LLMs learn is not statistical correlations but a *geometry*—a latent shape in high-dimensional space where meaning is encoded as position, and computation is navigation through that space.

---

## 1.2 The Vacuum Forming Hypothesis

The core analogy that launched this research program is the **vacuum forming hypothesis**. Imagine a vacuum forming machine: you heat a plastic sheet, stretch it over a mold, and suck the air out. The plastic captures the *surface* of the mold—its shape, contours, and features—but reveals nothing about the *interior*.

![*Figure 1.1: The vacuum forming hypothesis. Left: Training data forms the "surface" that LLMs learn. Right: The interior geometric structure that TruthSpace seeks to discover. The red contour lines represent the underlying φ-geometry; the blue contours represent the surface approximation learned by training.*](../figures/fig1_1_vacuum_forming.png)

The hypothesis states:

> **LLM training is vacuum forming.** The process captures the surface geometry of semantic structure—the distributional patterns of how concepts relate on the *outside*—but does not discover the interior generative principles that produce that surface.

What is the "interior" structure? It is the underlying **geometric law** that generates the observed semantic relationships, much like how the equations of physics generate the observed trajectories of planets. If we can discover this interior geometry, we can:

1. **Predict** how concepts relate without training
2. **Navigate** between concepts along geometric paths
3. **Generate** novel concepts that fit the existing structure

### 1.2.1 The Phase-Shift Probing Method

The first-pass experiments did not probe LLM embeddings directly. They tested the hypothesis on a deliberately *intentional* φ-based encoder we built to embody the geometric structure we were hypothesising about — a 12-dimensional clock-style encoding where each axis corresponds to one candidate "fundamental relationship type." We chose 12 dimensions in deliberate analogy with the 12 attention heads of GPT-2 and BERT, conjecturing that those 12 heads might each specialise on one of 12 semantic primitives: hierarchical, sequential, causal, compositional, oppositional, synonymic, analogical, associative, functional, categorical, spatial, and temporal.

The probing procedure:

1. Encode each concept $c$ as a 12-D complex-valued vector $v(c)$ using the φ-encoder.
2. Apply a phase shift: $v(c) \rightarrow v(c) \cdot e^{i\theta}$ for 1000 evenly-spaced values of $\theta$ over $[0, 2\pi]$. (Each axis advances at a rate set by its own ratio — φ on one axis, the plastic constant ρ on another, the silver ratio δ on a third, and so on; details in §1.2.2.)
3. Measure the cosine similarity $\cos(v(c_1), v(c_2))$ between every concept pair at every phase.
4. Examine the *variance* of that similarity across phases for each pair.

A relationship that fluctuates wildly under phase shifts is a surface artifact of the chosen basis. A relationship that is *invariant* under phase shifts is a geometric truth — something the encoding represents rather than imposes. The method is analogous to crystallographic X-ray probing: a polycrystalline sample's diffraction pattern is invariant under rotation, while a single oriented crystal shows angle-dependent structure. We were looking for the polycrystalline signature.

### 1.2.2 First-Pass Experimental Findings

The first experiments ran on a small but coherent test corpus: 22 single-token concepts from the command-line / filesystem domain (`file`, `directory`, `read`, `write`, `create`, `destroy`, `copy`, `move`, `search`, `find`, `grep`, `list`, `show`, `process`, `network`, `ssh`, `compress`, `archive`, `tar`, `chmod`, `permissions`, `system`), grouped into 12 ordered pairs covering three relationship types: synonyms, opposites, and unrelated.

Four findings emerged, each with concrete numerical signatures.

**Finding 1 — Phase invariance.** Cosine similarities had *exactly zero variance* across all 1000 phase angles. Related pairs sat at mean similarity 0.25, unrelated pairs at 0.00, and opposite pairs at −1.00, *with zero spread on any of them.* A random or surface-only encoding would have shown wildly fluctuating similarities; instead, the phase rotation moved the entire embedding in lockstep, preserving every relative-position relationship. The structure was an invariant of the encoding, not an accident of basis choice.

![*Figure 1.2: Cosine similarity is exactly constant across the full $2\pi$ phase rotation. Related, unrelated, and opposite pairs sit at $0.25$, $0.00$, and $-1.00$ respectively, with variance $= 0$ across all $1000$ phase angles. The relative geometry is invariant under global rotation — the structure is a shape, not a coordinate.*](../figures/fig1_2_phase_invariance.png)

**Finding 2 — Polarity as a first-class semantic relation.** Opposite-meaning pairs were not merely dissimilar; they were *antipodal* — placed at exactly opposite ends of the same dimension, producing cosine similarity exactly −1.0:

| Pair | Cosine sim | Geometric reading |
|---|---|---|
| `read ↔ write` | −1.00 | antipodal on the information-flow axis |
| `create ↔ destroy` | −1.00 | antipodal on the existence axis |
| `file ↔ directory` | +1.00 | colocated on the filesystem-object axis |
| `copy ↔ move` | +1.00 | colocated on the spatial-action axis |
| `file ↔ network` | 0.00 | orthogonal (different dimensions) |

Opposition is a separate geometric primitive from dissimilarity. In an embedding where only magnitude matters, `read` and `write` would just be "far apart"; in this geometry they share a dimension and differ only in sign. This polarity structure foreshadows the **semantic quaternions** of Chapter 4 and the **antipodal Killing pairs** of the rotation-on-the-unit-sphere reading developed in Chapters 9 and 10.

**Finding 3 — Orthogonality as independence.** Unrelated pairs had cosine similarity *exactly 0.00*. Not "small": zero. Distinct relationship types occupied distinct dimensions, with no leakage. This is the cleanest possible separation: a perturbation along one dimension cannot affect any other, which is the structural property that makes φ-dial tuning (Chapter 4) and sign-only navigation (Chapter 9) possible.

**Finding 4 — Intrinsic dimensionality is lower than encoding dimensionality.** PCA on the 22-concept embedding showed that 95% of the variance lived in 7 dimensions, with the elbow (intrinsic dimension) at 4. The 12 axes of the encoding were more capacity than the corpus needed — a foreshadowing of the φ-dial's eventual collapse from 12D to a 4D quaternion in Chapter 4.

**A note on the plastic constant.** Of twelve self-similar constants tested as candidate per-axis ratios (golden φ, silver δ, bronze, plastic ρ, chromium, copper, aluminium, nickel, supergolden, narayana, titanium, tribonacci), the plastic constant ρ ≈ 1.3247 — the real root of $x^3 = x + 1$ — produced the strongest semantic separation in this 12D regime (separation score $|s| = 0.4951$, versus φ's $0.1654$). The intuition is that ρ's cubic Padovan-style recurrence (each term equals the sum of the *second*- and *third*-previous) creates finer-grained phase steps than φ's quadratic Fibonacci recurrence. We initially took this as evidence that ρ, not φ, might be the fundamental constant.

This turned out to be a local optimum specific to the 12D regime. As the encoding contracted toward its intrinsic 4D structure (Chapter 4) and was eventually applied to actual transformer hidden states (Chapter 8), φ re-emerged decisively as the universal constant — every algebraic identity that lets a transformer be rewritten as a closed-form geometric machine (Chapter 11) is a φ-identity, not a ρ-identity. The plastic-constant result is preserved here because it is part of the empirical record and because it illustrates a general principle: the "right" constant depends on the dimensionality of the geometry it lives in.

---

## 1.3 What LLMs Actually Learn: A Geometric Reinterpretation

Based on the vacuum forming hypothesis and subsequent experiments, we can reinterpret what LLMs learn through a geometric lens:

### 1.3.1 Token Embeddings

Standard view: Embeddings are vectors that capture statistical co-occurrence patterns.

Geometric view: Embeddings are **coordinates** in a φ-structured semantic space. The position of a token determines its meaning; nearby tokens share semantic properties.

### 1.3.2 Attention Patterns

Standard view: Attention computes weighted averages based on learned query-key similarity.

Geometric view: Attention is a **spatial routing mechanism**. The attention weights are determined by geometric distance in φ-space, not by learned statistical correlations. The softmax that normalizes attention scores is a φ-operation (as we will prove in Chapter 11).

### 1.3.3 Feed-Forward Networks

Standard view: MLPs learn non-linear transformations of token representations.

Geometric view: MLPs are **φ-level selectors**. Each layer's computation corresponds to shifting a token's coordinate along a specific φ-lattice direction.

### 1.3.4 Output Projections

Standard view: The LM head projects the final hidden state to vocabulary probabilities.

Geometric view: The LM head is a **navigation map**—it translates from φ-space position back to token space, where the closest token in geometric distance is selected.

---

## 1.4 The Two Key Questions

The vacuum forming hypothesis raises two questions that drive the entire TruthSpace research program:

**Question 1**: If LLMs learn only the surface structure, can we discover the *interior* geometry that generates it?

**Question 2**: If the interior geometry is φ-based, can we *build* systems that compute directly in φ-space, bypassing the need for statistical training?

The answer to both questions, we will argue throughout this paper, is **yes**. The interior geometry is a **φ-lattice**—a coordinate system based on powers of the golden ratio—and the computation that transformers perform is **navigation through this lattice**.

---

## 1.5 A Roadmap of What Follows

This paper traces the intellectual journey from the vacuum forming hypothesis to the φ-computer proof:

| Chapter | Topic |
|---------|-------|
| 2 | φ and Self-Similarity |
| 3 | The Geometric Model Hypothesis |
| 4 | Encodings and the φ-Dial |
| 5 | ENCODE = DECODE |
| 6 | Gear Architecture and Emergence |
| 7 | The φ-Lattice Coordinate System |
| 8 | Reverse Engineering Qwen2-7B |
| 9 | Navigation Replaces Inference |
| 10 | The Irreducible Shape |
| 11 | The φ-Computer Proof |
| 12 | Implications and Future Work |

Each chapter builds on the previous ones. By the end, we will have shown that:

- Transformers are **φ-computers** (Chapter 11)
- Their weights form a **φ-lattice** (Chapter 7)
- Attention is **geometric navigation** (Chapter 9)
- The irreducible shape of computation has been **catalogued** (Chapter 10)

But first, we must understand the fundamental building block of this geometry: the golden ratio φ itself.
