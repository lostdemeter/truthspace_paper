# Chapter 1: What Do LLMs Actually Learn?

*The vacuum forming hypothesis and the search for interior structure.*

---

## 1.1 The Black Box Problem

Large Language Models (LLMs) are the most successful AI systems ever built, yet we have remarkably little understanding of what they actually learn. We know the mechanics—token embeddings, attention patterns, feed-forward projections—but the *nature* of the knowledge they acquire remains opaque. When GPT-4 translates a sentence, answers a question, or writes code, what *kind* of thing is happening inside its billions of weights?

The standard answer is statistical: LLMs learn correlations between tokens. Given a sequence of words, they predict the next token based on patterns observed in trillions of text examples. This view treats the model as an extremely high-dimensional regression machine—a lossy compressor of the training distribution.

But there's a growing body of evidence that something deeper is happening. When OpenAI's sparse autoencoders [23] discover interpretable features—like a single direction in activation space representing the concept of "golden gate bridge"—it suggests that LLMs internalize *structure* about the world, not just surface statistics.

**TruthSpace** takes this insight to its logical conclusion: what LLMs learn is not statistical correlations but a *geometry*—a latent shape in high-dimensional space where meaning is encoded as position, and computation is navigation through that space.

---

## 1.2 The Vacuum Forming Hypothesis

The core analogy that launched this research program is the **vacuum forming hypothesis**[3]. Imagine a vacuum forming machine: you heat a plastic sheet, stretch it over a mold, and suck the air out. The plastic captures the *surface* of the mold—its shape, contours, and features—but reveals nothing about the *interior*.

![Vacuum Forming Hypothesis](../figures/fig1_1_vacuum_forming.png)

*Figure 1.1: The vacuum forming hypothesis. Left: Training data forms the "surface" that LLMs learn. Right: The interior geometric structure that TruthSpace seeks to discover. The red contour lines represent the underlying φ-geometry; the blue contours represent the surface approximation learned by training.*

The hypothesis states:

> **LLM training is vacuum forming.** The process captures the surface geometry of semantic structure—the distributional patterns of how concepts relate on the *outside*—but does not discover the interior generative principles that produce that surface.

What is the "interior" structure? It is the underlying **geometric law** that generates the observed semantic relationships, much like how the equations of physics generate the observed trajectories of planets. If we can discover this interior geometry, we can:

1. **Predict** how concepts relate without training
2. **Navigate** between concepts along geometric paths
3. **Generate** novel concepts that fit the existing structure

### 1.2.1 Experimental Evidence

The initial experiments [4, 5] tested this hypothesis by probing LLM embedding spaces with **phase shifts**—rotating the phase of token embeddings and observing whether semantic relationships remained invariant. The key finding:

> Semantic similarity between concepts remained **consistent across phase shifts**, even when individual embedding magnitudes changed dramatically. This suggests an underlying geometric invariance that transcends surface correlations.

Specifically, when embeddings were shifted along φ-based phase angles [1]:

- **Zero-variance points** emerged—positions in semantic space where phase had no effect on meaning, corresponding to "semantic singularities"
- **Polarity encoding** was discovered: concepts were encoded not by magnitude but by *direction* in a low-dimensional signature space
- **Orthogonal dimensions** enabled independent tuning, where collisions only mattered within a dimension, not across them

The plastic constant ρ ≈ 1.3247 (the real root of x³ = x + 1) was found to provide finer semantic discrimination than φ in certain early 12D encodings [6], but this turned out to be a local optimum rather than a fundamental constant.

### 1.2.2 The Phase-Shift Probing Method

The phase-shift probing method works as follows:

1. Take a trained LLM's token embeddings
2. Apply a phase transformation: $v \rightarrow v \cdot e^{i\theta\phi}$ where $\phi$ is the golden ratio
3. Measure how semantic relationships (cosine similarity, analogies) change
4. Identify invariants—relationships that persist across all phase angles

This is analogous to probing a physical material with X-rays: the surface absorbs certain frequencies, but the interference patterns reveal the crystalline interior structure.

---

## 1.3 What LLMs Actually Learn: A Geometric Reinterpretation

Based on the vacuum forming hypothesis and subsequent experiments [2, 5], we can reinterpret what LLMs learn through a geometric lens:

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

| Chapter | Topic | Key Source Documents |
|---------|-------|---------------------|
| 2 | φ and Self-Similarity | 010, 124, 133, 137 |
| 3 | The Geometric Model Hypothesis | 022, 039, 127 |
| 4 | Encodings and the φ-Dial | 009, 041–044, 067, 142 |
| 5 | ENCODE = DECODE | 061, 089–091 |
| 6 | Gear Architecture and Emergence | 033, 049, 086, 103 |
| 7 | The φ-Lattice Coordinate System | 099–102, 162–163 |
| 8 | Reverse Engineering Qwen2-7B | 129, 134, 185–187, 190 |
| 9 | Navigation Replaces Inference | 161, 165–167, 175–176 |
| 10 | The Irreducible Shape | 039, 141, 154, 159–160 |
| 11 | The φ-Computer Proof | 145, 191, 199–200 |
| 12 | Implications and Future Work | 140, 155, 180, 202 |

Each chapter builds on the previous ones. By the end, we will have shown that:

- Transformers are **φ-computers** (Chapter 11)
- Their weights form a **φ-lattice** (Chapter 7)
- Attention is **geometric navigation** (Chapter 9)
- The irreducible shape of computation has been **catalogued** (Chapter 10)

But first, we must understand the fundamental building block of this geometry: the golden ratio φ itself.

---

*Sources: Docs 1, 2, 3, 4, 5, 6, 23, 33, 127*
