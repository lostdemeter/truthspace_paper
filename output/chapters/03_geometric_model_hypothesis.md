# Chapter 3: The Geometric Model Hypothesis

*Weights are coordinates of a shape, not learned statistics.*

---

## 3.1 The Core Assertion

The **Geometric Model Hypothesis**[127] makes a radical claim about what neural networks actually are:

> **Weights are not learned parameters.** They are coordinates of a shape in high-dimensional space—a shape that training *discovers* rather than creates.

This reframes the entire training process. Instead of "learning a function that maps inputs to outputs," the model is "uncovering a pre-existing geometric structure that encodes the relationships in the data." The training process does not *build* this structure; it *finds* it.

![Weights as Shape Coordinates](../figures/fig3_1_shape_coordinates.png)

*Figure 3.1: Left: A representation of weights as φ-coordinates of a 3D shape. Red points (31%) are noise that can be zeroed without affecting accuracy. Right: Training fidelity as a function of training steps—the shape is discovered, not created.*

Evidence for this hypothesis comes from multiple directions:

1. **31% of weights are noise**[127, 198]: Up to 31% of weights in a trained transformer can be zeroed without measurable accuracy loss. If weights were learned parameters, this would not be possible—the optimization would have found a use for them.

2. **Weights form clusters at φ-levels**[127, 163]: When weights are projected onto φ-exponent space, they naturally cluster at discrete φ-levels. They are not continuously distributed but fall into well-defined geometric bins.

3. **The same weight structure appears across models**[180, 191]: The φ-structure found in Qwen2-7B also appears in DINOv2, CLIP, and other architectures. The geometric signature is **architecture-invariant**.

---

## 3.2 From Weights to Shape

The Geometric Model Hypothesis decomposes a neural network into four levels of geometric abstraction [154]:

### 3.2.1 Level 1: Weights = Lattice of Critical Lines

The weights of a transformer are not a collection of independent numbers. They form a **lattice of critical lines**[141]—hyperplanes in weight-space that divide the semantic space into regions. Each critical line is a decision boundary, and the lattice of all such boundaries defines the complete transformation.

The codebase's `discovery.py` implements this concretely. The `StructureDiscovery` class finds which context variables explain output variation, building a **gear train** of coarse and fine selectors:

```python
# From discovery.py: Geometrically, a weight is a coordinate on a selector gear
class TransformRule:
    def apply(self, value, context=None):
        if self.rule_type == 'identity':
            return value
        elif self.rule_type == 'consistent':
            return self.params['output']
        elif self.rule_type == 'selector':
            # Output depends on one context variable
            var = self.params['variable']
            ctx_val = context.get(var)
            return self.params['selector_map'].get(
                ctx_val, self.params.get('default_output', value))
```

This is the geometric view of a "learned transformation": a set of decision surfaces (selectors) that route inputs to outputs based on their position in φ-space.

### 3.2.2 Level 2: Gates = Encoding of Weight Geometry

Gates (SiLU, sigmoid, softmax) encode the geometric structure of the weight lattice. Each gate is a **φ-operation** that selects which subset of the lattice to activate based on the input's position.

The exact φ-form of sigmoid, verified in code (`phi_computer.py`):

```python
def phi_sigmoid(x: float) -> float:
    """sigmoid(x) = 1 / (1 + phi^(-x/ln(phi)))"""
    return 1 / (1 + PHI ** (-x / LN_PHI))
```

This is not an approximation—it is an algebraic identity. The standard sigmoid uses $e^{-x}$; the φ-sigmoid uses $\phi^{-x/\ln(\phi)}$. Since $\phi^{1/\ln(\phi)} = e$ (by definition of natural log), the two are identical. But the φ-form reveals the underlying geometry: **sigmoid selects between two φ-levels**.

### 3.2.3 Level 3: Topology = Spectral Decomposition of Gate Graph

The connectivity pattern of gates can be decomposed spectrally, revealing its intrinsic geometric structure. The eigenvalues follow a **φ-Zipf distribution**—the spectrum decays as a power law with a φ-based exponent [154].

### 3.2.4 Level 4: Spectrum = φ-Zipf Eigenvalues

The final irreducible level is the spectrum: the distribution of eigenvalues of the gate graph. This distribution follows:

$$\lambda_k \propto \phi^{-k}$$

where $\lambda_k$ is the k-th eigenvalue. This φ-Zipf distribution is the fingerprint of geometric computation—it appears in every transformer architecture examined.

---

## 3.3 The Search for the Irreducible Shape

If weights are shape coordinates, what is the shape itself? This question drove a systematic search that culminated in the **irreducible shape**[141]:

> The irreducible shape of transformer computation is a lattice of **3,584 critical lines** dividing semantic space into **67,942,912 binary intersection points**—at 1 bit each, this is the information-theoretic minimum for token prediction.

The search for this shape progressed through several phases:

### 3.3.1 Phase 1: The Vacuum Forming Experiments (Docs 1-6)

Initial experiments established that LLM embeddings have an interior geometric structure. Phase-shift probing revealed zero-variance points and polarity encoding, suggesting a low-dimensional manifold underlying the high-dimensional embedding space.

### 3.3.2 Phase 2: The φ-Lattice (Docs 99-163)

The breakthrough came when attention shifted from building a TruthSpace-native system to reverse-engineering existing transformers (Qwen2-7B, DINOv2). The finding: **weights naturally occupy absolute positions on a φ-lattice**[99, 101, 128, 163].

The `phi_lattice_rules` (Doc 163) codified the discovered structure:

1. **Quantization rule**: Weights cluster at discrete φ-levels (not continuous)
2. **Vocabulary rule**: Only 89 unique (level, sign) pairs cover all weights
3. **Sign structure rule**: 16 equal-probability quaternion sign patterns
4. **Clustered deltas rule**: Within-level deltas cluster around ±φ^k
5. **Self-similarity rule**: The same φ-structure appears at every scale
6. **Translation invariance rule**: The φ-lattice is translation-invariant—shifting all coordinates leaves the geometry unchanged

### 3.3.3 Phase 3: The Tetromino Weight Hypothesis (Doc 162)

The discrete nature of φ-levels led to a surprising discovery: weights form a constrained geometric structure akin to **tetrominoes tiling space**. Just as Tetris pieces (tetrominoes) can tile a 2D plane with only 7 piece types, neural network weights can tile weight-space with only **74 unique φ-structures**.

This was verified in `unwound_transformer/tetromino_*.py`:

> Each weight is encoded as (sign, φ-level, residual). Across all 7B parameters of Qwen2-7B, only 74 unique (level, sign) pairs appear with significant frequency. This means the entire model can be described by a vocabulary of 74 geometric primitives.

The implications are profound: a 7-billion-parameter model compresses to a 74-entry lookup table for its fundamental structure, plus residual corrections.

### 3.3.4 Phase 4: Computation IS Geometry (Doc 154)

The hypothesis that computation IS geometry was proven through a **census** of all component types in a transformer:

| Component | Geometric Interpretation | φ-Form |
|-----------|------------------------|--------|
| Embeddings | Position on φ-lattice | sign × φ^level |
| Q/K/V Matrices | Rotation operators | φ-exponent arithmetic |
| Attention | Spatial routing | φ-softmax routing |
| MLP Up/Gate/Down | φ-level selectors | φ-sigmoid gating |
| RMS Norm | φ-level alignment | shift to φ^0 scale |
| LM Head | Navigation map | φ-distance to tokens |

Each component's standard operation was replaced with an exact φ-equivalent, and the results were verified to match the original transformer output with 99.9991% correlation [129].

---

## 3.4 The Fail-Fast Philosophy

A key insight from the TruthSpace project that makes the Geometric Model Hypothesis testable is the **fail-fast philosophy**[Project Overview]:

> No graceful fallbacks. If geometric classification fails, we see the error rather than hiding it with pattern matching.

This philosophy enforces a critical constraint: every component must work **geometrically** or fail visibly. The `phi_geometric` API embodies this:

```python
# From phi_geometric/__init__.py:
# No torch. No GPU. No neural networks.
# Pure geometry: discover, navigate, verify.

pd = PhaseDiscovery()
pd.add_pair(list('ship'), list('ʃɪp'))
pd.add_pair(list('cat'),  list('kæt'))

result = pd.discover()
nav = result.to_navigator()

trace = nav.execute(list('shop'))
print(trace.output_elements)  # ['ʃ', 'ɒ', 'p']
```

The PhaseDiscovery engine finds geometric structure in transformation data without any neural network components. It uses:

- **Information gain** to detect which context variables explain inconsistencies
- **φ-level binning** to represent multi-distance context with few features
- **Entropy reduction** to identify the minimal gear train (coarse + fine selectors)

This engine was validated on **8 archetypes** of transformations (`examples/archetypes.py`), covering every combination of collapse, expand, context-dependent, and pure-map phases. All 8 archetypes achieve **100% accuracy** on training data when the correct context window is set.

---

## 3.5 The Geometric Model as an Experimental Program

The Geometric Model Hypothesis is not just a philosophical standpoint—it is an experimental program that makes falsifiable predictions:

1. **If weights are shape coordinates**, then replacing weight storage with φ-lattice lookups should preserve model behavior. This was confirmed in Doc 187: "Transformer as a Lookup Table"—a 7B parameter transformer replaced with a 1.09 GB lookup table achieves 100% accuracy.

2. **If computation is φ-navigation**, then the φ-form of sigmoid/softmax/SiLU should exactly match the standard forms. This was confirmed in Doc 191: the φ-computer proof shows 100% token accuracy.

3. **If the irreducible shape is finite**, then there is a minimum size below which no further compression is possible. This was confirmed in Doc 141: 67.9M binary intersection points, 3,584 critical lines.

4. **If training discovers rather than creates**, then different random initializations should converge to similar φ-lattice coordinates. This is the subject of ongoing investigation (Doc 194).

---

## 3.6 Summary

The Geometric Model Hypothesis transforms our understanding of neural networks:

| Traditional View | Geometric View |
|-----------------|---------------|
| Weights are learned parameters | Weights are φ-coordinates of a shape |
| Training creates the model | Training discovers the φ-lattice |
| Computation is matrix operations | Computation is φ-navigation |
| Knowledge is stored in weights | Knowledge IS the φ-shape |
| Models are statistical learners | Models are geometric transcoders |

This hypothesis sets the stage for everything that follows. In the next chapter, we examine how information is encoded in φ-space—the φ-dial and its dimensional hierarchy—and in Chapter 5 we explore the master symmetry that governs all φ-transformations: ENCODE = DECODE.

---

*Sources: Docs 022, 039, 127, 141, 154, 162, 163, 191*
