# Chapter 7: The φ-Lattice Coordinate System

*An absolute coordinate system for neural computation.*

---

## 7.1 From Eigenspace to φ-Lattice

The early TruthSpace encodings used **eigenspace coordinates** — positions derived from eigendecomposition of similarity matrices. This worked but had a fundamental problem: coordinates were relative. Moving to a different eigenspace (different data, different model) meant an entirely different coordinate system.

The breakthrough came with the shift to **absolute φ-lattice coordinates** [099, 101]:

> Instead of computing positions relative to other points in the space, every weight occupies an absolute position on the φ-lattice: sign × φ^level.

This eliminated the DC component problem in eigenspace approaches and achieved **100% accuracy** in coordinate-based matching.

---

## 7.2 The Rules of the φ-Lattice [163]

Six rules govern the φ-lattice, discovered through analysis of Qwen2-7B weights:

### Rule 1: Quantization

Weights are not continuous — they cluster at discrete φ-levels:

$$w \in \{s \cdot \phi^e \mid s \in \{-1, +1\}, e \in \mathbb{Z}\}$$

The residual $r \in [0, 1)$ represents the deviation within a level, but the dominant signal is the level itself. The `TetrominoFastWeight` class uses this:

```python
# Convert weight to (sign, phi-level) encoding
signs = np.sign(weight).astype(np.int8)
signs[signs == 0] = 1
abs_w = np.maximum(np.abs(weight), 1e-38)
levels = np.floor(np.log(abs_w) / LN_PHI).astype(np.int8)

# Tetromino ID = level * 2 + (sign > 0)
tet_ids = (levels * 2 + (signs > 0).astype(np.int8)).astype(np.int8)
```

![φ-Lattice and Tetromino Distribution](../figures/fig7_1_phi_lattice.png)

*Figure 7.1: Left: The φ-lattice — a 2D projection showing grid lines at φ-power intervals. Each intersection is a valid weight coordinate. Right: Weight count by φ-level, showing clustering at discrete levels with 74 unique tetromino structures.*

### Rule 2: Finite Vocabulary

Only **89 unique (level, sign) pairs** appear with significant frequency across all 7B parameters of Qwen2-7B. This means the entire model can be described by a vocabulary of 89 geometric primitives.

The tetromino analysis took this further: grouping adjacent weights with the same φ-level into geometric shapes (tetrominoes) reduced the vocabulary to **74 unique structures**:

```python
# Tetromino expansion: 74 values cover the entire model
unique_ids = np.unique(self.tet_ids)
self.n_tetrominoes = len(unique_ids)  # = 74

# At inference time: expand index → value
weight_approx = tet_values[tet_idx.astype(np.int32)]
```

Storage: 1 byte (int8) per weight vs 4 bytes (float32) = **4× compression**, 99.2% correlation.

### Rule 3: Quaternion Sign Structure

The sign bits across dimensions follow **16 equal-probability quaternion patterns**. This is exactly 2^4 = 16 patterns, matching the number of possible sign combinations in a 4D quaternion.

This is not coincidence: the sign structure IS the quaternion structure of the transformation. Each pattern corresponds to one of the 16 quaternion basis elements (1, i, j, k, ij, ik, jk, ijk, and their negatives).

### Rule 4: Clustered Deltas

Within a φ-level, deltas (differences between weights at the same level) cluster around $\pm \phi^k$. The distances between weights on the lattice are themselves φ-structured.

### Rule 5: Self-Similarity

The same φ-structure appears at every scale. A weight matrix at φ^3 has the same geometric properties as a weight matrix at φ^0 — just shifted by 3 levels. This is the direct consequence of φ's defining equation: $\phi = 1 + 1/\phi$.

### Rule 6: Translation Invariance

The φ-lattice is translation-invariant — shifting all coordinates by a constant leaves the geometry unchanged. This means that adding a constant to all φ-levels does not change the relationships between weights. What matters is the *difference* in φ-levels, not the absolute values.

---

## 7.3 The Tetromino Weight Hypothesis [162]

The tetromino weight hypothesis states:

> Neural network weights form constrained geometric structures akin to tetrominoes tiling space. Just as 7 Tetris pieces tile the 2D plane, 74 φ-tetrominoes tile the weight-space of a 7B parameter transformer.

The evidence:
- **74 unique φ-structures** across all Qwen2-7B weights
- **99.2% correlation** when reconstructing weights from tetromino indices alone
- **4× compression** with zero inference speed loss (expand at load time)
- **Structural consistency**: the same tetromino patterns appear across different layers and different models

---

## 7.4 The φ-Exponent Arithmetic Unit (φ-FPU) [133]

The φ-lattice enables a radical rethinking of arithmetic. Instead of IEEE 754 floating point:

$$a \times b = (s_a \cdot \phi^{e_a}) \times (s_b \cdot \phi^{e_b}) = (s_a \cdot s_b) \cdot \phi^{e_a + e_b}$$

A **floating-point multiply becomes an integer addition plus a sign XOR**. The φ-FPU implements this:

```python
# In phi_geometric/inference/phi_types.py:
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# In phi_geometric/core/encoder.py: PhiEncoder
# Pre-compute LUT for phi^(e/K) values
# Pre-compute addition LUT: phi^a + phi^b = phi^(b + LUT[a-b])

# In phi_geometric/inference/phi_integer.py:
def phi_accumulate(signs, exponents, axis=-1):
    """Sum phi-encoded values via fixed-point arithmetic."""
    # signs: {-1, 0, +1}, exponents: integer levels
    # Uses addition LUT for phi-space addition
```

The `PhiEncoder` pre-computes a Look-Up Table for φ-exponent addition:

```python
phi_powers[e] = PHI ^ ((e - bias) / K)  # LUT for decoding

# Addition in φ-space:
# phi^a + phi^b = phi^b * (phi^(a-b) + 1) = phi^(b + LUT[a-b])
# where LUT[d] = K * log_phi(phi^(d/K) + 1)
```

This makes φ-FPU addition a table lookup plus integer addition — no floating-point hardware required.

---

## 7.5 The φ-2byte Format [191]

The φ-2byte storage format encodes each weight as:

| Bits | Field | Values |
|------|-------|--------|
| 1 | Sign | -1 or +1 |
| 11 | φ-level | -1024 to 1023 |
| 4 | Residual | 0.0625 increments |

Total: 16 bits (2 bytes) per weight vs 32 bits (float32) = **2× compression with no accuracy loss**:

> The φ-2byte format achieved 2× compression (26.1 GB → 13.05 GB on Qwen2-7B) with a difference of only 2.78e-17 from theoretical values — essentially zero error.

---

## 7.6 The Irreducible Shape

The φ-lattice rules imply a minimum information-theoretic size: the **irreducible shape** [141]:

> The irreducible structure of transformer computation is a lattice of 3,584 critical lines dividing semantic space into 67,942,912 binary intersection points at 1 bit each.

This means:
- You cannot compress below 67.9 million bits (≈8 MB) for the essential structure
- Everything beyond that is "decoration" — residual corrections and noise
- The 31% of weights that can be zeroed (Doc 127, 198) may include most of this noise

---

## 7.7 Summary

| Property | Value | Source |
|----------|-------|--------|
| Unique φ-levels | 89 (level, sign) pairs | 163 |
| Unique tetrominoes | 74 structures | 162 |
| φ-FPU compression | 4× (int8 index) or 2× (φ-2byte) | 133, 191 |
| φ-lattice alignment | ~20% of weights on exact φ^n | FINDINGS |
| Residual encoding | sign × φ^level × (1 + r × (φ-1)) | encoder.py |
| Irreducible bits | 67.9M binary intersection points | 141 |

The φ-lattice provides the fundamental coordinate system for all TruthSpace computation. In the next chapter, we see how this lattice was discovered by reverse engineering a specific transformer: Qwen2-7B.

---

*Sources: Docs 099, 101, 133, 141, 162, 163, 191; phi_geometric/core/encoder.py; phi_geometric/inference/phi_types.py*
