# Chapter 7: The φ-Lattice Coordinate System

*An absolute coordinate system for neural computation.*

---

## 7.1 From Eigenspace to φ-Lattice

The early TruthSpace encodings used **eigenspace coordinates** — positions derived from eigendecomposition of similarity matrices. This worked but had a fundamental problem: coordinates were relative. Moving to a different eigenspace (different data, different model) meant an entirely different coordinate system.

The breakthrough came with the shift to **absolute φ-lattice coordinates**:

> Instead of computing positions relative to other points in the space, every weight occupies an absolute position on the φ-lattice: sign × φ^level.

This eliminated the DC component problem in eigenspace approaches and achieved **100% accuracy** in coordinate-based matching.

---

## 7.2 The Rules of the φ-Lattice

Six rules govern the φ-lattice, discovered through analysis of Qwen2-7B weights:

### Rule 1: Quantization

Weights are not continuous — they cluster at discrete φ-levels:

$$w \in \{s \cdot \phi^e \mid s \in \{-1, +1\}, e \in \mathbb{Z}\}$$

The residual $r \in [0, 1)$ represents the deviation within a level, but the dominant signal is the level itself. A reference `TetrominoFastWeight` encoding:

```python
# Convert weight to (sign, phi-level) encoding
signs = np.sign(weight).astype(np.int8)
signs[signs == 0] = 1
abs_w = np.maximum(np.abs(weight), 1e-38)
levels = np.floor(np.log(abs_w) / LN_PHI).astype(np.int8)

# Tetromino ID = level * 2 + (sign > 0)
tet_ids = (levels * 2 + (signs > 0).astype(np.int8)).astype(np.int8)
```

![*Figure 7.1: Left: The φ-lattice — a 2D projection showing grid lines at φ-power intervals. Each intersection is a valid weight coordinate. Right: Weight count by φ-level, showing clustering at discrete levels with 74 unique tetromino structures.*](../figures/fig7_1_phi_lattice.png)

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

### Rule 3: Sign-Space of 4D Blocks

Weights group into 4D blocks (one per `head_dim / 4` partition in Qwen2-7B). Each block has 4 real components, each with an independent sign. The sign space of such a block is

$$\mathbb{Z}_2^4 \;=\; \{(s_1, s_2, s_3, s_4) : s_i \in \{-1, +1\}\}$$

which is the elementary abelian 2-group of order $2^4 = 16$. Empirically, **all 16 patterns appear with uniform probability** — observed frequencies range from 6.24% to 6.26%, indistinguishable from the maximum-entropy expectation of 6.25%. No pattern is forbidden, no pattern dominates.

The block dimensionality of 4 matches the four real components of a quaternion $(w, x, y, z)$, so this is the *sign-space of a quaternion-shaped block* — but the group itself is $\mathbb{Z}_2^4$, **not** the quaternion group $Q_8 = \{\pm 1, \pm i, \pm j, \pm k\}$. They are easy to confuse because both arise in 4D contexts, but $Q_8$ has 8 elements and a non-abelian product, while $\mathbb{Z}_2^4$ has 16 elements and an abelian (componentwise) product. The empirical claim is sharper than any algebraic identification: in the *sign* axis of the φ-lattice, weights are *maximally entropic* — all the geometric information lives in the level axis (Rule 1), not in the signs.

### Rule 4: Clustered Deltas

Within a φ-level, deltas (differences between weights at the same level) cluster around $\pm \phi^k$. The distances between weights on the lattice are themselves φ-structured.

### Rule 5: Self-Similarity

The same φ-structure appears at every scale. A weight matrix at φ^3 has the same geometric properties as a weight matrix at φ^0 — just shifted by 3 levels. This is the direct consequence of φ's defining equation: $\phi = 1 + 1/\phi$.

### Rule 6: Translation Invariance

The φ-lattice is translation-invariant — shifting all coordinates by a constant leaves the geometry unchanged. This means that adding a constant to all φ-levels does not change the relationships between weights. What matters is the *difference* in φ-levels, not the absolute values.

---

## 7.3 The Tetromino Weight Hypothesis

The tetromino weight hypothesis states:

> Neural network weights form constrained geometric structures akin to tetrominoes tiling space. Just as 7 Tetris pieces tile the 2D plane, 74 φ-tetrominoes tile the weight-space of a 7B parameter transformer.

The evidence:
- **74 unique φ-structures** across all Qwen2-7B weights (71 cover 90% by count)
- **99.2% per-layer correlation** when reconstructing weights from tetromino indices alone
- **4× compression** with zero inference speed loss (expand at load time)
- **Structural consistency**: the same tetromino patterns appear across different layers and different models

The 99.2% per-layer correlation tells only half the story. The 28-layer transformer compounds these per-layer errors, and *tetromino-only* reconstruction (no residuals) yields **33% full-model token accuracy** — far short of the 100% needed for a functional model. The residual correction (§7.5) is what closes the gap from 33% to 100%, and that is the empirical reason the φ-2byte format keeps a 7-bit residual alongside the tetromino index.

---

## 7.4 The φ-Exponent Arithmetic Unit (φ-FPU)

The φ-lattice enables a radical rethinking of arithmetic. Instead of IEEE 754 floating point:

$$a \times b = (s_a \cdot \phi^{e_a}) \times (s_b \cdot \phi^{e_b}) = (s_a \cdot s_b) \cdot \phi^{e_a + e_b}$$

A **floating-point multiply becomes an integer addition plus a sign XOR**. A reference φ-FPU implementation:

```python
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

# PhiEncoder:
# Pre-compute LUT for phi^(e/K) values
# Pre-compute addition LUT: phi^a + phi^b = phi^(b + LUT[a-b])

def phi_accumulate(signs, exponents, axis=-1):
    """Sum phi-encoded values via fixed-point arithmetic."""
    # signs: {-1, 0, +1}, exponents: integer levels
    # Uses addition LUT for phi-space addition
```

The φ-encoder pre-computes a Look-Up Table for φ-exponent addition:

```python
phi_powers[e] = PHI ** ((e - bias) / K)  # LUT for decoding

# Addition in φ-space:
# phi^a + phi^b = phi^b * (phi^(a-b) + 1) = phi^(b + LUT[a-b])
# where LUT[d] = K * log_phi(phi^(d/K) + 1)
```

This makes φ-FPU addition a table lookup plus integer addition — no floating-point hardware required.

---

## 7.5 The φ-2byte Format

The tetromino index alone caps reconstruction at 99.2% per-layer correlation, which compounds across 28 layers to 33% token accuracy (§7.3). To recover the missing precision, the **φ-2byte format** adds a residual byte. The 16-bit layout (from the reference `PhiTensor2Byte.from_float` implementation):

| Byte | Bits | Field | Encoding |
|------|------|-------|----------|
| 0 | 8 | φ-level | `int8`, range $-128$ to $+127$ |
| 1 | 1 | Sign | `0` = positive, `1` = negative |
| 1 | 7 | Residual | `uint8`, 0–127 → fractional offset on $[0, \phi-1)$ |

The reconstruction formula:

$$w \;=\; \text{sign} \cdot \phi^{\text{level}} \cdot \left(1 + \frac{\text{residual}}{127} \cdot (\phi - 1)\right)$$

The level places the weight on the φ-lattice; the residual adjusts within a level by the interval $\phi - 1 \approx 0.618$ in 127 equal steps. The forward direction computes the level as $\lfloor \ln |w| / \ln \phi \rfloor$ and the residual as $(|w| / \phi^{\text{level}} - 1) / (\phi - 1)$ clipped to $[0, 1]$.

Empirical performance on Qwen2-7B (28 layers, all `q/k/v/o/gate/up/down` projections converted):

| Metric | Value |
|---|---|
| Storage | $26.1$ GB (`float32`) $\to$ $13.05$ GB (φ-2byte) |
| Compression ratio | $2.00\times$ |
| Roundtrip weight correlation | $0.9999993$ |
| Token-prediction accuracy | $100\%$ (byte-for-byte identical to original on the verification suite) |

The residual is what carries the compression story from "geometric curiosity" (§7.3 tetromino-only, 33% accuracy) to "drop-in `float32` replacement" — without it the per-layer error compounds catastrophically; with it the full 28-layer stack reproduces original outputs exactly while halving storage.

### 7.5.1 The Seed Insight: Negative Zero

The `(sign, level, residual)` encoding has one property that IEEE-754 floating point does not. The points $(+1, \text{level}, 0)$ and $(-1, \text{level}, 0)$ are *distinct* coordinates in φ-space, even as their reconstructed magnitudes coincide. IEEE-754 enforces `-0 == +0`; the φ-2byte format preserves the sign at zero magnitude.

This was originally an incidental feature of the encoding — a bookkeeping detail that fell out of representing the sign separately. But it was the *seed insight* for a much larger result: if the encoding distinguishes $+0$ from $-0$, perhaps the trained network does too.

Empirical test on Qwen2-7B (SiLU) and DDColor (GELU) confirmed it. The activation gate is not a binary switch — it is a **4-state holographic encoder**:

| State | Region | SiLU behaviour | Role |
|-------|--------|----------------|------|
| `+1` EXPAND | $x \geq +\log\phi$ | $\approx x$ | bright fringe, full fire |
| `+0` PRESERVE+ | $0 \leq x < +\log\phi$ | $\approx x/2$ | bright fringe, linear positive |
| `−0` PRESERVE− | $-\log\phi \leq x < 0$ | $\approx x/2$ | **dark fringe**, linear negative — *the negative zero* |
| `−1` CONTRACT | $x < -\log\phi$ | $\approx x \cdot e^x$ | dark fringe, deep leakage |

The boundaries at $\pm\log\phi \approx \pm 0.481$ are exact, not arbitrary: $\sigma(\log\phi) = 1/\phi$ identically (the same identity that grounds §4.5's holographic φ-encoding). At layer 14 of Qwen2-7B, the "dead" channels in the `−0` and `−1` states carry **42.4% of the output energy** via destructive interference with the live channels; removing them drops end-to-end argmax from 4/5 to 0/5. The sign at zero magnitude carries roughly *four times more information* than the magnitude itself.

The full mechanism — bright fringes, dark fringes, and how the gate field acts as a holographic plate — is developed in Chapter 9 as the **holographic gate field**, with the corresponding correction term in φ-SiLU appearing in Chapter 11 §11.4. Standalone, runnable demonstrations live in two external repositories:

- `lostdemeter/holographic_gate` — the 4-state gate, reproduced on Qwen2-7B and synthetic MLPs.
- `lostdemeter/geometric_ipa` — English → IPA phonetic transcription built only from the `gate_step(x, t, s)` primitive (sharpness $s = \phi^2$, exact `IdealGate` form of GELU). No neural network, no gradient descent. The system discovers context-dependent rules (`g` before `a` is hard, before `e` is soft) using information gain — the same "gear shift" discipline as Chapter 6.

The tetromino encoding's natural $\pm 0$ distinction was therefore not just a compression trick; it was the geometric scaffold that made these discoveries possible.

---

## 7.6 The Irreducible Shape

The φ-lattice rules imply a minimum information-theoretic size: the **irreducible shape**:

> The irreducible structure of transformer computation is a lattice of 3,584 critical lines dividing semantic space into 67,942,912 binary intersection points at 1 bit each.

This means:
- You cannot compress below 67.9 million bits (≈8 MB) for the essential structure
- Everything beyond that is "decoration" — residual corrections and noise
- The 31% of weights that can be zeroed (Chapter 3) may include most of this noise

---

## 7.7 Summary

| Property | Value |
|----------|-------|
| Unique φ-levels | 89 (level, sign) pairs |
| Unique tetrominoes | 74 structures |
| φ-FPU compression | 4× (int8 index) or 2× (φ-2byte) |
| φ-lattice alignment | ~20% of weights on exact φ^n |
| Residual encoding | sign × φ^level × (1 + r × (φ-1)) |
| Irreducible bits | 67.9M binary intersection points |

The φ-lattice provides the fundamental coordinate system for all TruthSpace computation. In the next chapter, we see how this lattice was discovered by reverse engineering a specific transformer: Qwen2-7B.
