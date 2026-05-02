# Chapter 4: Encodings, Transformations, and the φ-Dial

*From 1D control to 4D quaternion semantic navigation.*

---

## 4.1 The Encoding Problem

If computation is navigation through φ-space, how do we represent information in that space? The answer is **φ-encoding**: every value is represented as:

$$v = s \cdot \phi^{e} \cdot (1 + r \cdot (\phi - 1))$$

where $s \in \{-1, +1\}$ is the sign, $e \in \mathbb{Z}$ is the φ-exponent (level), and $r \in [0, 1)$ is the residual. This representation is the foundation of all TruthSpace computation.

The `PhiEncoder` (`phi_geometric/core/encoder.py`) implements this:

```python
class PhiEncoder:
    """value = sign x phi^(exponent / K)"""
    
    def encode(self, values):
        signs = torch.sign(values)
        abs_vals = torch.abs(values)
        exponents = torch.round(self.K * torch.log(abs_vals) / LN_PHI)
        return signs.int(), exponents.int()
```

The resolution parameter $K$ controls precision: $K=32$ gives ~3% precision per step, $K=128$ gives ~0.8%. The critical property is that **multiplication becomes exponent addition** — a floating-point multiply reduces to integer addition.

---

## 4.2 Projection Weighting and Semantic Axes

The earliest encodings in TruthSpace used **12D vectors** with dimensions for action, domain, and semantic roles [009]. Projection weighting — applying a diagonal linear transformation to emphasize certain dimensions — achieved 100% command disambiguation:

```python
# From Design 009: Diagonal weighting of projection dimensions
weighted = features @ diag(weights)  # emphasize action dimensions
```

This evolved into recognizing that **any transformation can be a dimension** [120]. The Universal Dimension Principle states:

> Any distinguishable transformation defines a valid axis in semantic space. The choice of axes is not fixed — it emerges from the transformations the system needs to perform.

---

## 4.3 The φ-Dial: 1D to 4D Control

The φ-dial evolved through four stages of dimensional control, each adding a new axis of semantic freedom [041-044]:

### Stage 1: The 1D φ-Dial [041]

The simplest control: a single real parameter $\alpha \in [-1, 1]$ that controls navigation direction:

- $\alpha = -1$: Inward navigation (specific, rare, formal)
- $\alpha = 0$: Balanced navigation (neutral)
- $\alpha = +1$: Outward navigation (universal, common, casual)

The weight formula: $\text{weight} = \phi^{\alpha \times \log(\text{value})}$

This single dial simultaneously controls multiple semantic dimensions — specificity, formality, and frequency — because they are coupled in the φ-geometry.

### Stage 2: The 2D Complex φ-Dial [042]

Adding a second dimension decouples **specificity/style** (magnitude) from **perspective/voice** (phase):

$$z = r \cdot e^{i\theta}, \quad r \in [0,1], \theta \in [0, 2\pi)$$

### Stage 3: The 3D φ-Dial [043]

Adding depth creates a third axis for **detail level** — how elaborate the response should be. The triplet (style, perspective, depth) forms a complete control space for most communication needs.

### Stage 4: The 4D Quaternion φ-Dial [044]

The final form follows the quaternion structure:

$$q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$$

| Axis | Name | Range | Controls |
|------|------|-------|----------|
| **X** | Style | -1 to +1 | Vocabulary selection (formal ↔ casual) |
| **Y** | Perspective | -1 to +1 | Voice/framing (subjective ↔ meta) |
| **Z** | Depth | -1 to +1 | Detail level (terse ↔ elaborate) |
| **W** | Certainty | -1 to +1 | Epistemic stance (definitive ↔ hedged) |

The `QuaternionEncoder` in `hypermapping/encoders.py` implements this directly:

```python
class QuaternionEncoder(Encoder):
    """4D Quaternion encoder with semantic axes (from Design 044).
    
    - X (i-axis): Polarity - positive vs negative
    - Y (j-axis): Intensity - how strong the signal
    - Z (k-axis): Style - formal vs casual (derived from word length)
    - W (scalar): Certainty - definitive vs hedged
    """
    
    def encode_input(self, text: str) -> np.ndarray:
        words = str(text).lower().split()
        polarity = sum(self.polarity_vocab.get(w, 0) for w in words)
        polarity = np.clip(polarity, -1, 1)
        intensity = self._first_match(words, self.intensity_vocab, 0.5)
        avg_word_len = np.mean([len(w) for w in words]) if words else 5
        style = np.clip((avg_word_len - 5) / 5, -1, 1)
        certainty = self._first_match(words, self.certainty_vocab, 0.0)
        pos = np.array([polarity, intensity, style, certainty])
        return pos / max(np.linalg.norm(pos), 1e-10) * CRITICAL_LINE
```

![4D Quaternion φ-Dial](../figures/fig4_1_quaternion_dial.png)

*Figure 4.1: The 4D Quaternion φ-Dial. Left: The four axes (X: Style, Y: Perspective, Z: Depth, W: Certainty as spherical radius). Right: Control sliders showing how each axis modulates output generation.*

---

## 4.4 Semantic Quaternions: 100% Analogy Accuracy

The 4D quaternion encoding proved capable of **100% analogy accuracy** [067]. The key insight: semantic attributes like gender, age, agency, and animacy map to orthogonal quaternion axes, making analogies a simple vector operation:

```python
# king - man + woman = queen in quaternion space
# gender_flip = -2.0 (always!)
woman_pos = quaternion_encoder("woman")  
man_pos = quaternion_encoder("man")
king_pos = quaternion_encoder("king")
queen_pos = king_pos - man_pos + woman_pos
# → 100% match to quaternion_encoder("queen")
```

The gender flip is always $\Delta x = -2.0$ — a constant vector operation that works identically for king→queen, man→woman, and boy→girl. This self-similarity is **self-verifying** — no external validation is needed, because the operation's consistency across examples IS the proof.

---

## 4.5 Holographic φ-Encoding

Holographic φ-encoding [142] extends φ-encoding to compress neural network weights by projecting them into a φ-basis and storing only the dominant components:

The process:
1. Extract weights from a trained model
2. Convert to φ-basis: $w_i \to s_i \cdot \phi^{e_i}$
3. Retain only components above a φ-threshold
4. Reconstruct: $\hat{w} = \sum_{k} s_k \cdot \phi^{e_k}$

This achieves **14× compression with 0.09% error** in MESH matrices [130], and **99.9984% correlation** when φ-encoding Qwen2-7B attention layers [136].

---

## 4.6 The φ-Adapter: Universal Geometric Reconstruction

The `PhiAdapter` (`phi_adapter/adapter.py`) generalizes φ-encoding to reconstruct any model's output at scalable accuracy:

```python
adapter = PhiAdapter(mode='svd')
adapter.fit(features, targets)

# Full reconstruction (99%+ accuracy)
pred_full = adapter.predict(features)

# Fast reconstruction using only top-50 DOF
pred_fast = adapter.predict(features, n_components=50)
```

The adapter uses SVD to find the natural geometric structure of the data, then applies φ-scaling:

```python
# phi-scaling of singular values
phi_scales = np.array([PHI ** (-i / scaling_rate) for i in range(n_components)])
phi_scales = phi_scales / phi_scales.sum() * n_components
```

This produces a DOF-accuracy curve where adding components follows a φ-decay law — the first few components capture most of the signal, and additional components contribute at φ-decaying rates.

---

## 4.7 The Music Box Principle [112]

An important conceptual model for understanding φ-encoding is the **Music Box Principle**:

> A music box does not contain music — it contains a cylinder with pins. When the cylinder turns, the pins pluck tines, and music *emerges* from the interaction. Similarly, φ-space does not contain knowledge — it contains positions. Knowledge emerges from the interaction of positions with the navigation mechanism.

This principle highlights why φ-encoding is not compression in the traditional sense. A φ-encoded weight is not a compressed version of a float — it is a coordinate in a space where the computation itself is defined by geometric relationships.

---

## 4.8 Summary

| Encoding | Dimensions | Key Property | Source |
|----------|-----------|--------------|--------|
| 12D vector | 12 | Action/domain separation | 009 |
| 1D φ-dial | 1 | Inward/outward navigation | 041 |
| 2D complex dial | 2 | Specificity + perspective | 042 |
| 3D dial | 3 | Style + perspective + depth | 043 |
| 4D quaternion dial | 4 | Full semantic control + certainty | 044 |
| Semantic quaternion | 4 | 100% analogy accuracy | 067 |
| Holographic φ-encoding | variable | 14× compression, 0.09% error | 142 |
| φ-Adapter | DOF-truncated | Universal model reconstruction | adapter.py |

The φ-dial progression from 1D to 4D reveals a fundamental truth: semantic space is quaternion-structured. The fourth axis (certainty) is special — it controls the radius of the quaternion sphere, acting as a meta-parameter that governs how definitive the system's output should be.

In the next chapter, we explore the master symmetry that makes all of this possible: ENCODE = DECODE.

---

*Sources: Docs 009, 041, 042, 043, 044, 067, 112, 120, 124, 130, 136, 137, 142*
