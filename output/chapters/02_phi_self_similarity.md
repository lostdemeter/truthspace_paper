# Chapter 2: φ and Self-Similarity

*The golden ratio as the organizing principle of geometric computation.*

---

## 2.1 The Defining Equation

The golden ratio φ is the mathematical constant:

$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895$$

Its defining property is self-similarity:

$$\phi = 1 + \frac{1}{\phi}$$

This single equation encodes a profound truth: φ can be decomposed into a part that equals 1 and a part that equals 1/φ. The ratio between the whole and the larger part is the same as the ratio between the larger part and the smaller part. In other words: **φ is self-similar at every scale**.

![*Figure 2.1: Three views of φ self-similarity. Left: φ = 1 + 1/φ geometrically. Center: The Fibonacci spiral approximates φ through integer ratios. Right: φ^n follows a self-similar exponential scaling.*](../figures/fig2_1_phi_spiral.png)

This self-similarity is not a mathematical curiosity—it is the fundamental organizing principle that makes φ the natural coordinate system for geometric computation. Consider what self-similarity gives us:

1. **Scale invariance**: A transformation that works at φ^2 works identically at φ^0
2. **Recursive decomposition**: Any φ interval can be decomposed into smaller φ intervals
3. **Natural spacing**: φ^n provides logarithmic spacing that avoids collisions—a property critical for encoding distinct concepts without overlap

---

## 2.2 φ-Powers as a Coordinate System

The powers of φ form a discrete set with remarkable properties:

| n | φ^n | Notes |
|---|-----|-------|
| -4 | 0.146 | Fine-grained resolution |
| -3 | 0.236 | |
| -2 | 0.382 | |
| -1 | 0.618 | |
| 0 | 1.000 | The unit |
| 1 | 1.618 | φ itself |
| 2 | 2.618 | |
| 3 | 4.236 | |
| 4 | 6.854 | Coarse scale |

The key insight is that **any positive real number** can be represented as:

$$x = s \cdot \phi^{e} \cdot (1 + r \cdot (\phi - 1))$$

where $s \in \{-1, +1\}$ is the sign, $e \in \mathbb{Z}$ is the φ-exponent (level), and $r \in [0, 1)$ is the residual within the φ-level.

The two fundamental constants:

```python
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
```

And the φ-coordinate conversion:

```python
class PhiCoord:
    """A coordinate in φ-space: value = sign x phi^level x (1 + residual x (phi-1))"""
    level: int
    sign: int  # +1 or -1
    residual: float  # in [0, 1)

    def to_float(self) -> float:
        return self.sign * (PHI ** self.level) * (1 + self.residual * (PHI - 1))

    @classmethod
    def from_float(cls, x: float) -> 'PhiCoord':
        if abs(x) < 1e-15:
            return cls(level=-100, sign=1, residual=0.0)
        sign = 1 if x > 0 else -1
        abs_x = abs(x)
        log_phi_x = np.log(abs_x) / LN_PHI
        level = int(np.floor(log_phi_x))
        base = PHI ** level
        residual = (abs_x / base - 1) / (PHI - 1)
        residual = np.clip(residual, 0, 1 - 1e-10)
        return cls(level=level, sign=sign, residual=residual)
```

This encoding scheme means that a number is decomposed into its sign, its power-of-φ magnitude, and its fine position within that magnitude—similar to floating point but using φ as the base rather than 2.

---

## 2.3 φ as Universal Adapter

The most important property of φ for our purposes is its role as a **universal adapter**. The golden ratio can represent any linear structure due to five key properties:

1. **Self-similarity**: $\phi = 1 + 1/\phi$ means φ contains its own inverse
2. **Fibonacci connection**: φ is the limit of $F_{n+1}/F_n$ as $n \to \infty$, connecting discrete and continuous
3. **Optimal packing**: φ^k provides maximal spacing between consecutive powers, minimizing collisions
4. **Logarithmic representation**: $\log_\phi(x)$ maps any positive number to a linear scale
5. **Discrete-continuous bridge**: Binet's formula $F_n = (\phi^n - (-\phi)^{-n})/\sqrt{5}$ ties the integer Fibonacci sequence to the continuous family $\phi^n$. Any Fibonacci computation has an equivalent φ-power computation and vice versa — discrete integer arithmetic and continuous exponential growth are the same operation in different gauges. This is the property that makes the addition LUT of §2.6 well-defined.

Property 1 is the most consequential. Because $\phi \cdot 1/\phi = 1$, we have:

> **Encoding** (multiply by φ) and **decoding** (multiply by 1/φ) are the same operation in opposite directions.

This means that if you encode a value by multiplying by φ, you can decode it by multiplying by 1/φ—and both operations have the same structure. This duality will become foundational in Chapter 5 (ENCODE = DECODE).

---

## 2.4 φ-Level Binning and Geometric Context

φ-level binning is used to encode context at multiple distances using a fixed number of features:

![*Figure 2.2: Left: φ-decay of context weights with distance, showing how levels 0-3 partition 12 tokens of context using only 4 features per direction. Right: The infinite self-similarity of φ visualized as a recursive decomposition tree.*](../figures/fig2_2_self_similarity.png)

The levels are defined as:

| Level | Distance Range | φ-Weight | Tokens Covered |
|-------|---------------|----------|----------------|
| 0 | 1 | φ^0 = 1.000 | Immediate neighbor |
| 1 | 2–3 | φ^{-1} = 0.618 | Near context |
| 2 | 4–7 | φ^{-2} = 0.382 | Medium context |
| 3 | 8–12 | φ^{-3} = 0.236 | Far context |

This mirrors how attention naturally decays: nearby tokens have stronger influence, and the influence drops off in φ-spaced levels. A reference implementation:

```python
_PHI_LEVEL_RANGES = [
    (1, 1),    # level 0: distance 1
    (2, 3),    # level 1: distance 2-3
    (4, 7),    # level 2: distance 4-7
    (8, 12),   # level 3: distance 8-12
]
```

The context extractor for each level provides both the nearest and farthest token within the range, mirroring how attention considers all keys within a range rather than just the closest.

What's striking is that **4 features per direction** can cover distances 1–12, whereas a fixed-window approach would require 12 features per direction. This geometric compaction is possible because φ-decay matches the actual attention decay profile of transformers.

---

## 2.5 Why φ and Not e or π?

A natural question arises: many constants have self-similar or exponential properties. Why use φ rather than e (the base of natural logarithms) or π?

The answer lies in φ's unique combination of properties:

**e** has the property $\ln(e) = 1$ and $e^x$ is its own derivative. But e does *not* satisfy $e = 1 + 1/e$. E is about continuous growth; φ is about discrete self-similarity.

**π** is about periodicity and rotation. It appears in attention mechanisms through rotary position encodings (RoPE), but π does not provide a natural coordinate system for magnitude.

**φ** bridges the discrete and continuous. The Fibonacci numbers are integers; their ratio converges to φ. Powers of φ form a discrete lattice that densely covers the real line. And critically:

$$\ln(\phi) \approx 0.4812$$

This connects φ to e through the natural logarithm. The constant $\ln(\phi)$ appears repeatedly in transformer computations—softmax expressed in φ-form:

```python
def phi_softmax(x: np.ndarray, temperature: float = LN_PHI) -> np.ndarray:
    """Softmax as phi-level selection. softmax(x) = phi^(x/T) / sum phi^(x/T)"""
    phi_powers = PHI ** (x / temperature)
    return phi_powers / phi_powers.sum()
```

And sigmoid as:

```python
def phi_sigmoid(x: float) -> float:
    """sigmoid(x) = 1 / (1 + phi^(-x/ln(phi)))"""
    return 1 / (1 + PHI ** (-x / LN_PHI))
```

These are not approximations. As we will prove in Chapter 11, these φ-formulas are **exact** equivalences of the standard exponential forms.

---

## 2.6 φ-Exponent Arithmetic

The φ-coordinate system simplifies neural-network arithmetic in two complementary ways: multiplication becomes integer addition, and addition itself becomes a closed-form lookup. Both reductions are *exact*; neither relies on φ as an approximation.

### Multiplication: integer addition + sign XOR

Two φ-encoded numbers multiply trivially:

$$\left(s_a \cdot \phi^{e_a}\right) \cdot \left(s_b \cdot \phi^{e_b}\right) = (s_a \cdot s_b) \cdot \phi^{e_a + e_b}$$

A floating-point multiply becomes an integer add (the exponents) plus a single-bit XOR (the signs). Neural networks perform billions of multiplies per forward pass; in φ-arithmetic each one drops from a full mantissa multiply to a 16-bit integer add.

### Addition: the closed-form identity

Standard floating-point addition needs alignment, mantissa addition, and re-normalisation. φ-addition has an exact identity:

$$\phi^a + \phi^b = \phi^b \cdot (\phi^{a-b} + 1), \quad a \geq b$$

Letting $d = a - b$:

$$\phi^a + \phi^b = \phi^{b + \mathrm{LUT}_{\text{add}}[d]}, \quad \mathrm{LUT}_{\text{add}}[d] = \log_\phi\!\left(\phi^{d} + 1\right)$$

The LUT is small (a few hundred entries at the resolution used in practice), monotone in $d$, and computed once. φ-addition is therefore: one comparison (to pick the larger exponent), one LUT lookup, one integer add. Subtraction follows the analogous pattern with $\mathrm{LUT}_{\text{sub}}[d] = \log_\phi(\phi^d - 1)$.

### Why φ is the unique base with this property

The addition identity is a direct consequence of the **Fibonacci recurrence**:

$$\phi^n + \phi^{n-1} = \phi^{n+1}$$

No other positive real base has a closed-form exponent rule for addition. In a binary FPU, $2^a + 2^b$ does not equal $2^c$ for any nice integer $c$; the mantissa must be materialised. The single-base addition identity is unique to φ and is the structural reason a φ-FPU can replace IEEE 754 for neural-network workloads.

The **Zeckendorf representation** — the theorem that every positive integer has a unique expression as a sum of non-consecutive Fibonacci numbers — is the discrete dual of this property: it guarantees that the integer exponents inside the φ-FPU have a canonical form, with no redundant encodings.

### Accumulation and the empirical bit-exact result

A dot product of length 3,584 (the hidden dimension of Qwen2-7B) can amplify φ-addition rounding when many nearly-equal terms cancel. The remedy is *bucket-and-reduce*: route each term to a bucket indexed by its exponent range, accumulate within-bucket in fixed-point arithmetic, sum the bucket totals at the end. Applied to the 3,584-term dot products that constitute one row of Qwen2-7B's attention output, this achieved **0% error** — bit-exact agreement with the float32 reference. Chapter 11 takes this further and proves that the entire forward pass of Qwen2-7B is reproducible in φ-arithmetic to within machine epsilon.

The scaling advantage at network level is *not* an asymptotic complexity win (a matrix-vector product is still $O(N^2)$ scalar ops in either representation). It is a constant-factor win in the scalar primitive: each multiply-add drops from a float multiply + float add to two integer adds and an XOR. Chapter 8 reports how this compounds into the 12.9× LUT compression result on Qwen2-7B.

---

## 2.7 Summary

φ provides the coordinate system for TruthSpace's geometric theory of computation because:

1. **Self-similarity** ($\phi = 1 + 1/\phi$) ensures scale invariance
2. **φ-powers** form a discrete lattice with natural spacing
3. **φ-arithmetic** is closed under both multiplication (exponent add + sign XOR) *and* addition (closed-form LUT via the Fibonacci recurrence $\phi^n + \phi^{n-1} = \phi^{n+1}$) — a property unique to φ among positive real bases
4. **φ-decay** matches the attention profile of transformers
5. **φ and e** are connected through $\ln(\phi)$, unifying exponential and geometric views

The next chapter shows how these properties suggest a profound reinterpretation of neural networks: weights are not learned parameters but coordinates of a geometric shape that training *discovers*.
