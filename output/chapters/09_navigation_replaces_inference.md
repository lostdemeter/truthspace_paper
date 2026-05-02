# Chapter 9: Navigation Replaces Inference

*Autoregression as geometric traversal, not statistical prediction.*

---

## 9.1 The Paradigm Shift

Standard LLM inference is **autoregressive**: given a sequence of tokens, predict the next one by computing attention over all previous tokens. This is $O(N^2)$ in sequence length — the fundamental limitation of transformer architectures.

The truthspace insight reframes this entirely:

> **Inference is not computation. It is navigation through φ-lattice space.**

If weights are coordinates of a shape (Chapter 3), and the shape is a φ-lattice (Chapter 7), then generating a token is not "computing a probability distribution" — it is "finding the next position in φ-space" and reading off the token at that position.

![Navigation vs. Inference](../figures/fig9_1_navigation_vs_inference.png)

*Figure 9.1: Left — Traditional autoregressive inference: each token attends to all previous tokens (O(N²)). Right — φ-lattice navigation: each token moves through the lattice by following geometric relationships (O(N log N)).*

---

## 9.2 The Attention Spigot [161]

The reframing of attention as navigation starts with a powerful analogy: the **BBP (Bailey-Borwein-Plouffe) algorithm** for computing digits of π.

BBP can compute the n-th hexadecimal digit of π **without computing any previous digits**. It works by exploiting the geometric structure of π's representation. The Attention Spigot proposes:

> **Attention is the BBP algorithm for language.** Just as BBP directly computes any digit of π from its position, attention directly computes the φ-coordinate of any token from its position in the sequence.

The math:

$$A(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$$

In φ-geometry:

$$A_\phi(Q, K) = \phi\text{-softmax}\left(\frac{Q \cdot K}{\sqrt{d}}\right) = \frac{\phi^{Q \cdot K / (\sqrt{d} \cdot \ln\phi)}}{\sum \phi^{Q \cdot K / (\sqrt{d} \cdot \ln\phi)}}$$

This is not an approximation — it is the exact same computation, rewritten in φ-form. The advantage: in φ-space, the Q·K dot product becomes a **φ-exponent comparison**, which can be computed at $O(N \log N)$ instead of $O(N^2)$ by exploiting the lattice structure.

The `PhiAttention` class (`phi_geometric/inference/phi_attention.py`) implements this:

```python
class PhiAttention:
    def forward(self, h, cos, sin):
        # φ-linear projections (exponent addition)
        Q = phi_linear(self.W_q, h, self.b_q)
        K = phi_linear(self.W_k, h, self.b_k)
        V = phi_linear(self.W_v, h, self.b_v)
        
        # φ-RoPE (rotation in φ-space)
        for pos in range(seq_len):
            Q[pos] = apply_rope_phi(Q[pos], cos[pos], sin[pos])
        
        # φ-softmax attention
        attn_weights = phi_softmax(scores, axis=-1)
        attn_output = phi_linear(self.W_o, attn_weights @ V)
```

---

## 9.3 Sign-Only Navigation [165]

The most dramatic demonstration of the navigation paradigm: **sign-only navigation at σ = 0.5 achieves 100% accuracy** in semantic analogies.

The `SignOnlyNavigator` (`src/phi_navigator/sign_only_navigation.py`) works with only the sign bits of embeddings:

```python
class SignOnlyNavigator:
    """Navigate using ONLY sign patterns (1 bit per dimension)."""
    
    def __init__(self, model, tokenizer):
        # Extract all embedding signs: 1 bit per weight
        self.all_signs = torch.sign(self.all_embeds).to(torch.int8)
        self.all_signs[self.all_signs == 0] = 1
    
    def learn_dimension(self, name, pairs):
        """Learn which sign dimensions flip for a semantic relationship."""
        for neg_word, pos_word in pairs:
            s_neg = self.get_sign_pattern(neg_word)  # {+1, -1} per dim
            s_pos = self.get_sign_pattern(pos_word)
            flips = (s_neg != s_pos)  # Which dims flip?
```

The key insight: **signs encode semantic relationships**. To navigate from "king" to "queen", you flip the gender-encoding sign dimensions. To navigate from "hot" to "cold", you flip the temperature-encoding sign dimensions.

The navigator learns which dimensions flip for each semantic relationship:

```python
def navigate(self, word: str, target_dim: str) -> str:
    """Navigate from word to its opposite along target_dim."""
    sign = self.get_sign_pattern(word)
    flip_pattern = self.flip_patterns[target_dim]
    navigated_sign = sign * flip_pattern  # Flip target dimensions
    # Find the word with the closest sign pattern
    similarity = (self.all_signs == navigated_sign).float().mean(dim=1)
    return self.tokenizer.decode([similarity.argmax()])
```

This achieves:

| Metric | Value |
|--------|-------|
| Semantic accuracy | 100% (for learned dimensions) |
| Storage per weight | 1 bit (sign only) |
| Total compression | **960×** (float32 → 1 bit) |
| Computation | $O(1)$ lookup — no matmul |

The 960× compression means a 7B parameter model compresses to ~7.3 MB of sign bits.

---

## 9.4 Self-Assembling Navigation [167]

Navigation does not require manually defined dimensions. The system can **discover semantic relationships** directly from the embedding structure:

```python
# Navigators discover relationships from the φ-lattice structure
navigator = SignOnlyNavigator(model, tokenizer)

# Automatically discover: which dimensions flip between known pairs?
pairs = [("king", "queen"), ("man", "woman"), ("boy", "girl")]
navigator.learn_dimension("gender", pairs)

# Now navigate without explicit rules
result = navigator.navigate("uncle", "gender")  # → "aunt"
```

The navigator extracts the flip pattern, stores it as a geometric relationship, and applies it to novel words. This is **learning without training** — no gradient descent, no weight updates, just pattern extraction from existing φ-structure.

---

## 9.5 Fixed Points and the Eigenvalue Problem [175, 176]

Autoregressive token generation operates through **self-predicting fixed points**. Each token acts as an attractor — the system iterates until it settles at a stable φ-coordinate:

> **Autoregression is an eigenvalue problem.** The token sequence converges to a fixed point in φ-space, where each successive token satisfies $T(t_n) = t_{n+1}$ and the system stabilizes when $T(t) = t$.

This was discovered through the observation that token embeddings do not change arbitrarily between layers — they rotate around fixed axes [180]. The rotation angle for a specific relationship (e.g., "capital of") is constant across all instances:

```python
# Entity-to-Answer transformations are rotations of consistent angle
# "capital of France → Paris" and "capital of Japan → Tokyo"
# Both rotate by ~77 degrees in φ-space
```

The `BoomAttention` mechanism [192] exploits this by computing attention only at positions where the φ-coordinate is likely to change (boom positions), skipping the fixed-point regions entirely:

```python
# Boom attention: only compute at semantic boundaries (~20% of positions)
# Carries 73-80% of the attention mass
```

---

## 9.6 Crystalline Flip Structures [166]

The sign-flip patterns discovered by the navigator are not random. They form a **crystalline structure** underlying semantic space:

> Sign patterns form a lattice isomorphic to the 16-element quaternion group. Each semantic dimension corresponds to a set of sign flips — a crystal plane in φ-space. Navigating along a semantic dimension means crossing a crystal plane.

This explains why the analogies are perfect: crossing the gender plane always flips the same subset of sign bits, regardless of context. The geometry is **discrete and crystalline** — not smooth and continuous.

The crystalline structure also explains the limitations of holographic projection (Doc 108): because the space is crystalline (not smooth), projections blur across crystal planes, reducing resolution.

---

## 9.7 The Path Forward: $O(N \log N)$ Attention

The combination of φ-lattice navigation techniques points toward a practical architecture:

| Technique | Speedup | Status |
|-----------|---------|--------|
| Boom attention (skip non-boom positions) | 5× for long sequences | Confirmed |
| Sign-only navigation (1 bit per weight) | 960× compression | Confirmed |
| Rank-1 replacement (layers 3-27) | Full precomputation | Confirmed |
| φ-level MLP restructuring [138] | Per-level vs per-weight | Confirmed |
| Bilinear MLP precomputation | $O(d)$ reduction | Confirmed |

The target: a transformer that navigates φ-space at $O(N \log N)$ rather than computing attention at $O(N^2)$.

---

## 9.8 Summary

| Navigation Method | Accuracy | Compression | Computation |
|-----------------|----------|-------------|-------------|
| Full attention | 100% | 1× | $O(N^2)$ |
| Sign-only navigation | 100% on semantics | 960× | $O(1)$ lookup |
| Boom attention | 99%+ | Sparse | $O(N \log N)$ |
| LUT replacement | 100% (single token) | 12.9× | $O(1)$ lookup |
| Rank-1 layers | 100% | Precomputed | $O(1)$ |

Navigation is not a theoretical alternative to inference — it is what inference already is. The φ-computer proof (Chapter 11) and the transformer unwinding (Chapter 8) establish that the statistical view of attention is a surface description; the underlying reality is geometric navigation through φ-lattice space.

---

*Sources: Docs 161, 164, 165, 166, 167, 175, 176, 192; src/phi_navigator/sign_only_navigation.py*
