#!/usr/bin/env python3
"""
phi-Exponent Arithmetic Demo
=============================
Demonstrates phi-coordinate encoding and arithmetic operations.
Shows how numbers become sign x phi^level, and how multiplication
reduces to simple exponent addition.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


def to_phi_coord(x: float) -> tuple:
    """Convert float to (sign, level, residual) in phi-space.
    
    value = sign x phi^level x (1 + residual x (phi-1))
    """
    if abs(x) < 1e-15:
        return (1, -100, 0.0)

    sign = 1 if x > 0 else -1
    abs_x = abs(x)

    log_phi_x = np.log(abs_x) / LN_PHI
    level = int(np.floor(log_phi_x))

    base = PHI ** level
    residual = (abs_x / base - 1) / (PHI - 1)
    residual = np.clip(residual, 0, 1 - 1e-10)

    return (sign, level, residual)


def from_phi_coord(sign: int, level: int, residual: float) -> float:
    """Convert phi-coordinate back to float."""
    return sign * (PHI ** level) * (1 + residual * (PHI - 1))


def phi_sigmoid(x: float) -> float:
    """Sigmoid as phi-operation: 1 / (1 + phi^(-x/ln(phi)))"""
    return 1 / (1 + PHI ** (-x / LN_PHI))


def phi_softmax(x: np.ndarray, temperature: float = None) -> np.ndarray:
    """Softmax as phi-level selection: phi^(x/T) / sum phi^(x/T)"""
    if temperature is None:
        temperature = LN_PHI
    phi_powers = PHI ** (x / temperature)
    return phi_powers / phi_powers.sum()


def main():
    print("=" * 60)
    print("phi-Exponent Arithmetic Demo")
    print("=" * 60)

    # Demo 1: phi-coordinate encoding
    print("\n--- Demo 1: phi-Coordinate Encoding ---")
    test_values = [0.001, 0.1, 0.618, 1.0, 1.618, 2.618, 10.0, 100.0, -1.618]

    print(f"{'Value':>12} {'Sign':>6} {'Level':>6} {'Residual':>10} {'Reconstructed':>15} {'Error':>10}")
    print("-" * 65)
    for v in test_values:
        sign, level, residual = to_phi_coord(v)
        reconstructed = from_phi_coord(sign, level, residual)
        error = abs(v - reconstructed) / max(abs(v), 1e-10)
        print(f"{v:>12.6f} {sign:>6} {level:>6} {residual:>10.6f} {reconstructed:>15.6f} {error:>10.2e}")

    # Demo 2: Multiplication as exponent addition
    print("\n--- Demo 2: Multiplication as Exponent Addition ---")
    a, b = 3.0, 5.0
    a_enc = to_phi_coord(a)
    b_enc = to_phi_coord(b)
    product_float = a * b
    product_phi = from_phi_coord(a_enc[0] * b_enc[0], a_enc[1] + b_enc[1], 0.0)

    print(f"Standard:      {a} x {b} = {product_float}")
    print(f"phi-coord:     {a_enc} + {b_enc}")
    print(f"phi-multiply:  sign={a_enc[0]*b_enc[0]}, level={a_enc[1]+b_enc[1]}")
    print(f"phi-result:    {product_phi}")
    print(f"Error:         {abs(product_float - product_phi):.6f}")

    # Demo 3: phi-sigmoid equivalence
    print("\n--- Demo 3: phi-Sigmoid vs Standard Sigmoid ---")
    from scipy.special import expit as sigmoid

    x_values = np.linspace(-5, 5, 11)
    max_diff = 0
    print(f"{'x':>8} {'sigmoid(x)':>12} {'phi-sigmoid(x)':>14} {'Difference':>12}")
    print("-" * 50)
    for x in x_values:
        std = sigmoid(x)
        phi = phi_sigmoid(x)
        diff = abs(std - phi)
        max_diff = max(max_diff, diff)
        print(f"{x:>8.2f} {std:>12.6f} {phi:>14.6f} {diff:>12.2e}")

    print(f"\nMax difference: {max_diff:.2e}")
    print(f"IDENTICAL: {max_diff < 1e-14}")

    # Demo 4: phi-softmax
    print("\n--- Demo 4: phi-Softmax ---")
    logits = np.array([2.0, 1.0, 0.5, 1.5])
    probs = phi_softmax(logits)
    print(f"Logits:  {logits}")
    print(f"Probs:   {probs}")
    print(f"Sum:     {probs.sum():.6f}")
    print(f"Argmax:  {np.argmax(probs)} (phi-level {logits[0]:.2f})")

    # Demo 5: phi-Weight clustering demonstration
    print("\n--- Demo 5: Weights Cluster at phi-Levels ---")
    np.random.seed(42)
    n_weights = 20
    phi_levels = np.random.randint(-5, 6, n_weights)
    weights = PHI ** phi_levels + np.random.normal(0, 0.1, n_weights)

    for w in sorted(weights):
        sign, level, residual = to_phi_coord(w)
        print(f"  weight={w:>8.4f} -> sign={sign:+d}, level={level:>2d}, residual={residual:.4f}")

    print(f"\nphi = {PHI:.15f}")
    print(f"ln(phi) = {LN_PHI:.15f}")
    print(f"phi^1 = {PHI**1:.6f}")
    print(f"phi^-1 = {PHI**-1:.6f}")


if __name__ == '__main__':
    main()
