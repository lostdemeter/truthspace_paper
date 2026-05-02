#!/usr/bin/env python3
"""
Phi-Encoding and phi-Dial Demo (Batch 2)
=========================================
Demonstrates phi-encoding, quaternion dial, and gear chain concepts.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


def to_phi_coord(x):
    if abs(x) < 1e-15: return (1, -100, 0.0)
    sign = 1 if x > 0 else -1
    abs_x = abs(x)
    level = int(np.floor(np.log(abs_x) / LN_PHI))
    base = PHI ** level
    residual = (abs_x / base - 1) / (PHI - 1)
    return (sign, level, np.clip(residual, 0, 1 - 1e-10))


def from_phi_coord(sign, level, residual):
    return sign * (PHI ** level) * (1 + residual * (PHI - 1))


class Quaternion:
    """Simple Quaternion for gear chain accumulation."""
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w, self.x, self.y, self.z = w, x, y, z
    def __mul__(self, o):
        return Quaternion(
            self.w*o.w - self.x*o.x - self.y*o.y - self.z*o.z,
            self.w*o.x + self.x*o.w + self.y*o.z - self.z*o.y,
            self.w*o.y - self.x*o.z + self.y*o.w + self.z*o.x,
            self.w*o.z + self.x*o.y - self.y*o.x + self.z*o.w)
    def __repr__(self):
        return f"Q({self.w:.2f}, {self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class PhiDialSpace:
    """A geometric space with phi-dial control."""
    def __init__(self, dims=4):
        self.dims = dims
        self.points = {}
    
    def place(self, name, position):
        self.points[name] = np.array(position)
    
    def query(self, name, dial=0.0):
        """Find nearest neighbor with phi-dial modulation."""
        if name not in self.points:
            return None
        center = self.points[name]
        # dial modulates distance: dial>0 = further out, dial<0 = closer in
        modulated = {}
        for k, v in self.points.items():
            if k == name: continue
            raw_dist = np.linalg.norm(v - center)
            # phi-dial: dial = -1 (specific), 0 (neutral), +1 (general)
            adjusted = raw_dist * (PHI ** (-dial))
            modulated[k] = adjusted
        return min(modulated, key=modulated.get), modulated


def main():
    print("=" * 60)
    print("Batch 2 Demo: phi-Encoding, phi-Dial, and Gears")
    print("=" * 60)
    
    # Demo 1: phi-Encoding roundtrip
    print("\n--- Demo 1: phi-Encoding (with residual) ---")
    values = [0.0, 0.5, 1.0, PHI, 2.718, 10.0, -1.618]
    for v in values:
        s, l, r = to_phi_coord(v)
        reconstructed = from_phi_coord(s, l, r)
        err = abs(v - reconstructed) / max(abs(v), 1e-10)
        print(f"  {v:>8.4f} -> ({s:+d}, {l:>2d}, {r:.4f}) -> {reconstructed:>8.4f}  error={err:.2e}")
    
    # Demo 2: phi-Dial control
    print("\n--- Demo 2: phi-Dial Semantic Control ---")
    space = PhiDialSpace(dims=4)
    space.place("cat", [0.1, 0.8, 0.2, 0.3])
    space.place("dog", [0.2, 0.9, 0.1, 0.4])
    space.place("pet", [0.15, 0.5, 0.3, 0.1])
    space.place("animal", [0.3, 0.2, 0.5, 0.0])
    space.place("mammal", [0.25, 0.3, 0.4, 0.1])
    
    for dial in [-1.0, 0.0, 1.0]:
        nearest, all_dists = space.query("cat", dial)
        direction = "specific (inward)" if dial < 0 else ("balanced" if dial == 0 else "general (outward)")
        print(f"  dial={dial:+.1f} ({direction}): nearest to 'cat' = '{nearest}'")
        for k, d in sorted(all_dists.items(), key=lambda x: x[1]):
            print(f"    {k}: dist={d:.4f}")
    
    # Demo 3: Quaternion gear chain accumulation
    print("\n--- Demo 3: Quaternion Accumulation in Gear Chain ---")
    identity = Quaternion(1, 0, 0, 0)
    role_gear = Quaternion(0.923, 0.382, 0, 0)        # role transform
    action_gear = Quaternion(0.923, 0, 0.382, 0)      # action transform
    output_gear = Quaternion(0.923, 0, 0, 0.382)       # output transform
    
    total = identity * role_gear * action_gear * output_gear
    print(f"  Identity:  {identity}")
    print(f"  Role gear: {role_gear}")
    print(f"  Action gear: {action_gear}")
    print(f"  Output gear: {output_gear}")
    print(f"  Total:     {total}")
    print(f"  Norm:      {np.sqrt(total.w**2+total.x**2+total.y**2+total.z**2):.4f}")
    
    # Demo 4: Semantic analogy via quaternion offset
    print("\n--- Demo 4: Semantic Quaternion Analogy ---")
    # In phi-space: king - man + woman = queen
    king = np.array([0.8, 0.2, 0.5, 0.3])
    man = np.array([0.7, 0.3, 0.4, 0.2])
    woman = np.array([-0.7, 0.3, 0.4, -0.2])
    queen_pred = king - man + woman
    queen_actual = np.array([-0.8, 0.2, 0.5, -0.3])
    similarity = np.dot(queen_pred, queen_actual) / (np.linalg.norm(queen_pred) * np.linalg.norm(queen_actual))
    print(f"  king - man + woman = predicted queen offset")
    print(f"  Cosine similarity to actual queen: {similarity:.4f}")
    print(f"  100% match: {similarity > 0.99}")
    
    # ENCODE = DECODE demonstration
    print("\n--- Demo 5: ENCODE = DECODE ---")
    x = 2.5
    encoded = x * PHI
    decoded = encoded / PHI
    print(f"  x = {x}")
    print(f"  encode(x) = x * phi = {encoded:.4f}")
    print(f"  decode(encode(x)) = encode(x) / phi = {decoded:.4f}")
    print(f"  x == decode(encode(x)): {abs(x - decoded) < 1e-10}")
    print(f"  phi * 1/phi = {PHI * (1/PHI):.15f}")

if __name__ == '__main__':
    main()
