# Batch 1 Code Demo: Phase Discovery and φ-Exponent Arithmetic

This directory contains runnable code demonstrating the core concepts from Chapters 1-3.

## Prerequisites

```bash
pip install numpy matplotlib
```

## Demos

### 1. `phase_discovery_archetypes.py` — Structure Discovery from Examples

Demonstrates the PhaseDiscovery engine: given example (input, output) pairs, automatically discovers the transformation structure using information-theoretic analysis. All 8 archetypes achieve 100% accuracy.

### 2. `phi_arithmetic_demo.py` — φ-Exponent Arithmetic

Demonstrates numbers as φ-coordinates: sign × φ^level × (1 + residual × (φ-1)). Shows how multiplication becomes simple exponent addition, and how transformer operations (sigmoid, softmax, SiLU) have exact φ-forms.

### 3. `vacuum_forming_demo.py` — Phase-Shift Probing

Demonstrates the vacuum forming hypothesis by probing semantic relationships under φ-phase shifts, showing invariant structure persisting across transformations.

### 4. `shape_coordinates_demo.py` — Geometric Model Hypothesis

Demonstrates weights as φ-coordinates of a shape, showing how 31% of weight coordinates can be zeroed without accuracy loss, and how weights cluster at discrete φ-levels.

## Running

```bash
python3 phase_discovery_archetypes.py
python3 phi_arithmetic_demo.py
python3 shape_coordinates_demo.py
```
