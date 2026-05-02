#!/usr/bin/env python3
"""
Phase Discovery Archetypes Demo (self-contained)
=================================================
Demonstrates PhaseDiscovery: given example (input, output) pairs,
automatically discovers the transformation structure.

No torch or neural networks. Pure information-theoretic structure detection.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from phi_geometric.core.phase_discovery import PhaseDiscovery


def _map_pairs():
    MAP = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G'}
    def apply(seq): return [MAP.get(t, t) for t in seq]
    training = [['a','b','c'], ['d','e','f'], ['g','a','b'], ['c','d','e']]
    return [(s, apply(s)) for s in training], apply


def _context_map_pairs():
    def apply(seq):
        result = []
        for i, tok in enumerate(seq):
            nxt = seq[i+1] if i+1 < len(seq) else None
            result.append('s' if tok == 'c' and nxt == 'i' else ('k' if tok == 'c' else tok))
        return result
    training = [['c','a','t'], ['c','i','t','y'], ['c','o','t'], ['c','i','t']]
    return [(s, apply(s)) for s in training], apply


def _collapse_map_pairs():
    CHORDS = {('C','E'): ('Cmaj',), ('D','F'): ('Dmin',),
              ('E','G'): ('Emin',), ('G','B'): ('Gmaj',)}
    MAP = {'A': 'La', 'F': 'Fa', 'C': 'Do'}
    def apply(seq):
        result = []; i = 0
        while i < len(seq):
            if i+1 < len(seq) and (seq[i], seq[i+1]) in CHORDS:
                result.extend(CHORDS[(seq[i], seq[i+1])]); i += 2
            else:
                result.append(seq[i]); i += 1
        return [MAP.get(t,t) for t in result]
    training = [
        ['C','E','A'], ['D','F','A'], ['E','G','A'], ['G','B','A'],
        ['C','E','D','F'], ['G','B','C','E'], ['A','C','E'], ['A','D','F'],
        ['F','G','B'], ['C','E','G','B'], ['A','E','G','F'],
        ['D','F','G','B','A'], ['C','E','F'], ['D','F','C'],
        ['A','C','E','A'], ['F','D','F','A'],
    ]
    return [(s, apply(s)) for s in training], apply


def _expand_map_pairs():
    MAP = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E'}
    def apply(seq):
        result = []
        for tok in seq:
            if tok == 'x': result.extend(['k','s'])
            elif tok == 'q': result.extend(['k','w'])
            else: result.append(MAP.get(tok, tok))
        return result
    training = [
        ['a','x','b'], ['c','x','d'], ['e','x','a'], ['b','x','c'],
        ['a','q','b'], ['c','q','d'], ['e','q','a'], ['b','q','c'],
        ['x','a','q'], ['q','x','a'],
        ['a','b','c'], ['d','e','a'], ['b','c','d'],
    ]
    return [(s, apply(s)) for s in training], apply


ARCHETYPES = {
    'map': _map_pairs,
    'context_map': _context_map_pairs,
    'collapse_map': _collapse_map_pairs,
    'expand_map': _expand_map_pairs,
}


def main():
    import time
    print("=" * 60)
    print("PhaseDiscovery: Archetype Validation")
    print("No gradients. No neural networks. Pure information geometry.")
    print("=" * 60)

    results = {}
    for name in ['map', 'context_map', 'collapse_map', 'expand_map']:
        pairs, _ = ARCHETYPES[name]()
        context = 3 if name == 'context_map' else 1
        pd = PhaseDiscovery(context_window=context)
        for inp, out in pairs:
            pd.add_pair(inp, out)

        t0 = time.time()
        result = pd.discover()
        elapsed = time.time() - t0

        nav = result.to_navigator()
        correct = sum(1 for inp, out in pairs
                     if nav.execute(inp).output_elements == out)
        total = len(pairs)
        accuracy = correct / total * 100

        results[name] = {
            'phases': result.n_phases,
            'rules': result.n_rules,
            'archetype': result.archetype,
            'accuracy': accuracy,
            'time': elapsed,
        }

        print(f"\n  {name:<20} {result.archetype:<20} "
              f"acc={accuracy:4.0f}%  time={elapsed:.3f}s  "
              f"phases={result.n_phases} rules={result.n_rules}")

        if accuracy < 100:
            # Show sample error
            for inp, expected in pairs:
                trace = nav.execute(inp)
                if trace.output_elements != expected:
                    print(f"    Error: {inp} -> got {trace.output_elements}, "
                          f"expected {expected}")
                    break

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Archetype':<25} {'Signature':<20} {'Accuracy':<10} {'Time':<8}")
    print("-" * 63)
    for name, r in results.items():
        print(f"{name:<25} {r['archetype']:<20} {r['accuracy']:3.0f}%{r['time']:>7.3f}s")
    print()
    print("Note: Full 8/8 archetype validation (100% accuracy) is available")
    print("at phi_geometric/examples/archetypes.py with the official test set.")


if __name__ == '__main__':
    main()
