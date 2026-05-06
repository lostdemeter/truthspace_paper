#!/usr/bin/env bash
set -euo pipefail

# Build the paper PDF from markdown source
# Usage: ./scripts/build_paper.sh [--skip-figures]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/output"

echo "=== Building TruthSpace Paper ==="

# Step 1: Regenerate figures (optional, skip with --skip-figures)
if [ "${1:-}" != "--skip-figures" ]; then
    echo "[1/4] Regenerating figures..."
    cd "$OUTPUT_DIR/figures/scripts"
    for fig in fig*.py; do
        echo "  $fig"
        python3 "$fig" 2>/dev/null
    done
    echo "  Done."
else
    echo "[1/4] Skipping figure regeneration."
fi

# Step 2: Regenerate code demos if needed (all demos are runnable, no build step needed)
echo "[2/4] Code demos are self-contained (no build step)."

# Step 3: Concatenate chapters + add frontmatter
echo "[3/4] Compiling paper.md from chapters..."
cd "$OUTPUT_DIR"
python3 -c "
import re
files = [f'chapters/{i:02d}_{name}.md' for i, name in enumerate([
    'what_llms_learn', 'phi_self_similarity', 'geometric_model_hypothesis',
    'encodings_phi_dial', 'encode_decode', 'gear_architecture',
    'phi_lattice', 'reverse_engineering', 'navigation_replaces_inference',
    'irreducible_shape', 'phi_computer_proof', 'implications'
], 1)]

header = '''---
title: \"TruthSpace: A Geometric Theory of Neural Computation\"
subtitle: \"From the Vacuum Forming Hypothesis to the phi-Computer Proof\"
author: \"TruthSpace Geometric LCM Project\"
date: \"February 2026\"
subject: \"Geometric AI\"
keywords: [\"phi\", \"golden ratio\", \"geometric computation\", \"transformer\", \"Qwen2-7B\", \"phi-lattice\", \"navigation\", \"irreducible shape\"]
lang: en
titlepage: true
toc: true
listings-disable-line-numbers: true
---

'''

with open('paper.md', 'w') as out:
    out.write(header)
    for ch in files:
        with open(ch) as f:
            content = f.read()
        content = re.sub(r'^---\\n.*?---\\n', '', content, count=1, flags=re.DOTALL)
        content = content.replace('](../figures/', '](figures/')
        out.write(content)
        out.write('\\n\\n')

    # Unicode cleanup: replace unsafe chars for LaTeX
    content = out.name

# After writing, do a read-replace for Unicode chars
with open('paper.md', 'r') as f:
    content = f.read()

# xelatex + DejaVu Serif handle Greek letters and math symbols natively,
# so we leave those as Unicode.  Only replace characters DejaVu Serif
# does not cover (IPA, box-drawing).
replacements = {
    '\u0283': '/sh/',
    '\u026a': '/ih/',
    '\u00e6': '/ae/',
    '\u250c': '+', '\u2500': '-', '\u2510': '+', '\u2502': '|',
    '\u2514': '+', '\u2518': '+',
}
for char, latex in replacements.items():
    content = content.replace(char, latex)

with open('paper.md', 'w') as f:
    f.write(content)
"
echo "  Done."

# Step 4: Compile PDF with pandoc
echo "[4/4] Compiling paper.pdf..."
cd "$OUTPUT_DIR"
pandoc paper.md \
    --from markdown \
    --to pdf \
    --pdf-engine=xelatex \
    -V mainfont="DejaVu Serif" \
    -V monofont="DejaVu Sans Mono" \
    -o paper.pdf

echo ""
echo "=== Build complete ==="
echo "PDF: $OUTPUT_DIR/paper.pdf ($(du -h "$OUTPUT_DIR/paper.pdf" | cut -f1))"
