#!/usr/bin/env bash
set -euo pipefail

# Build the paper PDF from markdown source
# Usage: ./scripts/build_paper.sh [--skip-figures]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/output"

echo "=== Building TruthSpace Paper ==="

# Pick a Python that has mpmath (needed for fig5_3's high-precision Z(t)).
# Override with FIG_PY=/path/to/python; otherwise we probe common venvs.
# The figure scripts all fall back gracefully if mpmath is missing, so
# the build still succeeds either way -- this just preserves precision.
pick_fig_py() {
    if [ -n "${FIG_PY:-}" ] && "$FIG_PY" -c "import mpmath" 2>/dev/null; then
        echo "$FIG_PY"; return
    fi
    for cand in python3 \
                "$HOME/cleanup/srt_windsurf/.venv/bin/python3" \
                "$HOME/.venv/bin/python3" \
                "$PROJECT_DIR/.venv/bin/python3"; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import mpmath" 2>/dev/null; then
            echo "$cand"; return
        fi
    done
    echo "python3"   # last resort: scripts will use the fallback path
}

# Step 1: Regenerate figures (optional, skip with --skip-figures)
if [ "${1:-}" != "--skip-figures" ]; then
    FIG_PY_RESOLVED="$(pick_fig_py)"
    if "$FIG_PY_RESOLVED" -c "import mpmath" 2>/dev/null; then
        echo "[1/4] Regenerating figures (using $FIG_PY_RESOLVED, mpmath OK)..."
    else
        echo "[1/4] Regenerating figures (using $FIG_PY_RESOLVED, no mpmath -- fig5_3 will use Riemann-Siegel fallback)..."
    fi
    cd "$OUTPUT_DIR/figures/scripts"
    for fig in fig*.py; do
        echo "  $fig"
        "$FIG_PY_RESOLVED" "$fig" 2>/dev/null
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
    'irreducible_shape', 'phi_computer_proof', 'implications',
    'appendix_b_critical_line'
], 1)]

header = '''---
title: \"TruthSpace: A Geometric Theory of Neural Computation\"
subtitle: \"From the Vacuum Forming Hypothesis to the phi-Computer Proof\"
author: \"TruthSpace Geometric LCM Project\"
date: \"February 2026\"
subject: \"Geometric AI\"
keywords: [\"phi\", \"golden ratio\", \"geometric computation\", \"transformer\", \"Qwen2-7B\", \"phi-lattice\", \"navigation\", \"irreducible shape\"]
lang: en
documentclass: article
classoption:
  - twocolumn
  - 10pt
  - a4paper
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

# Step 4: Compile PDF with pandoc + xelatex (two-stage so we can fix
# pandoc's longtable output to work inside twocolumn).
echo "[4/4] Compiling paper.pdf..."
cd "$OUTPUT_DIR"

# 4a: pandoc -> .tex (standalone)
pandoc paper.md \
    --from markdown \
    --to latex \
    --standalone \
    --include-in-header="$SCRIPT_DIR/preamble.tex" \
    -V mainfont="DejaVu Serif" \
    -V monofont="DejaVu Sans Mono" \
    -o paper.tex

# 4b: rewrite longtable blocks for twocolumn compatibility.  Pandoc
# emits each table as:
#     \begin{longtable}[]{...}
#     \toprule\noalign{}
#     HEADER \\
#     \midrule\noalign{}
#     \endhead
#     \bottomrule\noalign{}      <- belongs at end, not after header
#     \endlastfoot
#     ROWS
#     \end{longtable}
# The preamble redefines longtable->tabular, but the misplaced
# \bottomrule after \endhead then appears before the rows.  Strip it
# here and let the preamble append \bottomrule at \end{longtable}.
python3 - <<'PYFIX'
import re
src = open('paper.tex').read()
# Drop the \bottomrule\noalign{} that appears immediately after
# \endhead (this only matches inside longtable blocks, since that
# pattern doesn't appear elsewhere).
src = re.sub(
    r'(\\endhead\s*\n)\\bottomrule\\noalign\{\}\s*\n\\endlastfoot\s*\n',
    r'\1',
    src,
)
# At the end of every (former) longtable block we still need a
# \bottomrule.  The redefined longtable in preamble.tex wraps content
# in a tabular; pandoc's row format ends with `\\` so we add a final
# rule just before \end{longtable} (which the preamble has redefined
# to close the tabular).
src = re.sub(
    r'(\n)(\\end\{longtable\})',
    r'\1\\bottomrule\\noalign{}\n\2',
    src,
)
# Force \maketitle and \tableofcontents into single-column mode (in
# twocolumn class, they would otherwise render as cramped two-column
# blocks).  Wrap the title + TOC in \onecolumn ... \twocolumn.
src = re.sub(
    r'(\\begin\{document\}\s*\n)(\\maketitle\s*\n\s*\n\{\s*\n\\setcounter\{tocdepth\}\{\d+\}\s*\n\\tableofcontents\s*\n\})',
    r'\1\\onecolumn\n\2\n\\twocolumn',
    src,
)
open('paper.tex', 'w').write(src)
PYFIX

# 4c: xelatex twice (for TOC) -- final stage
xelatex -interaction=nonstopmode -halt-on-error paper.tex > paper.xelatex.log 2>&1 || {
    echo "  xelatex pass 1 failed -- see output/paper.xelatex.log"
    exit 1
}
xelatex -interaction=nonstopmode -halt-on-error paper.tex > paper.xelatex.log 2>&1 || {
    echo "  xelatex pass 2 failed -- see output/paper.xelatex.log"
    exit 1
}

# Clean up aux/log clutter (keep .pdf, .tex)
rm -f paper.aux paper.log paper.out paper.toc paper.xelatex.log

echo ""
echo "=== Build complete ==="
echo "PDF: $OUTPUT_DIR/paper.pdf ($(du -h "$OUTPUT_DIR/paper.pdf" | cut -f1))"
