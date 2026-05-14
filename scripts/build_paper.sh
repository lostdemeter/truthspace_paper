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
#     \begin{longtable}[]{@{}<spec>@{}}
#     \toprule\noalign{}
#     HEADER \\
#     \midrule\noalign{}
#     \endhead
#     \bottomrule\noalign{}      <- belongs at end, not after header
#     \endlastfoot
#     ROWS
#     \end{longtable}
# longtable is incompatible with twocolumn mode, so we rewrite each
# block as
#     \begin{tsxtable}\begin{tabularx}{\linewidth}{<Y-spec>}
#     ...
#     \end{tabularx}\end{tsxtable}
# where <Y-spec> has every l/c/r replaced with the Y column type
# (raggedright X) defined in preamble.tex, so cells auto-wrap.
# The misplaced \bottomrule\noalign{} between \endhead and
# \endlastfoot is stripped, and a final \bottomrule is inserted
# immediately before \end{tabularx}.
python3 - <<'PYFIX'
import re
src = open('paper.tex').read()

# Step A: Drop the misplaced \bottomrule\noalign{} immediately after
# \endhead inside every longtable block (longtable's old syntax).
src = re.sub(
    r'(\\endhead\s*\n)\\bottomrule\\noalign\{\}\s*\n\\endlastfoot\s*\n',
    r'\1',
    src,
)

# Step B: Rewrite each \begin{longtable}[]{@{}<spec>@{}} ...
# \end{longtable} block as \begin{tsxtable}\begin{tabularx}{\linewidth}
# {<Y-spec>} ... \bottomrule\noalign{}\end{tabularx}\end{tsxtable}.
# Specs that already use p{...} or X are mapped 1:1.
def _rewrite_longtable(m):
    inner = m.group(1)  # whatever sits between @{} ... @{}
    body = m.group(2)
    if 'p{' in inner or '>{' in inner or 'X' in inner:
        # Already a proportional / X spec; rewrite \columnwidth ->
        # \linewidth so wide tables placed inside table* (which set
        # \linewidth to \textwidth) actually fill the page width.
        new_inner = inner.replace(r'\columnwidth', r'\linewidth')
    else:
        # Plain l/c/r columns: map to Y (auto-wrapping raggedright X).
        new_inner = re.sub(r'[lcr]', 'Y', inner)
    return (
        f'\\begin{{tsxtable}}\\begin{{tabularx}}{{\\linewidth}}'
        f'{{@{{}}{new_inner}@{{}}}}'
        f'{body}\\bottomrule\\noalign{{}}\n'
        f'\\end{{tabularx}}\\end{{tsxtable}}'
    )

# Match \begin{longtable}[]{@{}<spec>@{}}<body>\end{longtable}.
# <spec> may contain `{...}` (proportional p{...} columns), so we
# match the inner content non-greedily between literal @{} markers.
src = re.sub(
    r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}(.*?)\\end\{longtable\}',
    _rewrite_longtable,
    src,
    flags=re.DOTALL,
)

# Force \maketitle and \tableofcontents into single-column mode (in
# twocolumn class, they would otherwise render as cramped two-column
# blocks).  Wrap the title + TOC in \onecolumn ... \twocolumn.
src = re.sub(
    r'(\\begin\{document\}\s*\n)(\\maketitle\s*\n\s*\n\{\s*\n\\setcounter\{tocdepth\}\{\d+\}\s*\n\\tableofcontents\s*\n\})',
    r'\1\\onecolumn\n\2\n\\twocolumn',
    src,
)

# Promote EVERY figure float to span both columns.  Pandoc emits
# every alone-in-paragraph image as a column-width \begin{figure}
# float; at that width every single-panel figure ends up squished
# (especially titles, axis labels, and legends generated at the
# default matplotlib DPI).  Promoting to figure* gives every figure
# the full text width and the keepaspectratio default scales the
# height accordingly, so nothing is distorted -- the figure simply
# renders at the size matplotlib intended.
# dblfloatfix (loaded in preamble.tex) extends figure*'s placement
# options so [!tbp] becomes valid: t = top of page, b = bottom of
# page, p = float page, ! = override the float quota.  Without
# dblfloatfix, b would be silently ignored for wide floats.
def _widen(m):
    block = m.group(0)
    block = block.replace(
        r'\begin{figure}', r'\begin{figure*}[!tbp]', 1
    )
    block = block.replace(r'\end{figure}', r'\end{figure*}', 1)
    return block

src = re.sub(
    r'\\begin\{figure\}.*?\\end\{figure\}',
    _widen,
    src,
    flags=re.DOTALL,
)

# Wrap any long \texttt{...} run that contains a slash, underscore,
# or dot in a \seqsplit{...} so it can break mid-token at the column
# edge.  Without this, paths like
# experiments/hypermapping_full_comparison.py refuse to hyphenate
# and overflow the column (the 35pt overflow we saw on p.77).  We
# only rewrite \texttt content that is (a) >=20 chars long and (b)
# contains at least one of `/`, `_`, or `.` -- short identifiers do
# not need breaking and we want to leave them alone.  We also skip
# any \texttt that is part of a \texorpdfstring (i.e. inside a
# section heading), because \seqsplit is fragile in moving-argument
# contexts like the TOC.
def _split_long_texttt(m):
    body = m.group(1)
    if len(body) < 20:
        return m.group(0)
    if not any(c in body for c in '/_.'):
        return m.group(0)
    # Look backwards in the source from the match start: if we see
    # an unbalanced \texorpdfstring{ before the match, we're inside
    # a section heading and should not seqsplit.
    start = m.start()
    window = src[max(0, start - 400):start]
    last_tps = window.rfind(r'\texorpdfstring{')
    if last_tps != -1:
        # Count braces from \texorpdfstring{ up to our match start;
        # if still inside braces, skip.
        tail = window[last_tps + len(r'\texorpdfstring{'):]
        depth = 1
        for c in tail:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
        if depth > 0:
            return m.group(0)
    # \seqsplit re-typesets each character allowing a break after
    # it, so the tokenisation of LaTeX-special chars inside \texttt
    # is preserved.  We keep the surrounding \texttt so the font
    # stays monospace.
    return r'\texttt{\seqsplit{' + body + r'}}'

src = re.sub(
    r'\\texttt\{([^{}]+)\}',
    _split_long_texttt,
    src,
)

# Insert \FloatBarrier before each top-level \section (which in our
# pandoc setup corresponds to a chapter heading -- "Chapter 5",
# "Appendix B" etc.).  This stops floats from migrating across
# chapter boundaries (e.g. a Chapter 9 figure rising up into the
# Chapter 7 area) without imposing per-subsection barriers that
# would isolate wide figures on their own pages.
src = re.sub(
    r'(\n)(\\section\{[^}]+\}\\label\{[^}]*\})',
    r'\1\\FloatBarrier\n\2',
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
