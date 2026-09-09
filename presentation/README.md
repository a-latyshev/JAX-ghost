# From distributed FEM to JAX GPU ghost exchange

Seven slides numbered **0–6**, in English, for a roughly ten-minute FEM/HPC talk.

- [Editable PowerPoint](jax-ghost-presentation.pptx), including speaker notes.
- [Vector PDF](jax-ghost-presentation.pdf), with the same slide content and layout.
- [Speaker notes and references](speaker-notes.md), including timings and technical qualifications.
- [Slide overview](overview.png); individual slide renders are in `previews/`.

The slides use native PowerPoint text, shapes and lines. All 571 objects are
editable; none of the diagrams is a screenshot. The PDF embeds its fonts and
contains no raster images. The typography uses DejaVu Sans and DejaVu Sans Mono;
the fonts and their redistribution license are included in `fonts/`. Install
these fonts when editing on a machine that does not already have them, to keep
PowerPoint from substituting a different typeface.

## Rebuild

The generation script uses one set of positioned slide definitions for both
formats. It does not load JAX, MPI or DOLFINx, launch simulations, or change
numerical source code. Edit content and diagrams in `build_presentation.py`.
PowerPoint-only manual edits are not propagated back into the source or PDF.

From the repository root, with Python 3.12:

```bash
python3.12 -m venv /tmp/jaxghost-presentation-env
/tmp/jaxghost-presentation-env/bin/python -m pip install -r presentation/requirements.txt
/tmp/jaxghost-presentation-env/bin/python presentation/build_presentation.py
```

With `uv` (used to prepare the isolated environment in this workspace):

```bash
uv venv --python python3.12 /tmp/jaxghost-presentation-env
UV_CACHE_DIR=/tmp/jaxghost-presentation-cache uv pip install \
  --python /tmp/jaxghost-presentation-env/bin/python \
  -r presentation/requirements.txt
/tmp/jaxghost-presentation-env/bin/python presentation/build_presentation.py
```

Set `PRESENTATION_FONT_DIR` only if using another directory containing the same
three DejaVu font files. Dependencies are pinned in `requirements.txt`.

## Content and verification

The ownership illustration is a hand-selected twelve-vertex triangular mesh,
with eight owned DoFs on rank 0 and four on rank 1. Interface ownership is split
to make forward updates visible in both directions. It illustrates IndexMap
semantics rather than claiming to reproduce a particular partitioner's output.
The sharding slide reuses these exact local layouts, padded to shape `[2, 9]`.
The four-DoF matrix is explicitly a separate algebraic toy.

The generator checks:

- The mesh ownership sets, ghost IDs, and padding counts.
- The toy matrix result `[0, -1, -2, 12]` and the two local row calculations.
- Seven slides and seven PDF pages, with notes embedded in every slide.
- Complete text agreement between shared definitions, PowerPoint and PDF.
- Text widths and page boundaries, and that the PDF contains vector content.

All seven PDF pages were rendered and visually inspected. The PowerPoint was
reopened and its structure and text checked using `python-pptx`; a native
PowerPoint/LibreOffice renderer was not available. The PDF is generated directly
from the shared definitions, not exported by a PowerPoint application.
Machine-readable results are in `validation.json`.

The forward-update and matvec semantics were reviewed against
`src/jaxghost/jaxghost.py`, `src/jaxghost/sharded.py`, and
`src/jaxghost/sharded_matrix.py`. The presentation does not change or retest those
backends. HPC examples come from `HPC-GPU/notes.md` and
`HPC-GPU/transport-investigation.md`, including the later successful UCX run and
its observed small-message host staging. References and qualifications appear
in the speaker notes; no performance or physical-transport claim is inferred.
