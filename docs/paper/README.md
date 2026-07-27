# UKCI 2026 Paper - LaTeX Source

This directory contains the working LaTeX source for the UKCI 2026
submission. It is the manuscript we edit, build, and eventually submit.

The manuscript folder includes the Springer SVProc class, bibliography style,
and support files required to compile the paper.

## Build

```bash
make            # produces ../../output/pdf/ukci2026_camera_ready.pdf
make clean      # remove auxiliary files
make distclean  # remove auxiliary files and PDF
make watch      # rebuild on every save (latexmk -pvc)
make wordcount  # rough word count via texcount
```

Use `make` so the required Springer/BibTeX assets are staged and the
intermediate files are removed automatically.

## File Structure

```text
docs/paper/
|-- main.tex                 # Top-level paper source
|-- references.bib           # Bibliography
|-- svproc.cls               # Springer proceedings class
|-- aliascnt.sty             # Springer support file
|-- remreset.sty             # Springer support file
|-- spmpsci.bst              # Springer bibliography style
|-- figures/
|   `-- fig_pipeline_architecture.png
|-- sections/
|   |-- 01_introduction.tex   # related work is folded in here
|   |-- 03_forecasting.tex
|   |-- 04_optimisation.tex
|   |-- 05_case_study.tex
|   |-- 06_results.tex         # discussion is folded in here
|   `-- 08_conclusion.tex
`-- ../../output/pdf/ukci2026_camera_ready.pdf
```

## Status

Complete draft at 12 pages: introduction (with folded related work),
forecasting methodology, optimisation formulation, case study, results (with
folded discussion), and conclusion. All result tables/figures are populated
from the pipeline outputs; no `\todo` markers or placeholders remain.

Still required before final CMT upload:

- Confirm the author names, affiliations, and emails in `main.tex`.
- Decide whether an in-paper conflict-of-interest statement is needed (the
  block in `main.tex` is currently commented out).
- Re-confirm the page count is at most 12 pages after any late edits.

## Final Submission Shape

The UKCI upload should include the paper source, figures, bibliography, the
final PDF at `output/pdf/ukci2026_camera_ready.pdf`, and the Springer files
needed to compile the paper. The title entered in CMT must match `main.tex`
and the PDF exactly.
