# UKCI 2026 Paper - LaTeX Source

This directory contains the working LaTeX source for the UKCI 2026
submission. It is the manuscript we edit, build, and eventually submit.

`docs/ukci_springer_template/` is different: it is the original UKCI/Springer
SVProc template bundle downloaded from the UKCI 2026 website. Keep that folder
as the reference copy. This manuscript folder uses a minimal copied subset of
that template: `svproc.cls`, `aliascnt.sty`, `remreset.sty`, and `spmpsci.bst`.

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
|-- svproc.cls               # Springer proceedings class copied from docs/ukci_springer_template
|-- aliascnt.sty             # Springer support file copied from docs/ukci_springer_template
|-- remreset.sty             # Springer support file copied from docs/ukci_springer_template
|-- spmpsci.bst              # Springer bibliography style copied from docs/ukci_springer_template
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

- Confirm coauthor emails in `authors.txt`.
- Decide whether an in-paper conflict-of-interest statement is needed (the
  block in `main.tex` is currently commented out).
- Re-confirm the page count is at most 12 pages after any late edits.

## Final Submission Shape

The UKCI upload should include the paper source, the final PDF at
`output/pdf/ukci2026_camera_ready.pdf`, and the copied Springer files needed
to compile the paper.
