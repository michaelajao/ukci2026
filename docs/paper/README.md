# UKCI 2026 Paper - LaTeX Source

This directory contains the working LaTeX source for the UKCI 2026
submission. It is the manuscript we edit, build, and eventually submit.

`docs/ukci_springer_template/` is different: it is the original UKCI/Springer
SVProc template bundle downloaded from the UKCI 2026 website. Keep that folder
as the reference copy. This manuscript folder uses a minimal copied subset of
that template: `svproc.cls`, `aliascnt.sty`, `remreset.sty`, and `spmpsci.bst`.

## Build

```bash
make            # produces out/main.pdf
make clean      # remove auxiliary files
make distclean  # remove auxiliary files and PDF
make watch      # rebuild on every save (latexmk -pvc)
make wordcount  # rough word count via texcount
```

Or compile manually:

```bash
latexmk -pdf -outdir=out main.tex
```

## File Structure

```text
docs/paper/
|-- main.tex                 # Top-level paper source
|-- out/                     # Generated LaTeX artifacts and PDF
|-- references.bib           # Bibliography
|-- title.txt                # UKCI CMT metadata
|-- authors.txt              # UKCI CMT metadata
|-- svproc.cls               # Springer proceedings class copied from docs/ukci_springer_template
|-- aliascnt.sty             # Springer support file copied from docs/ukci_springer_template
|-- remreset.sty             # Springer support file copied from docs/ukci_springer_template
|-- spmpsci.bst              # Springer bibliography style copied from docs/ukci_springer_template
|-- sections/
|   |-- 01_introduction.tex   # related work is folded in here
|   |-- 02_related_work.tex    # NOT \input by main.tex (kept for the journal version)
|   |-- 03_forecasting.tex
|   |-- 04_optimisation.tex
|   |-- 05_case_study.tex
|   |-- 06_results.tex         # discussion is folded in here
|   `-- 08_conclusion.tex
`-- figures/
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

The UKCI upload should be a ZIP containing this paper source, the final PDF
from `out/`, `title.txt`, `authors.txt`, and the copied Springer files needed
to compile the paper. Do not upload `docs/files_paper.zip` or the nested
`ukci2026_project.tar.gz`.
