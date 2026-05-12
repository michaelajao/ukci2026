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
|   |-- 01_introduction.tex
|   |-- 02_related_work.tex
|   |-- 03_forecasting.tex
|   |-- 04_optimisation.tex
|   |-- 05_case_study.tex
|   |-- 06_results.tex
|   |-- 07_discussion.tex
|   `-- 08_conclusion.tex
`-- figures/
```

## Status

Drafted: introduction, related work, forecasting methodology, optimisation
formulation, case study, discussion, and conclusion.

Still required before final CMT upload:

- Fill all result placeholders in E1-E6.
- Replace all `\todo{...}` markers and dummy values.
- Add real figures for the Pareto front, sensitivity heatmap, and NHS map.
- Confirm coauthor emails in `authors.txt`.
- Confirm page count is at most 12 pages including references, figures, and
  tables.

## Final Submission Shape

The UKCI upload should be a ZIP containing this paper source, the final PDF
from `out/`, `title.txt`, `authors.txt`, and the copied Springer files needed
to compile the paper. Do not upload `docs/files_paper.zip` or the nested
`ukci2026_project.tar.gz`.
