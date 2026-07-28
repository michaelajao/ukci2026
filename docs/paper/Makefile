# Makefile for UKCI 2026 paper
# Usage:
#   make                 # build PDF
#   make clean           # remove build artefacts
#   make distclean       # remove PDF too
#   make watch           # rebuild on changes (requires latexmk -pvc)
#   make wordcount       # rough word count on tex sources

MAIN = main
JOBNAME = ukci2026_camera_ready
OUT = ../../output/pdf
LATEXMK = latexmk
LATEXMKFLAGS = -pdf -jobname=$(JOBNAME) -outdir=$(OUT) -interaction=nonstopmode -halt-on-error
POWERSHELL = powershell.exe -NoProfile -NonInteractive -Command

# BibTeX on MiKTeX (Windows) does not reliably honour BSTINPUTS when
# launched inside the output directory, so stage the bibliography style,
# class file, support styles, and bibliography beside the build products.
# ``clean-aux`` removes those staged copies and all intermediates after a
# successful build, leaving one canonical PDF in ``output/pdf/``.

.PHONY: all clean clean-aux distclean watch wordcount stage-assets sync-figures

# Every figure the paper cites is vendored in docs/paper/figures/ so this
# directory can be uploaded to Overleaf unchanged. These two are pipeline
# outputs owned by the repo-root figures/ directory -- re-run sync-figures
# after regenerating them. fig_pipeline_architecture.png is authored for the
# paper, has no repo-root source, and is therefore not synced.
PIPELINE_FIGURES = fig_pinn_arima_gru_ci.png fig_alloc_budget.png

all: $(OUT)/$(JOBNAME).pdf

sync-figures:
	$(POWERSHELL) "New-Item -ItemType Directory -Force -Path 'figures' | Out-Null"
	$(foreach f,$(PIPELINE_FIGURES),$(POWERSHELL) "Copy-Item -Path '../../figures/$(f)' -Destination 'figures' -Force" &&) true

stage-assets:
	$(POWERSHELL) "New-Item -ItemType Directory -Force -Path '$(OUT)' | Out-Null"
	$(POWERSHELL) "Copy-Item -Path '*.bst','*.sty','*.cls','references.bib' -Destination '$(OUT)' -Force"

$(OUT)/$(JOBNAME).pdf: $(MAIN).tex sections/*.tex references.bib stage-assets
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN).tex
	"$(MAKE)" clean-aux

clean-aux:
	$(POWERSHELL) "Remove-Item -Path '$(OUT)/*.aux','$(OUT)/*.log','$(OUT)/*.out','$(OUT)/*.toc','$(OUT)/*.bbl','$(OUT)/*.blg' -Force -ErrorAction SilentlyContinue"
	$(POWERSHELL) "Remove-Item -Path '$(OUT)/*.synctex.gz','$(OUT)/*.fdb_latexmk','$(OUT)/*.fls' -Force -ErrorAction SilentlyContinue"
	$(POWERSHELL) "Remove-Item -Path '$(OUT)/*.bst','$(OUT)/*.sty','$(OUT)/*.cls','$(OUT)/references.bib' -Force -ErrorAction SilentlyContinue"

clean: clean-aux

distclean: clean
	$(POWERSHELL) "Remove-Item -LiteralPath '$(OUT)/$(JOBNAME).pdf' -Force -ErrorAction SilentlyContinue"

watch: stage-assets
	$(LATEXMK) $(LATEXMKFLAGS) -pvc $(MAIN).tex

wordcount:
	$(POWERSHELL) "if (Get-Command texcount -ErrorAction SilentlyContinue) { texcount -inc -sum -1 '$(MAIN).tex' } else { Write-Output 'texcount not installed; install via your TeX distribution' }"
