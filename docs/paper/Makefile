# Makefile for UKCI 2026 paper
# Usage:
#   make                 # build PDF
#   make clean           # remove build artefacts
#   make distclean       # remove PDF too
#   make watch           # rebuild on changes (requires latexmk -pvc)
#   make wordcount       # rough word count on tex sources

MAIN = main
OUT = out
LATEXMK = latexmk
LATEXMKFLAGS = -pdf -outdir=$(OUT) -interaction=nonstopmode -halt-on-error

# BibTeX on MiKTeX (Windows) does not reliably honour BSTINPUTS when
# launched inside ``out/`` by latexmk, so we copy the bibliography style,
# class file, and any auxiliary .sty assets into ``out/`` before each
# build. This is what fixes the historical
# ``I couldn't open style file spmpsci.bst`` error.

.PHONY: all clean distclean watch wordcount stage-assets

all: $(OUT)/$(MAIN).pdf

stage-assets:
	mkdir -p $(OUT)
	cp -u *.bst $(OUT)/ 2>/dev/null || true
	cp -u *.sty $(OUT)/ 2>/dev/null || true
	cp -u *.cls $(OUT)/ 2>/dev/null || true
	cp -u references.bib $(OUT)/ 2>/dev/null || true

$(OUT)/$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib stage-assets
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN).tex

clean:
	$(LATEXMK) -c -outdir=$(OUT)
	rm -f $(OUT)/*.aux $(OUT)/*.log $(OUT)/*.out $(OUT)/*.toc $(OUT)/*.bbl $(OUT)/*.blg
	rm -f $(OUT)/*.synctex.gz $(OUT)/*.fdb_latexmk $(OUT)/*.fls

distclean: clean
	rm -f $(OUT)/$(MAIN).pdf

watch:
	$(LATEXMK) $(LATEXMKFLAGS) -pvc $(MAIN).tex

wordcount:
	@which texcount > /dev/null 2>&1 \
		&& texcount -inc -sum -1 $(MAIN).tex \
		|| echo "texcount not installed; install via your TeX distribution"
