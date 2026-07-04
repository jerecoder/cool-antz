LATEXMK ?= latexmk
PYTHON ?= python

.PHONY: all figures clean

all: figures main.pdf

figures:
	$(PYTHON) scripts/build_figures.py

main.pdf: main.tex references.bib figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
	rm -f figures/*.png figures/*.pdf figures/*.jpg
	rm -rf figures/behavior_frames
