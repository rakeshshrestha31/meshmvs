#!/usr/bin/env bash
BASE="egpaper_for_review"
pdflatex -synctex=1 -interaction=nonstopmode $BASE.tex
bibtex $BASE.aux
pdflatex -synctex=1 -interaction=nonstopmode $BASE.tex
pdflatex -synctex=1 -interaction=nonstopmode $BASE.tex
