all:
	pdflatex mlentary.tex
	pdflatex mlentary.tex
	evince mlentary.pdf

fast:
	pdflatex mlentary.tex
	evince mlentary.pdf
