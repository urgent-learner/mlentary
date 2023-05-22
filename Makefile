ml:
	pdflatex mlentary.tex
	evince mlentary.pdf

clean:
	rm *.aux
	rm *.log
	rm *.out

ml-full:
	pdflatex mlentary.tex
	pdflatex mlentary.tex
	evince mlentary.pdf
