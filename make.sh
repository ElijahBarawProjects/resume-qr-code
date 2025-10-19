# make latex resume to pdf
name=$1
rm $name.pdf
pdflatex $name.tex && rm $name.aux $name.log $name.out && rm .log