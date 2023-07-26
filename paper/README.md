<!-- cSpell:ignore pandoc, crossref, shiny-mdc -->

# Building paper

Building the paper requires a full TeX installation,
[pandoc](https://pandoc.org/),
[pandoc-crossref](https://lierdakil.github.io/pandoc-crossref/), and
[shiny-mdc](https://pypi.org/project/shiny-mdc/) (v1.9 or higher).
Build using the following command, from the `paper` directory:

<!-- cSpell: disable -->
```bash
shinymdc -i main.md -o main.pdf -t template.tex --pdf-engine pdflatex
```
<!-- cSpell: enable -->
