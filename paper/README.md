# Building paper

<!-- cSpell:ignore pandoc, crossref, shiny-mdc -->

Building the paper requires a full TeX installation,
[pandoc](https://pandoc.org/),
[pandoc-crossref](https://lierdakil.github.io/pandoc-crossref/), and
[shiny-mdc](https://pypi.org/project/shiny-mdc/). Build using the
following command, from the `paper` directory:

<!-- cSpell: disable -->
```bash
shinymdc -i main.md -o main.pdf -t stylish -m smalltabs=true,nonidan=true
```
<!-- cSpell: enable -->
