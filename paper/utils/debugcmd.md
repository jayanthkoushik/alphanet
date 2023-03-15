{% if debug == "true" %}
\newcommand{\pernote}[2]{\textcolor{#1}{#2}}
{% else %}
\newcommand{\pernote}[2]{}
{% endif %}
\newcommand{\nadine}[1]{\pernote{red}{#1}}
\newcommand{\jayanth}[1]{\pernote{green}{#1}}
\newcommand{\mike}[1]{\pernode{orange}{#1}}
\newcommand{\aarti}[1]{\pernote{blue}{#1}}
