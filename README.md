# pyseed-simple
Template repository for simple Python projects.

## Usage
Use the GitHub interface to create a new repository with the template. Then, follow the steps below to initialize the project.

1. Replace templates (all template names are suffixed with `_TEMPLATE`).
   + Rename `mainpackage_TEMPLATE` folder.
   + Replace `year_TEMPLATE` and `author_TEMPLATE` in file `LICENSE`.
   + Update `pyproject.toml`:
     + Replace `project_TEMPLATE` in `name = ...` with the project name.
     + Replace `description_TEMPLATE` with the project description.
     + Replace `author_TEMPLATE` and `email_TEMPLATE` in `authors = ...`, and add additional authors if present.

2. Set up environment.
   + Ensure the following are installed:
     + [python >= v3.9](https://www.python.org/downloads/).
     + [poetry >= v1.0](https://python-poetry.org/docs/#installation).
   + Create Python virtual env. If the minimum version needed by the project differs from the specification inside `pyproject.toml`, update it.
   + Add dependencies to `pyproject.toml`.
   + If working with jupyter notebooks, add `extras = ["jupyter"]` to the dependency specification for `black` inside `pyproject.toml`.
   + Install basic dependencies: `poetry install`.
   + Install git hooks: `pre-commit install -t pre-commit -t commit-msg`.


### Notes

+ Use [Google's style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings, enclosing inline literals in `, and code blocks in ```.
+ Committing will trigger the pre-commit hooks, which will, among other things, lint and format the code. If any of the hooks fails, the commit will fail--fix the issues, `git add` the modified files, and commit again.
