# Contributor Guide

## Cloning the repo

To begin, clone the repository, then check out your own branch. You can open a pull request with your changes from there.

```bash
$ git clone https://github.com/ResidentMario/pytorch-training-performance-guide.git
$ cd pytorch-training-performance-guide
$ git checkout -B new-branch-name
```

## Setting up the environment

You will need a fresh virtual environment (I recommend using [`conda`](https://conda.io/)) with Python 3.10 and the `jupyterbook` project installed.

Instructions on setting up an environment for running or replicating the code samples and benchmarks provided in this book are still a TODO.

## Editing

[Jupyter Book](https://jupyterbook.org/) is a project under the Jupyter umbrella for building developer-facing static websites. Jupyter Book supports both Markdown and Jupyter Notebook documents, but we only use Markdown documents. All of the pages are saved individually as `md` files in the root of the `book/` directory. Images are organized by chapter in the `book/img` subdirectory.

To make changes to the book locally, edit the Markdown files, then run `jupyter-book build book`. This will write a `book/_build` directory containing the static website files. You can then view the results however you would normmally serve static pages. You can use [Express](https://www.npmjs.com/package/express), personally I use the VSCode `Live Preview` extension.

## Publishing

This book is published using GitHub Pages. Instructions:

```bash
$ git checkout gh-pages
$ rm -rf *
$ git checkout master -- book/ .gitignore .nojekyll
$ jupyter-book build book --all
$ mv book/_build/html/* ./
$ rm -rf book/
$ git add .
$ git commit -m "Publishing updated book..."
$ git push origin gh-pages
$ git checkout master
```
