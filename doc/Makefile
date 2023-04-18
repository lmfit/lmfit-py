# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXBUILD   = sphinx-build
SPHINX_OPTS = -W
SPHINX_DEBUGOPTS = --keep-going -n
BUILDDIR      = _build

# Internal variables.
PAPER         =
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
SPHINX_OUTPUT   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER))
ALLSPHINXOPTS   = $(SPHINX_OUTPUT) $(SPHINXOPTS) .

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest latexpdf htmlzip
.PHONY: all pdf gallery debug

html:  gallery
	cp sphinx/ext_mathjax.py extensions.py
	$(SPHINXBUILD) -b html $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/html
	@echo
	@echo "html build finished: $(BUILDDIR)/html."

debug:  gallery
	cp sphinx/ext_mathjax.py extensions.py
	$(SPHINXBUILD) -b html $(SPHINX_OUTPUT) $(SPHINX_DEBUGOPTS) . $(BUILDDIR)/html
	./filter_spurious_link_from_html.py
	@echo
	@echo "html build finished: $(BUILDDIR)/html."

gallery: examples/index.rst

examples/index.rst:
	./doc_examples_to_gallery.py

htmlzip: html
	cp sphinx/ext_mathjax.py extensions.py
	$(SPHINXBUILD) -b html $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/lmfit_doc
	./filter_spurious_link_from_html.py
	cd $(BUILDDIR) && zip -pur html/lmfit_doc.zip lmfit_doc

epub: gallery
	cp sphinx/ext_imgmath.py extensions.py
	$(SPHINXBUILD) -b epub  $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/epub
	mkdir -p $(BUILDDIR)/html
	cp -pr $(BUILDDIR)/epub/*.epub $(BUILDDIR)/html/.

pdf: latex
	cp sphinx/ext_imgmath.py extensions.py
	cd $(BUILDDIR)/latex && make all-pdf
	mkdir -p $(BUILDDIR)/html
	cp -pr $(BUILDDIR)/latex/lmfit.pdf $(BUILDDIR)/html/.

all: html htmlzip epub pdf

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf $(BUILDDIR)
	-rm -f extensions.py
	-rm -f *.dat *.sav *.csv
	-rm -rf examples/*
	-rm -rf ../examples/documentation
	-rm -rf __pycache__

dirhtml:  gallery
	$(SPHINXBUILD) -b dirhtml $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle:  gallery
	$(SPHINXBUILD) -b pickle $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:  gallery
	$(SPHINXBUILD) -b json $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:  gallery
	$(SPHINXBUILD) -b htmlhelp $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

latex: gallery
	cp sphinx/ext_imgmath.py extensions.py
	$(SPHINXBUILD) -b latex $(SPHINX_OUTPUT) $(SPHINX_OPTS) . _build/latex
	@echo
	@echo "Build finished; the LaTeX files are in _build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

latexpdf:
	cp sphinx/ext_imgmath.py extensions.py
	$(SPHINXBUILD) -b latex $(SPHINX_OUTPUT) $(SPHINX_OPTS) . _build/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C _build/latex all-pdf
	@echo "pdflatex finished; the PDF files are in _build/latex."

changes:
	$(SPHINXBUILD) -b changes $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(SPHINX_OUTPUT) $(SPHINX_OPTS) . $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."
