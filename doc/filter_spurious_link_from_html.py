#! /usr/bin/env python

"""
Filter the lines that link to the Examples gallery index from the HTML output.

- it is correct for the default usage with Sphinx, but we use the "relbar" as
  a menu and then the additional link messes things up.

"""

from pathlib import Path

doc_dir = Path(__file__).parent.absolute()

examples_html_dir = doc_dir.parent / 'doc/_build/html/examples'
files = examples_html_dir.glob('*.html')

link = b'accesskey="U"'

for fn in files:
    with open(fn, 'rb') as file_in:
        lines = file_in.readlines()
        with open(fn, 'wb') as file_out:
            file_out.writelines(filter(lambda line: link not in line, lines))
