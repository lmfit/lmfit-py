#! /usr/bin/env python

"""
Process the examples in the documentation for inclusion in the Gallery:

- create a "documentation" directory within "examples"
- add a README.txt file
- copy the examples from the documentation, bu remove the "doc_" from the
   filename
- add the required docstring to the files for proper rendering
- copy the data files

"""
import os

basedir = os.getcwd()

examples_dir = os.path.join(os.getcwd(), '../examples')
files = [fn for fn in os.listdir(examples_dir) if fn.startswith('doc_')]

examples_documentation_dir = os.path.join(examples_dir, 'documentation')
os.makedirs(examples_documentation_dir, exist_ok=True)

with open(os.path.join(examples_documentation_dir, 'README.txt'), 'w') as out:
    out.write("Examples from the documentation\n")
    out.write("===============================\n\n")
    out.write("Below are all the examples that are part of the lmfit documentation.")

for fn in files:
    gallery_file = os.path.join(examples_documentation_dir, fn[4:])
    with open(gallery_file, 'w') as out:
        if fn == 'doc_model_loadmodel.py':
            msg = ('This example *does* actually work, but for some reason the '
                   'conversion using sphinx-gallery fails....')
            out.write('"""\n{}\n{}\n\n{}\n\n"""\n'.format(fn, "="*len(fn), msg))
        else:
            out.write('"""\n{}\n{}\n\n"""\n'.format(fn, "="*len(fn)))
    os.system('cat {} >> {}'.format(os.path.join(examples_dir, fn), gallery_file))

    # make sure the saved Models and ModelResult are available
    if 'save' in fn:
        os.chdir(examples_dir)
        os.system('python {}'.format(fn))
        os.chdir(basedir)

os.system('cp {}/*.dat {}'.format(examples_dir, examples_documentation_dir))
os.system('cp {}/*.csv {}'.format(examples_dir, examples_documentation_dir))
os.system('cp {}/*.sav {}'.format(examples_dir, examples_documentation_dir))

# data files for the other Gallery examples
os.system('cp {}/*.dat .'.format(examples_dir))
os.system('cp {}/*.csv .'.format(examples_dir))
os.system('cp {}/*.sav .'.format(examples_dir))
