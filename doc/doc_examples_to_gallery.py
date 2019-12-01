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
import time

basedir = os.getcwd()

examples_dir = os.path.abspath(os.path.join(basedir, '..', 'examples'))
files = [fn for fn in os.listdir(examples_dir) if fn.startswith('doc_')]

examples_documentation_dir = os.path.join(examples_dir, 'documentation')
os.makedirs(examples_documentation_dir, exist_ok=True)


scripts_to_run = []

with open(os.path.join(examples_documentation_dir, 'README.txt'), 'w') as out:
    out.write("Examples from the documentation\n")
    out.write("===============================\n\n")
    out.write("Below are all the examples that are part of the lmfit documentation.")

for fn in files:
    inp_path = os.path.join(examples_dir, fn)
    with open(inp_path, 'r') as inp:
        script_text = inp.read()

    gallery_file = os.path.join(examples_documentation_dir, fn[4:])
    with open(gallery_file, 'w') as out:
        msg = ""
        if 'model_loadmodel.py' in fn:
            msg = ('.. note:: This example *does* actually work, but running from within '
                   ' sphinx-gallery fails to find symbols saved in the save file.')
        out.write('"""\n{}\n{}\n\n{}\n"""\n'.format(fn, "="*len(fn), msg))
        out.write('##\nimport warnings\nwarnings.filterwarnings("ignore")\n##\n')
        out.write(script_text)

    # make sure the saved Models and ModelResult are available
    if 'save' in fn:
        scripts_to_run.append(fn[4:])

time.sleep(1.0)

os.system('cp {}/*.dat {}'.format(examples_dir, examples_documentation_dir))
os.system('cp {}/*.csv {}'.format(examples_dir, examples_documentation_dir))
os.system('cp {}/*.sav {}'.format(examples_dir, examples_documentation_dir))
#

os.chdir(examples_documentation_dir)

for script in scripts_to_run:
    os.system('python {}'.format(script))

os.chdir(basedir)

time.sleep(1.0)
# data files for the other Gallery examples
os.system('cp {}/*.dat .'.format(examples_documentation_dir))
os.system('cp {}/*.csv .'.format(examples_documentation_dir))
os.system('cp {}/*.sav .'.format(examples_documentation_dir))
