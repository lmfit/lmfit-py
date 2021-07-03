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
from pathlib import Path
import shlex
from shutil import copy2
import subprocess


def copy_data_files(src_dir, dest_dir):
    """Copy files with datafile extension from src_dir to dest_dir."""
    data_file_extension = [".dat", ".csv", ".sav"]

    for file in src_dir.glob("*"):
        if file.suffix in data_file_extension:
            copy2(file, dest_dir)


doc_dir = Path(__file__).parent.absolute()

examples_dir = doc_dir.parent / "examples"
files = examples_dir.glob("doc[_]*.py")

examples_documentation_dir = examples_dir / "documentation"
examples_documentation_dir.mkdir(exist_ok=True)


scripts_to_run = []

(examples_documentation_dir / "README.txt").write_text(
    "Examples from the documentation\n"
    "===============================\n\n"
    "Below are all the examples that are part of the lmfit documentation."
)

for fn in files:

    script_text = fn.read_text()

    gallery_file = examples_documentation_dir / fn.name[4:]
    msg = ""  # add optional message f
    gallery_file.write_text(f'"""\n{fn.name}\n{"=" * len(fn.name)}\n\n'
                            f'{msg}\n"""\n{script_text}')

    # make sure the saved Models and ModelResult are available
    if "save" in fn.name:
        scripts_to_run.append(gallery_file)

copy_data_files(examples_dir, examples_documentation_dir)

os.chdir(examples_documentation_dir)

for script in scripts_to_run:
    subprocess.run(shlex.split(f"python {script.as_posix()}"), check=True)

os.chdir(doc_dir)

# # data files for the other Gallery examples
copy_data_files(examples_documentation_dir, doc_dir)
