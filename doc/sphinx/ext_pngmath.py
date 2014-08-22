# sphinx extensions for pngmath
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.intersphinx',
              'numpydoc']
mathjax = 'sphinx.ext.mathjax'
pngmath = 'sphinx.ext.pngmath'

extensions.append(pngmath)
