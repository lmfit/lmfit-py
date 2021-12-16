## Contributing Code

We would love your help, either as ideas, documentation, or code. If you have a
new algorithm or want to add or fix existing code, please do! We try to follow
the Python coding style conventions (i.e., [PEP 8](https://www.python.org/dev/peps/pep-0008/))
closely. Additionally, we really want comprehensive docstrings that follow
[PEP 257](https://www.python.org/dev/peps/pep-0257/) using the
[numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard),
usable offline documentation, and good unit tests for the ``pytest`` framework. A
good contribution includes all of these. To ensure compliance with our coding
style, we make use of the [pre-commit](https://pre-commit.com/) framework to run
several *hooks* when committing code. Please follow the instructions below if
you intend to contribute to the lmfit repository:

- fork the GitHub repository
- clone your forked GitHub repository:
  ``git clone https://github.com/<your-name>/lmfit-py.git``
- install all (optional) dependencies either using ``pip`` or ``conda``:
  ``pip install lmfit[all]`` or
  ``conda install <all packages listed in setup.cfg>``
- initialize ``pre-commit`` by running ``pre-commit install`` in the lmfit directory
- create a new branch: ``git checkout -b <awesome_new_feature>``
- start coding
- install the latest version of your code using the PEP517/PEP518 way (``python -m build && pip install .``)
- make sure the test-suite passes locally: run ``pytest`` in the lmfit directory
- make sure the documentation builds locally: run ``make`` in the doc directory
- push to your fork: ``git push origin``
- open a Pull Request on [the lmfit GitHub repository](https://github.com/lmfit/lmfit-py/pulls)

If you need any additional help, please send a message to the
[mailing list](https://groups.google.com/group/lmfit-py) or use the
[GitHub discussions page](https://github.com/lmfit/lmfit-py/discussions).

## Using the Mailing List versus GitHub Issues

If you have ***questions, comments, or suggestions*** for lmfit, please use
the [mailing list](https://groups.google.com/group/lmfit-py) or
[GitHub discussions page](https://github.com/lmfit/lmfit-py/discussions).
These provide online conversations that are archived and can be searched
easily.

If you find a ***bug with the code or documentation***, please use
[GitHub Issues](https://github.com/lmfit/lmfit-py/issues) to submit a bug report.
If you have an idea for how to solve the problem and are familiar with Python
and GitHub, submitting a [Pull Request](https://github.com/lmfit/lmfit-py/pulls)
would be greatly appreciated (see above).

**If you are at all unsure whether to open an Issue, please start a
conversation on the discussions page or mailing list.**

Starting the conversation with "How do I do this?" or "Why didn't this work
the way I expected?"  instead of "This doesn't work" is preferred, and will
better help others with similar questions. No posting about fitting data is
inappropriate for the mailing list, but many questions are not Issues. We
will try our best to engage in all discussions, but we may simply close
GitHub Issues that are actually questions.

## Providing an Example with GitHub Issues

If you are reporting a bug with GitHub Issues, we do expect a small, complete,
working example that illustrates the problem. Yes, this forces you to invest
some time in writing a careful example. That is intentional. If you need to
read certain data or have code longer than a few pages, use a
[GitHub Gist](https://gist.github.com/) and provide a link in the Issue.

Please understand that the point of the example script is to be *read*.
We may not even run your example. Please do not expect that we know much
about your problem domain, or that we will read any example in enough detail
to fully understand what you are trying to do without adequate explanation.
State the problem, including what result you think you should have
gotten, and include what you got. If you get a traceback, include the
entire thing.

In addition, please include information on your operating system, Python
version and installed dependencies. You can paste the code below in your
Python shell to get this information:

```python
import sys, lmfit, numpy, scipy, asteval, uncertainties
print(f"Python: {sys.version}\n\nlmfit: {lmfit.__version__}, scipy: {scipy.__version__}, numpy: {numpy.__version__},"
	  f"asteval: {asteval.__version__}, uncertainties: {uncertainties.__version__}")
```

## Using IPython Notebooks to Show Examples

IPython Notebooks are very useful for showing code snippets and outcomes,
and are a good way to demonstrate a question or raise an issue. Please
see the above about providing examples. The notebook you provide will be
*read*, but will probably not be run.

## Secret Code for First Time Issues

If you have not done so in the past, and are going to submit a GitHub Issue,
you will need to include the phrase

```
Yes, I read the instructions and I am sure this is a GitHub Issue.
```

as the First Time Issue Code. If you do not copy and paste this in verbatim,
we will know that you did not read the instructions.
