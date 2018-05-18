## Contributing Code

We'd love your help, either as ideas, documentation, or code.  If you have a
new module or want to add or fix existing code, please do.  We try to follow
Python's PEP-8 closely.  We really want good, numpy-doc style docstrings,
usable off-line documentation, and good unit tests.  A good contribution
includes all of these.

## Using the Mailing List versus GitHub Issues

If you have ***questions, comments, or suggestions*** for lmfit, please use the
[mailing list](https://groups.google.com/group/lmfit-py).  This provides an
on-line conversation that is archived and can be searched easily.

If you find a ***bug with the code or documentation***, please use
[GitHub Issues](https://github.com/lmfit/lmfit-py/issues) to submit a bug report.
If you have an idea for how to solve the problem and are familiar with Python
and GitHub, submitting a [Pull Request](https://github.com/lmfit/lmfit-py/pulls)
would be greatly appreciated.

**If you are at all unsure whether to use the mailing list or open an Issue,
please start a conversation on the mailing list.**

Starting the conversation with "How do I do this?" or "Why didn't this work?"
instead of "This doesn't work" is preferred, and will better help others with
similar questions.  No posting about fitting data is inappropriate for the
mailing list, but many questions are not Issues.  We will try our best toengage
in all discussions, but we may simply close GitHub Issues that are actually
questions.

## Providing an Example with GitHub Issues

If you are reporting a bug with GitHub Issues, we do expect a small, complete,
working example that illustrates the problem.  Yes, this forces you to
invest some time in writing a careful example.  That is intentional.   If
you need to read certain data or have code longer than a few pages, use a
[GitHub Gist](https://gist.github.com/) and provide a link in the Issue.

Please understand that the point of the example script is to be *read*.
We may not even run your example.  Please do not expect that we know much
about your problem domain, or that we will read any example in enough detail to
fully understand what you're trying to do without adequate explanation.
State the problem, including what result you think you should have
gotten, and include what you got.  If you get a traceback, include the
entire thing.

In addition, please include information on your operating system, Python
version and installed dependencies. You can paste the code below in your
Python shell to get this information:

    import sys, lmfit, numpy, scipy, asteval, uncertainties, six
    print('Python: {}\n\nlmfit: {}, scipy: {}, numpy: {}, asteval: {}, uncertainties: {}, six: {}'\
          .format(sys.version, lmfit.__version__, scipy.__version__, numpy.__version__, \
          asteval.__version__, uncertainties.__version__, six.__version__))

## Using IPython Notebooks to show examples

IPython Notebooks are very useful for showing code snippets and outcomes,
and are a very good way to demonstrate a question or raise an issue.  Please
see the above about providing examples.  The notebook you provide will be
*read*, but will probably not be run.
