---
name: Issues
about: 'Create a bug report '
title: ''
labels: ''
assignees: ''

---

** DO NOT IGNORE **

Please Read These Instructions Fully
If you have not submitted a GitHub Issue to lmfit before,
read [this](https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md) first.

***DO NOT USE GitHub Issues for questions.  It is used for (and only for) bug reports***

Issues here should be concerned with errors or problems in the lmfit
code.  There are other places to get support and help with using lmfit
or to discuss ideas about the library.  Use the [mailing
list](https://groups.google.com/group/lmfit-py) or [GitHub discussions
page](https://github.com/lmfit/lmfit-py/discussions) for questions
about lmfit or things you think might be problems.


If you **think** something is an Issue, it probably is not an
Issue. If the behavior you want to report involves a fit that runs to
completion without raising an exception but that gives a result that
you think is incorrect, that is almost certainly NOT an Issue, though
it may be worth discussing.  We do not feel obligated to spend our
free time helping people who do not respect our chosen work processes,
so if you ignore this advice and start with an Issue anyway, it is
quite likely that your Issue will be closed and not answered. If you
have any doubt at all, start a Discussion rather than an Issue.

To submit an Issue, you MUST provide ALL of the following information.
If you delete any of these sections, your Issue may be closed. If you
think one of the sections does not apply to your Issue, explicitly
state that.  We will probably disagree with you and have to ask you to
provide that information. If we have to ask for it twice, we will
expect it to be correct and prompt, and may not be willing to help.

** Description **
Provide a short description of the issue, describe the expected
outcome, and give the actual result, including fit report if
available.

** A Minimal, Complete, and Verifiable example **
See https://stackoverflow.com/help/mcve on how to do
this.  If we cannot run your code, it is not "complete".

** Fit report: **
Paste the *full* fit report here

** Error message: **
If any, paste the *full* error message inside a code block (starting from line Traceback)


** Version information: **
Generate version information with this command in the Python shell and copy the output here:
```
import sys, lmfit, numpy, scipy, asteval, uncertainties
print(f"Python: {sys.version}\n\nlmfit: {lmfit.__version__}, scipy: {scipy.__version__}, numpy: {numpy.__version__},"
      f"asteval: {asteval.__version__}, uncertainties: {uncertainties.__version__}")
```

** Link(s): **

If there is a related discussion on the lmfit mailing list or
Discussion page, please provide the relevant link(s).  If there is not
a discussion on the lmfit mailing list or Discussion page, explain why
not.
