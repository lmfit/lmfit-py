---
name: Issues
about: 'Create a bug report '
title: ''
labels: ''
assignees: ''

---

** DO NOT IGNORE **

READ THESE INSTRUCTIONS FULLY.    If you have not submitted a GitHub Issue to lmfit before, read [this](https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md) first.

***DO NOT USE GitHub Issues for questions, it is only for bug reports***

Issues here should be concerned with errors or problems in the lmfit code.  There are other places to get support and help with using lmfit or to discuss ideas about the library. 

If you **think** something is an Issue, it probably is not an Issue. If the behavior you
want to report involves a fit that runs to completion without raising an exception but
that gives a result that you think is incorrect, that is almost certainly not an Issue.

Use the [mailing list](https://groups.google.com/group/lmfit-py) or [GitHub discussions
page](https://github.com/lmfit/lmfit-py/discussions) for questions about lmfit or things
you think might be problems.  We don't feel obligated to spend our free time helping
people who do not respect our chosen work processes, so if you ignore this advice and post
a question as a GitHub Issue anyway, it is quite likely that your Issue will be closed and
not answered. If you have any doubt at all, do NOT submit an Issue.

To submit an Issue, you MUST provide ALL of the following information.  If you delete any
of these sections, your Issue may be closed. If you think one of the sections does not
apply to your Issue, state that explicitly. We will probably disagree with you and insist
that you provide that information. If we have to ask for it twice, we will expect it to be
correct and prompt.

** Description **
Provide a short description of the issue, describe the expected outcome, and give the actual result

** A Minimal, Complete, and Verifiable example **
See, for example, https://stackoverflow.com/help/mcve on how to do this 

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
If you started a discussion on the lmfit mailing list, discussion page, or Stack Overflow, please provide the relevant link(s)
