### DO NOT IGNORE ###

READ THESE INSTRUCTIONS FULLY. IF YOU DO NOT, YOUR ISSUE WILL BE CLOSED.

If you have not submitted a GitHub Issue to lmfit before, read [this](https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md) first.

***DO NOT USE GitHub Issues for questions, it is only for bugs in the lmfit code!***

Issues here are concerned with errors or problems in the lmfit code.  We use it as our bug
tracker.  There are other places to get support and help with using lmfit.

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

#### First Time Issue Code
<!-- If this is your first Issue, you will write down the Secret Code for First Time Issues from the CONTRIBUTING.md file linked to above -->

#### Description
<!-- Provide a short description of the issue, describe the expected outcome, and give the actual result -->

###### A Minimal, Complete, and Verifiable example
<!-- see, for example, https://stackoverflow.com/help/mcve on how to do this -->


###### Fit report:
<!-- paste the *full* fit report here  -->


###### Error message:
<!-- If any, paste the *full* error message inside a code block (starting from line Traceback) -->

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  ...
```

###### Version information
<!-- Generate version information with this command in the Python shell and copy the output here:
import sys, lmfit, numpy, scipy, asteval, uncertainties
print(f"Python: {sys.version}\n\nlmfit: {lmfit.__version__}, scipy: {scipy.__version__}, numpy: {numpy.__version__},"
      f"asteval: {asteval.__version__}, uncertainties: {uncertainties.__version__}")
-->

###### Link(s)
<!-- If you started a discussion on the lmfit mailing list, discussion page, or Stack Overflow, please provide the relevant link(s) -->
