<!--
Thank you for submitting a PR to lmfit!

To help the review process for your PR, make sure to read these
instructions and complete the appropriate boxes.
-->


** DO NOT IGNORE **


Please read [this](https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md).

Unless you are a core developer, your Pull Request must refer to an
existing Discussion or Issue, and preferrably be invited.  Unsolicited
Pull Request that do not have a related Discussion or Issue are
proposing change to the code for a perceived problem.  The core
developers and maintainers may not agree with the characterization of
the issue. Even if we do agree tha there is a problem, we may not
agree on the solution you propose.

If your Issue or Discussion does not provice a a minimal and complete
example (https://stackoverflow.com/help/mcve), it is probably not
ready for a Pull Request unless told otherwise by a core developer.

If your PR is reviewed and questions are asked or changes suggested,
these are not idle requests.  It is unlikely that Your PR will be
further considered until you address the review comments.

#### Issue or Discussion

<!--- Include a link to an open Issue, Discussion or URL to the discussion on the mailing list. -->


#### Description
<!--- Describe your changes in detail: why is it required, what problem does it solve? -->


###### Type of Changesxs
<!--- What type of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring / maintenance
- [ ] Documentation / examples


###### Tested on
<!-- Generate version information with this command in the Python shell and copy the output here:
import sys, lmfit, numpy, scipy, asteval, uncertainties
print('Python: {}\n\nlmfit: {}, scipy: {}, numpy: {}, asteval: {}, uncertainties: {}'\
      .format(sys.version, lmfit.__version__, scipy.__version__, numpy.__version__, \
      asteval.__version__, uncertainties.__version__))
-->


###### Verification <!-- (delete not applicable items) -->
Have you
<!--- Put an `x` in all the boxes that apply OR describe why you think this is unnecessary. -->
- [ ] included docstrings that follow PEP 257?
<!-- Please use your favorite linter (e.g., pydocstyle) to check your docstrings. -->
- [ ] referenced existing Issue and/or provided relevant link to mailing list?
<!-- Please don't open a new Issue if you are submitting a pull request. -->
- [ ] verified that existing tests pass locally?
<!-- Please run the test-suite locally with pytest and make sure it passes. -->
- [ ] verified that the documentation builds locally?
<!-- Please build the documentation (i.e., type make in the "doc" directory) and make sure it finishes. -->
- [ ] squashed/minimized your commits and written descriptive commit messages?
<!-- We value a clean history with useful commit messages. Ideally, you will take care of this
         before submitting a PR; otherwise you'll be asked to do so before merging. -->
- [ ] added or updated existing tests to cover the changes?
- [ ] updated the documentation and/or added an entry to the release notes (doc/whatsnew.rst)?
- [ ] added an example?
