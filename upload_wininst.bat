REM
REM  %HOME%\.pypirc must be setup with PyPI info.
REM
SET PATH=C:\Python26;%PATH%
python setup.py bdist_wininst --target-version=2.6 upload

SET PATH=C:\Python27;%PATH%
python setup.py bdist_wininst --target-version=2.7 upload

SET PATH=C:\Python31;%PATH%
python setup.py bdist_wininst --target-version=3.1 upload


