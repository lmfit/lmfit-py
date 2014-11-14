REM
REM  %HOME%\.pypirc must be setup with PyPI info.
REM
SET HOME=M:\

SET PATH=C:\Python27;%PATH%
python setup.py install
python setup.py bdist_wininst --target-version=2.7 upload


SET PATH=C:\Python34;%PATH%
python setup.py install
python setup.py bdist_wininst --target-version=3.4 upload



