#!/usr/bin/env sh

# shell script for building the documentation and updating GitHub Pages

cd doc
echo '# Building lmfit documentation (PDF/EPUB/HTML)'
make clean
make all
cd ../

echo '# Building tarball of documentation'
tar czf lmfit_docs.tar.gz doc/_build/html/* -C doc/_build/html .

echo "# Switching to gh-pages branch"
git checkout gh-pages

if  [ $? -ne 0 ]  ; then
  echo ' failed.'
  exit
fi

echo '# Clean-up old documentation files'
rm -rf *.html *.js
rm -rf _download _images _sources _static

echo '# Unpack new documentation files'
tar xzf lmfit_docs.tar.gz
rm -f lmfit_docs.tar.gz
rm -f .buildinfo

echo '# Commit changes to gh-pages branch'
export version=`git tag | sort | tail -1`
git add *
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -am "DOC: update documentation for ${version}" --no-verify

if  [ $? -ne 0 ]  ; then
  echo ' failed.'
  exit
fi

echo '# Please check the commit and if everything looks good, push the changes:'
echo 'for example by doing: `git push` or `git push upstream`'
