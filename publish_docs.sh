installdir='/www/apache/htdocs/software/python/lmfit'
docbuild='doc/_build'

cd doc 
echo '# Making docs'
make all
cd ../

echo '# Building tarball of docs'
mkdir _tmpdoc
cp -pr doc/lmfit.pdf     _tmpdoc/lmfit.pdf
cp -pr doc/_build/html/*    _tmpdoc/.
cd _tmpdoc
tar czf ../../lmfit_docs.tar.gz .
cd ..
rm -rf _tmpdoc 

# 
echo "# Switching to gh-pages branch"
git checkout gh-pages

if  [ $? -ne 0 ]  ; then 
  echo ' failed.'
  exit 
fi

tar xzf ../lmfit_docs.tar.gz .

echo "# commit changes to gh-pages branch"
git commit -am "changed docs"

if  [ $? -ne 0 ]  ; then 
  echo ' failed.'
  exit 
fi

echo "# Pushing docs to github"
git push


echo "# switch back to master branch"
git checkout master

if  [ $? -ne 0 ]  ; then 
  echo ' failed.'
  exit 
fi

# install locally
echo "# Installing docs to CARS web pages"
cp ../lmfit_docs.tar.gz $installdir/..

cd $installdir
if  [ $? -ne 0 ]  ; then 
  echo ' failed.'
  exit 
fi

tar xvzf ../lmfit_docs.tar.gz
