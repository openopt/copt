#!/bin/sh
#
mkdir -p _build/html/
wget https://storage.googleapis.com/tm-github-builds/build/failure.svg -o _build/html/doc_status.svg
pip install -r requirements.txt
pip install -U sphinx loky joblib sphinx_copybutton memory_profiler jax jaxlib
# I need my fork of sphinx-gallery to run stuff in parallel
pip install git+https://github.com/fabianp/sphinx-gallery.git
python setup.py install
cd doc
make html
if [ $? -eq 0 ]; then
    # set up a badge depending on the result of the build
    rm _build/html/doc_status.svg
    wget https://storage.googleapis.com/tm-github-builds/build/success.svg -o _build/html/doc_status.svg
fi
