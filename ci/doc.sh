#!/bin/sh
#
pip install -r requirements.txt
pip install -U sphinx loky joblib sphinx_copybutton memory_profiler jax jaxlib
# I need my fork of sphinx-gallery to run stuff in parallel
pip install git+https://github.com/fabianp/sphinx-gallery.git
python setup.py install
cd doc
make html
if [ $? -eq 0 ]; then
    wget https://storage.googleapis.com/tm-github-builds/build/success.svg _build/html/doc_status.svg
else
    wget https://storage.googleapis.com/tm-github-builds/build/failure.svg _build/html/doc_status.svg
fi
