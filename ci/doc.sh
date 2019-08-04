#!/bin/sh
#
pip install -r requirements.txt
pip install -U sphinx loky joblib sphinx_copybutton memory_profiler
# I need my fork of sphinx-gallery to run stuff in parallel
# pip install git+https://github.com/fabianp/sphinx-gallery.git
# get the standard one since mine is not working on Gcloud
pip install -U sphinx-gallery
python setup.py install
cd doc
make html-noplot
gsutil -m rsync -r -c -d _build/html/ gs://copt
