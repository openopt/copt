#!/bin/sh
#
pip install -r requirements.txt
pip install sphinx loky joblib sphinx_copybutton memory_profiler
# I need my fork of sphinx-gallery to run stuff in parallel
pip install git+https://github.com/fabianp/sphinx-gallery.git
python setup.py install
cd doc
make html
