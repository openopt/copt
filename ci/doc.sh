#!/bin/sh
#
pip install -r requirements.txt
pip install sphinx sphinx-gallery sphinx_copybutton
python setup.py install
cd doc
make html