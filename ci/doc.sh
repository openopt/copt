#!/bin/sh
#
pip install -r requirements.txt
pip install sphinx
python setup.py install
cd doc
make html