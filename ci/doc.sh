#!/bin/sh
#
pip install -r requirements.txt
python setup.py install
cd doc
make html