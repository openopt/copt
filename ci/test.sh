#!/bin/sh
#
pip install -r requirements.txt
pip install pytest-parallel
python setup.py install
py.test --workers auto