#!/bin/sh
#
pip install -r requirements.txt
pip install pytest-parallel  # run tests in parallel
python setup.py install
py.test --workers auto