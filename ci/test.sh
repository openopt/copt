#!/bin/sh
#
pip install -r requirements.txt
pip install pytest-parallel  # run tests in parallel
python setup.py install
py.test --workers auto

# pylint
pip install pylint anybadge
pylint --rcfile=ci/pylintrc --output-format=text copt | tee pylint.txt
score=$(sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p' pylint.txt)
echo "Pylint score was $score"
anybadge --value=$score --file=pylint.svg pylint
