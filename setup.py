import gdprox
from distutils.core import setup
import setuptools

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name='gdprox',
    description='',
    long_description=open('README.rst').read(),
    version=gdprox.__version__,
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='http://pypi.python.org/pypi/gdprox',
    py_modules=['gdprox'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    license='BSD'
)
