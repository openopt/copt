import copt
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
    name='copt',
    description='Proximal gradient descent algorithm in Python',
    long_description=open('README.rst', encoding='utf-8').read(),
    version=copt.__version__,
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='http://pypi.python.org/pypi/copt',
    packages=['copt'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    package_data={'copt': ['data/img1.csv']},
    license='BSD'
)
