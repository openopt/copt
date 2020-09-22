from distutils.core import setup
import io
import setuptools

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name="copt",
    description="Library for composite optimization in Python",
    long_description=io.open("README.rst", encoding="utf-8").read(),
    version="0.9.0",
    author="Fabian Pedregosa",
    author_email="f@bianp.net",
    url="http://pypi.python.org/pypi/copt",
    packages=["copt"],
    install_requires=["numpy", "scipy", "tqdm", "sklearn", "six"],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    package_data={"copt": ["data/img1.csv"]},
    license="New BSD License",
)
