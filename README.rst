.. image:: https://travis-ci.org/openopt/copt.svg?branch=master
    :target: https://travis-ci.org/openopt/copt
.. image:: https://coveralls.io/repos/github/openopt/copt/badge.svg?branch=master
   :target: https://coveralls.io/github/openopt/copt?branch=master
.. image:: https://zenodo.org/badge/46262908.svg
   :target: https://zenodo.org/badge/latestdoi/46262908

C-OPT: composite optimization in Python
=======================================

C-OPT is an optimization library written in pure Python that follows the scipy.optimize API.


Philosophy
==========

Ø compiled code, Ø bullshit.

Let me explain this better:

- **Ø compiled code**, because we don't need to. `Numba <http://numba.pydata.org/>`_ generates machine-efficient code on the fly. This way, the code remains pure Python, which makes it both easier to develop and to deploy.
    
- **Ø bullshit**. Here you will only find methods that work out of the box. No more tweaking hyperparameters to make the method converge. Pinkie swear.

- **State of the art performance**. Only the best methods, implemented with obsessive care.


Documentation
=============

See http://copt.bianp.net
