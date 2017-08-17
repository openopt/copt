import numpy as np
from scipy import misc
import os
from os.path import expanduser
import hashlib
import urllib.request

# .. TODO: allow to be set also from environment variable ..
DATA_DIR = os.path.join(os.path.expanduser("~"), 'copt_data')


def load_img1(n_rows=20, n_cols=20):
    """Load sample image"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    grid = np.loadtxt(
        os.path.join(dir_path, 'data', 'img1.csv'),
        delimiter=',')
    dim1 = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape((dim1, dim1))
    return misc.imresize(grid, (n_rows, n_cols))


def load_rcv1(md5_check=True):
    """
    Download and return the RCV1 dataset.

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary

    Parameters
    ----------
    md5_check: bool
        Whether to do an md5 check on the downloaded files.

    Returns
    -------
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 1 or -1.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, 'rcv1_full.binary.bz2')
    if not os.path.exists(file_path):
        print('RCV1 dataset is not present in data folder. Downloading it ...')
        url = 'http://s3-eu-west-1.amazonaws.com/copt.bianp.net/datasets/rcv1_full.binary.bz2'
        urllib.request.urlretrieve(url, file_path)
        print('Finished downloading')
    if md5_check:
        h = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        if not h == '6131cf16515e9cce08d112c880b6b817':
            print('MD5 hash do not coincide')
            print('Removing file and re-downloading')
            os.remove(file_path)
            return load_rcv1()
    from sklearn import datasets
    return datasets.load_svmlight_file(file_path)


def load_url(md5_check=True):
    """
    Download and return the URL dataset.

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url

    Parameters
    ----------
    md5_check: bool
        Whether to do an md5 check on the downloaded files.

    Returns
    -------
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 1 or -1.
    """

    home = expanduser("~")
    dir_name = os.path.join(home, 'copt_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_path = os.path.join(dir_name, 'url_combined.bz2')
    if not os.path.exists(file_path):
        print('URL dataset is not present in data folder. Downloading it ...')
        url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2'
        urllib.request.urlretrieve(url, file_path)
        print('Finished downloading')
    if md5_check:
        h = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        if not h == '83673b8f4224c81968af2fb6022ee487':
            print('MD5 hash do not coincide')
            print('Removing file and re-downloading')
            os.remove(file_path)
            return load_url()
    from sklearn import datasets
    return datasets.load_svmlight_file(file_path)


def load_covtype():
    """
    Download and return the URL dataset.

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url

    Parameters
    ----------
    md5_check: bool
        Whether to do an md5 check on the downloaded files.

    Returns
    -------
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 1 or -1.
    """

    home = expanduser("~")
    dir_name = os.path.join(home, 'copt_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_name = os.path.join(dir_name, 'covtype.libsvm.binary.scale.bz2')
    if not os.path.exists(file_name):
        print('Covtype dataset is not present in data folder. Downloading it ...')
        url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2'
        urllib.request.urlretrieve(url, file_name)
        print('Finished downloading')
    from sklearn import datasets
    return datasets.load_svmlight_file(file_name)
