import numpy as np
from scipy import misc
import os
import urllib.request


def load_img1(n_rows=20, n_cols=20):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    grid = np.loadtxt(
        os.path.join(dir_path, 'data', 'img1.csv'),
        delimiter=',')
    dim1 = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape((dim1, dim1))
    return misc.imresize(grid, (n_rows, n_cols))


def load_rcv1():
    from os.path import expanduser
    home = expanduser("~")
    dir_name = os.path.join(home, 'copt_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_name = os.path.join(dir_name, 'rcv1_full.binary')
    if not os.path.exists(file_name):
        print('Downloading RCV1 dataset ...')
        url = 'http://s3-eu-west-1.amazonaws.com/copt.bianp.net/datasets/rcv1_full.binary'
        urllib.request.urlretrieve(url, file_name)
        print('Finished downloading')
    from sklearn import datasets
    return datasets.load_svmlight_file(file_name)
