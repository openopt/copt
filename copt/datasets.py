import numpy as np
from scipy import misc
import os


def load_img1(n_rows=20, n_cols=20):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    grid = np.loadtxt(
        os.path.join(dir_path, 'data', 'img1.csv'),
        delimiter=',')
    dim1 = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape((dim1, dim1))
    return misc.imresize(grid, (n_rows, n_cols))
