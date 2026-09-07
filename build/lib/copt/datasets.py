# python3
import hashlib
import os
import urllib
import urllib.request
import tarfile

import numpy as np
from scipy import misc
from scipy import sparse

try:
    from tensorflow.compat.v1.io import gfile

    HAS_TF = True
except ImportError:
    HAS_TF = False

DATA_DIR = os.environ.get(
    "COPT_DATA_DIR", os.path.join(os.path.expanduser("~"), "copt_data")
)


def load_img1(n_rows=20, n_cols=20):
    """Load sample image."""
    from PIL import Image

    dir_path = os.path.dirname(os.path.realpath(__file__))
    grid = np.loadtxt(os.path.join(dir_path, "data", "img1.csv"), delimiter=",")
    dim1 = int(np.sqrt(grid.shape[0]))
    grid = grid.reshape((dim1, dim1))
    img = Image.fromarray(grid).resize((n_rows, n_cols))
    return np.array(img)


def _load_dataset(name, subset, data_dir):
    """Low level driver to download and return dataset"""

    if HAS_TF:
        file_exists = gfile.exists
        makedirs = gfile.makedirs
        file_loader = gfile.GFile
    else:
        file_exists = os.path.exists
        makedirs = os.makedirs
        file_loader = open
    dataset_dir = os.path.join(data_dir, name)
    files_train = (
        "X_train.data.npy",
        "X_train.indices.npy",
        "X_train.indptr.npy",
        "y_train.npy",
    )
    if not np.all(
        [file_exists(os.path.join(dataset_dir, fname)) for fname in files_train]
    ):
        makedirs(dataset_dir)
        print(
            "%s dataset is not present in the folder %s. Downloading it ..."
            % (name, dataset_dir)
        )
        url = "https://storage.googleapis.com/copt-doc/datasets/%s.tar.gz" % name
        local_filename, _ = urllib.request.urlretrieve(url)
        print("Finished downloading")

        tar = tarfile.open(local_filename)
        for member in tar.getmembers():
            f_orig = tar.extractfile(member)
            if f_orig is None:
                continue

            print("Extracting data to %s" % os.path.join(data_dir, member.name))
            f_dest = file_loader(os.path.join(data_dir, member.name), "wb")
            chunk = 5000
            while True:
                data = f_orig.read(chunk)
                if not data:
                    break
                f_dest.write(data)
            f_dest.close()
            f_orig.close()

    tmp_train = []
    for fname in files_train:
        with file_loader(os.path.join(dataset_dir, fname), "rb") as f:
            tmp_train.append(np.load(f))

    data_train = sparse.csr_matrix((tmp_train[0], tmp_train[1], tmp_train[2]))
    target_train = tmp_train[3]

    if subset == "train":
        retval = (data_train, target_train)
    else:
        tmp_test = []
        for fname in (
            "X_test.data.npy",
            "X_test.indices.npy",
            "X_test.indptr.npy",
            "y_test.npy",
        ):
            with file_loader(os.path.join(dataset_dir, fname), "rb") as f:
                tmp_test.append(np.load(f))

        data_test = sparse.csr_matrix((tmp_test[0], tmp_test[1], tmp_test[2]))
        target_test = tmp_test[3]

        if subset == "test":
            retval = (data_test, target_test)
        elif subset == "full":
            data_full = sparse.vstack((data_train, data_test))
            target_full = np.concatenate((target_train, target_test))
            retval = (data_full, target_full)
        else:
            f.close()
            raise ValueError(
                "subset '%s' not implemented, must be one of ('train', 'test', 'full')."
                % subset
            )
    return retval


def load_madelon(subset="full", data_dir=DATA_DIR):
    """Download and return the madelon dataset.

    Properties:
        n_samples: 2600
        n_features: 500

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#madelon


    Args:
        md5_check: bool
            Whether to do an md5 check on the downloaded files.

        subset: string
            Can be one of 'full' for full dataset, 'train' for only the train set
            or 'test' for only the test set.

        standardize: boolean
            If True, each feature will have zero mean and unit variance.


    Returns:
        data: scipy.sparse CSR
            Return data as CSR sparse matrix of shape=(2600, 500).

        target: array of shape 2600
            Labels, only takes values 0 or 1.

    Examples:
        * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_sparse_benchmark.py`
        * :ref:`sphx_glr_auto_examples_frank_wolfe_plot_vertex_overlap.py`
    """
    return _load_dataset("madelon", subset, data_dir)


def load_rcv1(subset="full", data_dir=DATA_DIR):
    """Download and return the RCV1 dataset.

    Properties:
        n_samples: 697641
        n_features: 47236
        density: 0.1% of nonzero coefficienets in train set

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary

    Args:
        subset: string
        Can be one of 'full' for full dataset, 'train' for only the train set
        or 'test' for only the test set.

        data_dir: string
        Directory from which to read the data. Defaults to $HOME/copt_data/

    Returns:
        X : scipy.sparse CSR matrix

        y: numpy array
        Labels, only takes values 0 or 1.
    """
    return _load_dataset("rcv1", subset, data_dir)


def load_url(md5_check=True):
    """Download and return the URL dataset.

    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#url

    Args:
        md5_check: bool
            Whether to do an md5 check on the downloaded files.

    Returns:
        X : scipy.sparse CSR matrix
        y: numpy array
            Labels, only takes values 0 or 1.
    """
    from sklearn import datasets  # lazy import
    import bz2

    file_path = os.path.join(DATA_DIR, "url_combined.bz2")
    data_path = os.path.join(DATA_DIR, "url_combined.data.npy")
    data_indices = os.path.join(DATA_DIR, "url_combined.indices.npy")
    data_indptr = os.path.join(DATA_DIR, "url_combined.indptr.npy")
    data_target = os.path.join(DATA_DIR, "url_combined.target.npy")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(file_path):
        print("URL dataset is not present in data folder. Downloading it ...")
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2"
        urllib.request.urlretrieve(url, file_path)
        print("Finished downloading")
        if md5_check:
            h = hashlib.md5(open(file_path, "rb").read()).hexdigest()
            if not h == "83673b8f4224c81968af2fb6022ee487":
                print("MD5 hash do not coincide")
                print("Removing file and re-downloading")
                os.remove(file_path)
                return load_url()
        zipfile = bz2.BZ2File(file_path)
        data = zipfile.read()
        newfilepath = file_path[:-4]
        open(newfilepath, "wb").write(data)
        X, y = datasets.load_svmlight_file(newfilepath)
        np.save(data_path, X.data)
        np.save(data_indices, X.indices)
        np.save(data_indptr, X.indptr)
        np.save(data_target, y)
    X_data = np.load(data_path)
    X_indices = np.load(data_indices)
    X_indptr = np.load(data_indptr)
    X = sparse.csr_matrix((X_data, X_indices, X_indptr))
    y = np.load(data_target)
    y = ((y + 1) // 2).astype(np.int)
    return X, y


def load_covtype(data_dir=DATA_DIR):
    """Download and return the covtype dataset.

  This is the binary classification version of the dataset as found in the
  LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype


  Returns:
    X : scipy.sparse CSR matrix

    y: numpy array
        Labels, only takes values 0 or 1.
    """
    return _load_dataset("covtype", "train", data_dir)


def load_news20(data_dir=DATA_DIR):
    """Download and return the covtype dataset.

  This is the binary classification version of the dataset as found in the
  LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#news20.binary


  Returns:
    X : scipy.sparse CSR matrix

    y: numpy array
        Labels, only takes values 0 or 1.
    """
    return _load_dataset("news20", "train", data_dir)


def load_gisette(subset="full", data_dir=DATA_DIR):
    """Download and return the gisette dataset.

    Properties:
        n_samples: 6000 (train)
        n_features: 5000
        density: 22% of nonzero coefficients on train set.


    This is the binary classification version of the dataset as found in the
    LIBSVM dataset project:

        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#gisette

    :Args:
        standardize: boolean
        If True, each feature will have zero mean and unit variance.

    data_dir: string
      Directory from which to read the data. Defaults to $HOME/copt_data/


    Returns:
        data : scipy.sparse CSR matrix
        target: numpy array
            Labels, only takes values 0 or 1.
  """
    return _load_dataset("gisette", subset, data_dir)


def load_kdd10(md5_check=True):
    """Download and return the KDD10 dataset.

  This is the binary classification version of the dataset as found in the
  LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010
      (bridge to algebra)

  Args:
    md5_check: bool
        Whether to do an md5 check on the downloaded files.

  Returns:
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 0 or 1.
  """
    from sklearn import datasets  # lazy import

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, "kddb.bz2")
    if not os.path.exists(file_path):
        print("KDD10 dataset is not present in data folder. Downloading it ...")
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb.bz2"
        urllib.request.urlretrieve(url, file_path)
        print("Finished downloading")
    if md5_check:
        h = hashlib.md5(open(file_path, "rb").read()).hexdigest()
        if not h == "bc5b630fef6989c2f201039fef497e14":
            print("MD5 hash do not coincide")
            print("Removing file and re-downloading")
            os.remove(file_path)
            return load_kdd10()
    return datasets.load_svmlight_file(file_path)


def load_kdd12(md5_check=True, verbose=0):
    """Download and return the KDD12 dataset.

  This is the binary classification version of the dataset as found in the
  LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012

  Args:
    md5_check: bool
      Whether to do an md5 check on the downloaded files.

  Returns:
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 0 or 1.
  """
    from sklearn import datasets  # lazy import
    import bz2

    file_path = os.path.join(DATA_DIR, "kdd12.bz2")
    data_path = os.path.join(DATA_DIR, "kdd12.data.npy")
    data_indices = os.path.join(DATA_DIR, "kdd12.indices.npy")
    data_indptr = os.path.join(DATA_DIR, "kdd12.indptr.npy")
    data_target = os.path.join(DATA_DIR, "kdd12.target.npy")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(file_path):
        print("KDD12 dataset is not present in data folder. Downloading it ...")
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdd12.bz2"
        urllib.request.urlretrieve(url, file_path)
        print("Finished downloading")
        if md5_check:
            h = hashlib.md5(open(file_path, "rb").read()).hexdigest()
            if not h == "c6fc57735c3cf687dd182d60a7b51cda":
                print("MD5 hash do not coincide")
                print("Removing file and re-downloading")
                os.remove(file_path)
                return load_url()
        zipfile = bz2.BZ2File(file_path)
        data = zipfile.read()
        newfilepath = file_path[:-4]
        open(newfilepath, "wb").write(data)
        X, y = datasets.load_svmlight_file(newfilepath)
        np.save(data_path, X.data)
        np.save(data_indices, X.indices)
        np.save(data_indptr, X.indptr)
        np.save(data_target, y)
    X_data = np.load(data_path)
    X_indices = np.load(data_indices)
    X_indptr = np.load(data_indptr)
    X = sparse.csr_matrix((X_data, X_indices, X_indptr))
    y = np.load(data_target)
    y = ((y + 1) // 2).astype(np.int)
    return X, y


def load_criteo(md5_check=True):
    """Download and return the criteo dataset.

  This is the binary classification version of the dataset as found in the
  LIBSVM dataset project:

      https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#criteo

  Args:
    md5_check: bool
      Whether to do an md5 check on the downloaded files.

  Returns
    X : scipy.sparse CSR matrix
    y: numpy array
        Labels, only takes values 0 or 1.
  """
    from sklearn import datasets  # lazy import

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, "criteo.kaggle2014.svm.tar.gz")
    data_path = os.path.join(DATA_DIR, "criteo.kaggle2014.data.npz.npy")
    data_indices = os.path.join(DATA_DIR, "criteo.kaggle2014.indices.npy")
    data_indptr = os.path.join(DATA_DIR, "criteo.kaggle2014.indptr.npy")
    data_target = os.path.join(DATA_DIR, "criteo.kaggle2014.target.npy")
    if not os.path.exists(file_path):
        print("criteo dataset is not present in data folder. Downloading it ...")
        url = "https://s3-us-west-2.amazonaws.com/criteo-public-svm-data/criteo.kaggle2014.svm.tar.gz"
        urllib.request.urlretrieve(url, file_path)
        import tarfile

        tar = tarfile.open(file_path)
        tar.extractall(DATA_DIR)
        print("Finished downloading")
        if md5_check:
            h = hashlib.md5(open(file_path, "rb").read()).hexdigest()
            if h != "d852b491d1b3afa26c1e7b49594ffc3e":
                print("MD5 hash do not coincide")
                print("Removing file and re-downloading")
                os.remove(file_path)
                return load_criteo()
        X, y = datasets.load_svmlight_file(
            os.path.join(DATA_DIR, "criteo.kaggle2014.train.svm")
        )
        np.save(data_path, X.data)
        np.save(data_indices, X.indices)
        np.save(data_indptr, X.indptr)
        np.save(data_target, y)
        # optionally delete files
    else:
        X_data = np.load(data_path)
        X_indices = np.load(data_indices)
        X_indptr = np.load(data_indptr)
        X = sparse.csr_matrix((X_data, X_indices, X_indptr))
        y = np.load(data_target)
    return X, y
