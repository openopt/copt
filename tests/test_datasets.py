from copt import datasets


def test_img1():
    for n_features in [10, 20, 50]:
        img = datasets.load_img1(n_features, 20)
        assert img.shape[0] == n_features
        assert img.shape[1] == 20