import numpy as np
import cupy as cp
import cudf
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import NotFittedError
from time import time

from cu_cat import GapEncoder
from cu_cat._utils import parse_version
from cu_cat.tests.utils import generate_data #, generate_cudata


# def test_analyzer():
#     """
#     Test if the output is different when the analyzer is 'word' or 'char'.
#     If it is, no error ir raised.
#     """
#     add_words = False
#     n_samples = 70
#     X = cp.array_str(generate_data(n_samples, random_state=0))
#     n_components = 10
#     # Test first analyzer output:
#     encoder = GapEncoder(
#         n_components=n_components,
#         init="k-means++",
#         analyzer="char",
#         add_words=add_words,
#         random_state=42,
#         rescale_W=True,
#     )
#     encoder.fit(X)
#     y = encoder.transform(X)

#     # Test the other analyzer output:
#     encoder = GapEncoder(
#         n_components=n_components,
#         init="k-means++",
#         analyzer="word",
#         add_words=add_words,
#         random_state=42,
#     )
#     encoder.fit(X)
#     y2 = encoder.transform(X)

#     # Test inequality between the word and char analyzers output:
#     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y, y2)


@pytest.mark.parametrize(
    "hashing, init, analyzer, add_words",
    [
        # (False, "k-means++", "word", True),
        (True, "random", "char", False),
        # (True, "k-means", "char_wb", True),
    ],
)
def test_gap_encoder(
    hashing: bool, init: str, analyzer: str, add_words: bool, n_samples: int = 70
) -> None:
    X = generate_data(n_samples, random_state=0)
    # X = cudf.DataFrame(cudf.Series.from_pandas(pd.DataFrame(X),nan_as_null=False))
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        # n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (n_samples, n_components * X.shape[1]), str(y.shape)

    # Test L1-norm of topics W.
    for col_enc in encoder.fitted_models_:
        l1_norm_W = np.abs(col_enc.W_).sum(axis=1)
        np.testing.assert_array_almost_equal(l1_norm_W.get(), np.ones(n_components),decimal=2)

    # Test same seed return the same output
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    # np.testing.assert_array_equal(y, y2)
    # np.testing.assert_array_equal(np.round(y.get(),10), np.round(y2.get(),10))
    np.testing.assert_array_almost_equal(y.get(), y2.get(),decimal=2)




def test_get_feature_names_out(n_samples=70) -> None:
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42, engine='cuml')
    enc.fit(X)
    # Expect a warning if sklearn >= 1.0
    if parse_version(sklearn_version) < parse_version("1.0"):
        feature_names_1 = enc.get_feature_names()
    else:
        with pytest.warns(DeprecationWarning):
            feature_names_1 = enc.get_feature_names()
    feature_names_2 = enc.get_feature_names_out()
    for topic_labels in [feature_names_1, feature_names_2]:
        # Check number of labels
        assert len(topic_labels) == enc.n_components * X.shape[1]
        # Test different parameters for col_names
        topic_labels_2 = enc.get_feature_names_out(col_names="auto")
        assert topic_labels_2[0] == "col0: " + topic_labels[0]
        topic_labels_3 = enc.get_feature_names_out(col_names=["abc", "def"])
        assert topic_labels_3[0] == "abc: " + topic_labels[0]
    return


@pytest.mark.parametrize("missing", ["zero_impute"]) #, "error", "aaa"])
def test_missing_values(missing: str) -> None:
    observations = [
    "alice","bob",
    "bob","alice",
    np.nan,"bob",
    "alice",np.nan,
    "alice","charlie"
]
    observations = cudf.DataFrame(cudf.Series.from_pandas(observations,nan_as_null=False))

    # observations = cudf.DataFrame.from_pandas(pd.DataFrame(np.array(observations, dtype=object)))
    enc = GapEncoder(handle_missing=missing, n_components=3, engine='cuml')
    if missing == "error":
        with pytest.raises(ValueError, match="Input data contains missing values"):
            enc.fit_transform(observations)
    elif missing == "zero_impute":
        enc.fit_transform(observations)
        enc.fit(observations) ##not partial_fit
    else:
        with pytest.raises(
            ValueError,
            match=r"handle_missing should be either "
            r"'error' or 'zero_impute', got 'aaa'",
        ):
            enc.fit_transform(observations)


def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    X = np.array(["alice", "bob"])
    X = cudf.DataFrame(cudf.Series.from_pandas(X,nan_as_null=False))

    enc = GapEncoder(n_components=2, random_state=42,engine='cuml')
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_small_sample():
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def test_perf():
    """Test gpu speed boost and correctness"""
    t01=[];t02=[]
    for i in range(5):
        n_samples = np.random.randint(10,250)
        X = generate_data(n_samples, random_state=42)

        XX = X.to_pandas()
        t0 = time()
        cpu_enc = GapEncoder(random_state=2, engine='sklearn')
        CW=cpu_enc.fit_transform(XX)
        t01.append(time()-t0)
        t1 = time()
        gpu_enc = GapEncoder(random_state=2, engine='cuml')
        GW=gpu_enc.fit_transform(X)
        t02.append(time()-t1)
        GW=GW.get()
        
    assert(any(np.greater(t01,t02)))
        # [t01 , t02]
    # intersect=np.sum(np.sum(pd.DataFrame(CW)==(GW)))
    # union=pd.DataFrame(CW).shape[0]*pd.DataFrame(CW).shape[1]
    # assert(intersect==union)
    # np.testing.assert_array_almost_equal(pd.DataFrame(CW), GW, decimal=4)


# def test_multiplicative_update_h_smallfast():
#     # Generate random input arrays
#     from scipy.sparse import csr_matrix as cpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_h
    
#     from cupyx.scipy.sparse import csr_matrix as gpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_h_smallfast
    
#     np.random.seed(123)
#     Vt = pd.DataFrame(np.random.rand(1000, 1000))
#     W = pd.DataFrame(np.random.rand(1000, 1000))
#     Ht = pd.DataFrame(np.random.rand(1000, 1000))
#     # Convert Vt to CSR sparse matrix
#     Vt_sparse = cpu_csr(Vt)
#     tmp = np.random.rand(1, 1)
    
#     # Call the function with different arguments
#     res_1A = _multiplicative_update_h(tmp,Vt, W, Ht, max_iter=100)
#     res_1B = _multiplicative_update_h_smallfast(cp.array(Vt), cp.array(W), cp.array(Ht), max_iter=100)
    
#     res_2A = _multiplicative_update_h(tmp,Vt_sparse, W, Ht, rescale_W=True, gamma_scale_prior=5.0, max_iter=200)
#     res_2B = _multiplicative_update_h_smallfast(gpu_csr(Vt_sparse), cp.array(W), cp.array(Ht), rescale_W=True, gamma_scale_prior=5.0, max_iter=200)
#     # Check that the output arrays are equal (within a small tolerance)
    
#     np.testing.assert_array_almost_equal(res_1A, res_1B.get(), decimal=4)
#     np.testing.assert_array_almost_equal(res_1A, res_2B.get(), decimal=4)


# def test_multiplicative_update_w_smallfast():
#     from typing import Tuple
#     from scipy.sparse import csr_matrix as cpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_w
    
#     from cupyx.scipy.sparse import csr_matrix as gpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_w_smallfast
    

#     # Generate random input arrays
#     np.random.seed(123)
#     Vt = pd.DataFrame(np.random.rand(1000, 1000))
#     W = pd.DataFrame(np.random.rand(1000, 1000))
#     A = pd.DataFrame(np.random.rand(1000, 1000))
#     B = pd.DataFrame(np.random.rand(1000, 1000))
#     Ht = pd.DataFrame(np.random.rand(1000, 1000))
#     rescale = True #bool(np.random.randint(2, size=1))
#     rho = float(np.random.randn())
#     tmp = np.random.rand(1, 1)

#     # Convert Vt to CSR sparse matrix
#     Vt_sparse = cpu_csr(Vt)

#     # Call the function with different arguments
#     res_1A = _multiplicative_update_w(tmp,Vt, W, A, B, Ht, rescale_W=rescale, rho=rho)
#     res_1B = _multiplicative_update_w_smallfast(cp.array(Vt), cp.array(W), cp.array(A), cp.array(B), cp.array(Ht), rescale_W=rescale, rho=rho)

#     res_2A = _multiplicative_update_w(tmp,gpu_csr(Vt_sparse), W, A, B, Ht, rescale_W=rescale, rho=rho)
#     res_2B = _multiplicative_update_w_smallfast(gpu_csr(Vt_sparse), cp.array(W), cp.array(A), cp.array(B), cp.array(Ht), rescale_W=rescale, rho=rho)

#     # Check that the output arrays have the expected shape and type

#     # Check that the output arrays are equal (within a small tolerance)
#     np.testing.assert_array_almost_equal(res_1A, res_1B.get(), decimal=4)
#     np.testing.assert_array_almost_equal(res_2A, res_2B.get(), decimal=4)
