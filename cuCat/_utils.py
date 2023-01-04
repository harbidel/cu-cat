import collections
from typing import Any, Hashable

import cupy as np,cudf
from sklearn.utils import check_array

try:
    # Works for sklearn >= 1.0
    from sklearn.utils import parse_version  # noqa
except ImportError:
    # Works for sklearn < 1.0
    from sklearn.utils.fixes import _parse_version as parse_version  # noqa


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def check_cuml_input(X) -> np.array:
    
    for i in X.columns:
        X_=X.dropna()
        # try:
        #     X_[i]=X[i].astype(float)
        # except:
            # X_[i]=X[i]
        # if isinstance(X[i][0],str):
        
        if isinstance(X[i][0], (int, float, complex,np.float32)) and not isinstance(X[i][0], bool):
            X_[i]=X[i].astype(float)
        else:
            X_[i]=np.array_str(X[i]) #.to_cupy()
    
    return X_

def check_input(X) -> np.array:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much

    X_ = check_array(
        X, #.to_cupy(),
        dtype=None,
        ensure_2d=True,
        force_all_finite=False,
    )
    # If the array contains both NaNs and strings, convert to object type
    if X_.dtype.kind in {"U", "S"}:  # contains strings
        if np.any(X_ == "nan"):  # missing value converted to string
            return check_array(
                np.array(X, dtype=object),
                # X=X, #.to_cupy(),
                dtype=None,
                ensure_2d=True,
                force_all_finite=False,
            )

    return X_
