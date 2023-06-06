from typing import Any, Tuple

import numpy as np
import pandas as pd
import cudf
import pytest
import sklearn
from sklearn.exceptions import NotFittedError
from cuml.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from cu_cat import GapEncoder, SuperVectorizer, TableVectorizer
from cu_cat._utils import parse_version


def check_same_transformers(expected_transformers: dict, actual_transformers: list):
    # Construct the dict from the actual transformers
    actual_transformers_dict = {name: cols for name, trans, cols in actual_transformers}
    assert actual_transformers_dict == expected_transformers


def type_equality(expected_type, actual_type):
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the TableVectorizer).
    """
    if (isinstance(expected_type, object) or isinstance(expected_type, str)) and (
        isinstance(actual_type, object) or isinstance(actual_type, str)
    ):
        return True
    else:
        return expected_type == actual_type


def _get_clean_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return cudf.DataFrame(
        {
            "int": cudf.Series([15, 56, 63, 12, 44], dtype="float"),
            "float": cudf.Series([5.2, 2.4, 6.2, 10.45, 9.0], dtype="float"),
            "str1": cudf.Series(
                ["public", "private", "private", "private", "public"], dtype="str"
            ),
            "str2": cudf.Series(
                ["officer", "manager", "lawyer", "chef", "teacher"], dtype="str"
            ),
            "cat1": cudf.Series(["yes", "yes", "no", "yes", "no"]),
            "cat2": cudf.Series(
                ["20K+", "40K+", "60K+", "30K+", "50K+"])
            # ),
        }
    )


def _get_dirty_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with some missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to test the robustness of the vectorizer.
    """
    return cudf.DataFrame(
        {
            "int": cudf.Series([15, 56.0, pd.NA, 12, 44],nan_as_null=False),
            "float": cudf.Series([5.2, 2.4, 6.2, 10.45, np.nan],dtype='float', nan_as_null=False),
            "str1": cudf.Series(
                ["public", np.nan, "private", "private", "public"],dtype='object',nan_as_null=False
            ),
            "str2": cudf.Series(
                ["officer", "manager", None, "chef", "teacher"],dtype='object', nan_as_null=False
            ),
            "cat1": cudf.Series([np.nan, "yes", "no", "yes", "no"], dtype='object',nan_as_null=False),
            "cat2": cudf.Series(["20K+", "40K+", "60K+", "30K+", np.nan],dtype='object',nan_as_null=False),
        }
    )


def _get_datetimes_dataframe() -> pd.DataFrame:
    """
    Creates a DataFrame with various date formats,
    already converted or to be converted.
    """
    return cudf.DataFrame(
        {
            "pd_datetime": [
                pd.Timestamp("2019-01-01"),
                pd.Timestamp("2019-01-02"),
                pd.Timestamp("2019-01-03"),
                pd.Timestamp("2019-01-04"),
                pd.Timestamp("2019-01-05"),
            ],
            "np_datetime": [
                np.datetime64("2018-01-01"),
                np.datetime64("2018-01-02"),
                np.datetime64("2018-01-03"),
                np.datetime64("2018-01-04"),
                np.datetime64("2018-01-05"),
            ],
            "dmy-": [
                "11-12-2029",
                "02-12-2012",
                "11-09-2012",
                "13-02-2000",
                "10-11-2001",
            ],

            "ymd/": [
                "2014/12/31",
                "2001/11/23",
                "2005/02/12",
                "1997/11/01",
                "2011/05/05",
            ],
            "ymd/_hms:": [
                "2014/12/31 00:31:01",
                "2014/12/30 00:31:12",
                "2014/12/31 23:31:23",
                "2015/12/31 01:31:34",
                "2014/01/31 00:32:45",
            ],
        }
    )


def _test_possibilities(X):
    """
    Do a bunch of tests with the TableVectorizer.
    We take some expected transformers results as argument. They're usually
    lists or dictionaries.
    """
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    expected_transformers_df = {
        "numeric": ["int", "float"],
        "low_card_cat": ["str1", "cat1"],
        "high_card_cat": ["str2", "cat2"],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers)

    # Test with higher cardinality threshold and no numeric transformer
    expected_transformers_2 = {
        "low_card_cat": ["str1", "str2", "cat1", "cat2"],
    }
    vectorizer_default = TableVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers)

    # Test casting values
    vectorizer_cast = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # X_str = X.astype("object")
    # With pandas
    expected_transformers_plain = {
        "high_card_cat": ["str2", "cat2"],
        "low_card_cat": ["str1", "cat1"],
        "numeric": ["int", "float"],
    }
    vectorizer_cast.fit_transform(X)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers)


def test_with_clean_data():
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a clean dataset.
    """
    _test_possibilities(_get_clean_dataframe())


def test_with_dirty_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a dataset containing missing values.
    """
    _test_possibilities(_get_dirty_dataframe())


def test_auto_cast() -> None:
    """
    Tests that the TableVectorizer automatic type detection works as expected.
    """
    vectorizer = TableVectorizer()

    # Test datetime detection
    X = _get_datetimes_dataframe()

    expected_types_datetimes = {
        "pd_datetime": "datetime64[us]",
        "np_datetime": "datetime64[s]",
        "dmy-": "datetime64[ns]",
        "ymd/": "datetime64[ns]",
        "ymd/_hms:": "datetime64[ns]",
    }
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert expected_types_datetimes[col] == X_trans[col].dtype

    # Test other types detection

    expected_types_clean_dataframe = {
        "int": "int64",
        "float": "float64",
        "str1": "object",
        "str2": "object",
        "cat1": "object",
        "cat2": "object",
    }

    X = _get_clean_dataframe()
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert type_equality(expected_types_clean_dataframe[col], X_trans[col].dtype)

    # Test that missing values don't prevent type detection
    expected_types_dirty_dataframe = {
        "int": "float64",  # int type doesn't support nans -- NO SHIT
        "float": "float64",
        "str1": "object",
        "str2": "object",
        "cat1": "object",
        "cat2": "object",
    }

    X = _get_dirty_dataframe()
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert type_equality(expected_types_dirty_dataframe[col], X_trans[col].dtype)



def test_get_feature_names_out() -> None:
    X = _get_clean_dataframe()

    vec_w_pass = TableVectorizer(remainder="passthrough")
    vec_w_pass.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_pass = [
        "str1_private",
        "str1_public",
        "str2_chef",
        "str2_lawyer",
        "str2_manager",
        "str2_officer",
        "str2_teacher",
        "cat1_no",
        "cat1_yes",
        "cat2_20K+",
        "cat2_30K+",
        "cat2_40K+",
        "cat2_50K+",
        "cat2_60K+",
        "int",
        "float",
    ]
    # if parse_version(sklearn.__version__) < parse_version("1.0"):
    assert vec_w_pass.get_feature_names() == expected_feature_names_pass
   

def _is_equal(elements: Tuple[Any, Any]) -> bool:
    """
    Fixture for values that return false when compared with `==`.
    """
    elem1, elem2 = elements  # Unpack
    return pd.isna(elem1) and pd.isna(elem2) or elem1 == elem2


# def test_check_fitted_table_vectorizer():
#     """Test that calling transform before fit raises an error"""
#     X = _get_clean_dataframe()
#     tv = TableVectorizer()
#     with pytest.raises(NotFittedError):
#         tv.transform(X)

#     # Test that calling transform after fit works
#     tv.fit(X)
#     tv.transform(X)


def test_check_name_change():
    """Test that using SuperVectorizer raises a deprecation warning"""
    with pytest.warns(FutureWarning):
        SuperVectorizer()
