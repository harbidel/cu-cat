import pytest
import numpy as np, cupy as cp
import pandas as pd, cudf
from sklearn.exceptions import NotFittedError

from cu_cat._datetime_encoder import DatetimeEncoder

def get_date_array() -> cp.array:
    df = cudf.DataFrame()
    df['0'] = ['2019-01-15 16:05:39', '2022-02-02 16:33:38', '2015-01-22 16:33:38']
    df['0'] = df['0'].astype('datetime64[s]')
    df['1'] = ['2022-10-11 20:05:39', '2022-10-11 20:33:38', '2022-11-12 20:33:38']
    df['1'] = df['1'].astype('datetime64[s]')
    df['2'] = ['2015-02-11 16:05:39', '2022-10-11 04:33:38', '2013-04-14 03:33:38']
    df['2'] = df['2'].astype('datetime64[s]')
    return df


def get_constant_date_array() -> cp.array:
    df = cudf.DataFrame()
    df['0'] = ['2015-01-15 19:05:39', '2015-01-15 19:05:39', '2015-01-15 19:05:39']
    df['0'] = df['0'].astype('datetime64[s]')
    df['1'] = df['0'].astype('datetime64[s]')
    df['2'] = df['0'].astype('datetime64[s]')


    return df


def get_dirty_datetime_array() -> np.array:
    
    df = cudf.DataFrame(nan_as_null=False)
    df['0'] = ['2019-02-15 16:05:39', pd.NaT, '2015-01-10 12:33:38']
    df['0'] = df['0'].astype('datetime64[s]')
    df['1'] = ['2022-10-11 20:05:39', '2022-10-11 19:33:38', '2022-11-12 20:33:38']
    df['1'] = df['1'].astype('datetime64[s]')
    df['2'] = [np.nan, '2022-10-11 04:33:38', '2013-04-14 03:33:38']
    df['2'] = df['2'].astype('datetime64[s]')
    return df

def get_datetime_with_TZ_array() -> pd.DataFrame:

    res = cudf.DataFrame()
    res['0'] = ['2019-01-15 16:05:39', '2022-02-10 16:33:38', '2015-01-10 16:33:38']
    res['0'] = res['0'].astype('datetime64[s]')
    res['1'] = ['2022-10-11 20:05:39', '2022-10-11 20:33:38', '2022-11-12 20:33:38']
    res['1'] = res['1'].astype('datetime64[s]')
    res['2'] = ['2015-02-11 16:05:39', '2022-10-11 04:33:38', '2013-04-14 03:33:38']
    res['2'] = res['2'].astype('datetime64[s]')
    # return res
    for col in res.columns:
        res[col] = pd.DatetimeIndex(res[col]).tz_localize("Asia/Kolkata")
    return res


def test_fit() -> None:
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder()
    expected_to_extract = ["year", "month", "day", "hour"]
    expected_features_per_column_ = {
        0: ["year", "month", "day"],
        1: ["month", "day"],
        2: ["year", "month", "day","hour"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_to_extract = ["year", "month", "day", "hour", "dayofweek"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "dayofweek"],
        1: ["month", "day", "dayofweek"],
        2: ["year", "month", "day", "hour", "dayofweek"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Datetimes
    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_to_extract = ["year", "month", "day", "hour", "dayofweek"]
    expected_features_per_column_ = {
        0: ["year", "month", "day",  "dayofweek"], #, "total_time"],
        1: ["month", "day", "dayofweek"], # "total_time"],
        2: ["year", "month", "day", "hour", "dayofweek"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Dirty Datetimes
    X = get_dirty_datetime_array()
    enc = DatetimeEncoder()
    expected_to_extract = ["year", "month", "day", "hour"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "hour"], #  "total_time"],
        1: ["month", "day", "hour" ], #, "total_time"],
        2: ["year", "month", "day", "hour"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Feature names
    # Without column names
    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = [
        "0_year",
        "0_month",
        "0_day",
        # "0_hour",
        "0_dayofweek",
        # "0_total_time",
        "1_month",
        "1_day",
        # "1_hour",
        "1_dayofweek",
        # "1_total_time",
        "2_year",
        "2_month",
        "2_day",
        "2_hour",
        "2_dayofweek",
    ]
    enc.fit(X)
    assert enc.get_feature_names_out() == expected_feature_names

    # With column names
    X = get_date_array()
    # X = pd.DataFrame(X)
    X.columns = ["col1", "col2", "col3"]
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = [
        "col1_year",
        "col1_month",
        "col1_day",
        # "col1_hour",
        "col1_dayofweek",
        # "col1_total_time",
        "col2_month",
        "col2_day",
        # "col2_hour",
        "col2_dayofweek",
        # "col2_total_time",
        "col3_year",
        "col3_month",
        "col3_day",
        "col3_hour",
        "col3_dayofweek",
    ]
    enc.fit(X)
    assert enc.get_feature_names_out() == expected_feature_names


def test_transform():
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2019, 1, 15,1,10,11,1,2015,2,11,16,2],
            [2022, 2,2,2,10,11,1,2022,10,11,4,1],
            [2015, 1,22,3,11,12,5,2013,4,14,3,6],
        ]
    )
    # enc.fit(X)
    assert np.allclose(enc.fit_transform(X).to_numpy(), expected_result, equal_nan=True)

    # Dirty datetimes
    X = get_dirty_datetime_array()#[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2019, 2, 15,16, 2022,10,11,20,1970,1,2,3],
            [1970,1,2,3,2022,10, 11,19, 2022, 10,11,4],
            [2015, 1, 10,12,2022,11,12,20,2013,4,14,3],
        ]
    )
    # Time from epochs in seconds

    assert (enc.fit_transform(X).to_numpy(), expected_result)#, equal_nan=True)


def test_check_fitted_datetime_encoder():
    """Test that calling transform before fit raises an error"""
    X = get_date_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    with pytest.raises(NotFittedError):
        enc.transform(X)
    
    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)
