import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DEFAULT_TIME_SERIE = ["date", "SRF", "IRF", "PED", "RNFL", "GCL+IPL", "INL+OPL", "ONL", "PR+RPE", "CC+CS", "CVI", "DSCORE", "PV_AREA", "age"]
DEFAULT_OTHER_FEATURE = ["ID", "laterality", "sex"] # , "label"

DEFAULT_LOG_FEATURE = ["SRF", "IRF", "PED", "DSCORE"]
DEFAULT_SCALE_FEATURE = ["date", "SRF", "IRF", "PED", "RNFL", "GCL+IPL", "INL+OPL", "ONL", "PR+RPE", "CC+CS", "CVI", "DSCORE", "PV_AREA", "age"]

SCALER_TYPE = {"normal": StandardScaler(), "min_max": MinMaxScaler()}

def process_data(df: pd.DataFrame,
                 scale_type: str,
                 time_serie=DEFAULT_TIME_SERIE,
                 other=DEFAULT_OTHER_FEATURE,
                 log_feature=DEFAULT_LOG_FEATURE,
                 scale_feature=DEFAULT_SCALE_FEATURE,
                 transform_date = True,
                 remove_predict_point = False,
                 truncate = False,
                 visit = False):
    """Prepare data set for training

    Args:
        df (pd.DataFrame): data
        scale_type (str, optional): Scaling type [normalize, min_max, None]. Defaults to normalize.
        time_serie (list, optional): time serie features. Defaults to DEFAULT_TIME_SERIE.
        other (list, optional): other features. Defaults to DEFAULT_OTHER_FEATURE.
        log_feature (list, optional): feature to log. Defaults to DEFAULT_LOG_FEATURE.
        scale_feature (list, optional): feature to scale. Defaults to DEFAULT_SCALE_FEATURE.
        transform_date (bool, optional): true to transfrom date into delta time. Defaults to True.
        remove_predict_point (bool, optional): true to remove the last data point. Defaults to False.
        truncate (bool, optional): true to truncate time series before recurrence. Defaults to False.
        homogenize (int optional): length for time series homogenisation, -1 if no homogenisation. Defaults to -1.
        noise (bool, optional): add noise to homogenise signal. Defaults to True.
        visit (bool, optional): to output one visit per eye or one eye per row. Defaults to True
    """
    # drop -1 label
    df = df.loc[df["label"]!=-1].reset_index(drop=True)

    # transform date
    if transform_date:
        df["date"] = df["date"].apply(transform_dates)

    if truncate:
        df = truncate_ts(df)

    # remove prediction point
    if remove_predict_point:
        for col in time_serie:
            df[col] = df.apply(lambda x: x[col][:-1] if x["flag"] != 4 else x[col], axis=1)
            #df[col] = df[col].apply(lambda x: x[:-1])

    # to visit structure
    df = to_visit(df, time_serie=time_serie, other=other)

    # scale
    if scale_type is not None:
        df = scale_data(df, scale_type=scale_type, log_feature=log_feature, scale_feature=scale_feature)

    if not visit:
        # to patient structure
        df = to_patient(df, time_serie=time_serie, other=other)

    return df.reset_index(drop=True)

def to_visit(df: pd.DataFrame, time_serie=DEFAULT_TIME_SERIE, other=DEFAULT_OTHER_FEATURE):
    """Change structure of df to one row per eye to one row per visit 

    Args:
        df (pd.DataFrame): orginal dataframe
        time_serie (list, optional): time serie feature. Defaults to DEFAULT_TIME_SERIE.
        other (list, optional): other feature. Defaults to DEFAULT_FEATURE.
    """
    # extend scalar feature to list
    for f in other:
        df[f] = df.apply(extend_feature, f=f, axis=1)

    # create final dataset
    df_final = pd.DataFrame()
    for col in other + time_serie:
        df_final[col] = pd.concat([pd.Series(val) for val in df[col]])
    return df_final


def to_patient(df: pd.DataFrame, time_serie=DEFAULT_TIME_SERIE, other=DEFAULT_OTHER_FEATURE):
    """Change structure of df to one row per visit to one row per eye 

    Args:
        df (pd.DataFrame): orginal dataframe
        time_serie (list, optional): time serie to extract. Defaults to DEFAULT_TIME_SERIE.
        feature (list, optional): scalar feature to extract. Defaults to DEFAULT_FEATURE.
    """
    df_final = pd.DataFrame()
    df_final[["ID", "laterality"]] = df.groupby(["ID", "laterality"]).count().reset_index()[["ID", "laterality"]]
    for col in other + time_serie:
        if not (col in ["ID", "laterality"]):
            if col in time_serie:
                df_final[col] = df.groupby(["ID", "laterality"])[col].apply(list).values
            else:
                df_final[col] = df.groupby(["ID", "laterality"])[col].apply(lambda x: x.unique()[0]).values
    return df_final


def scale_data(df, scale_type, log_feature=DEFAULT_LOG_FEATURE, scale_feature=DEFAULT_SCALE_FEATURE):
    """scale with log and normalization features

    Args:
        df (pd.DataFrame): data
        scale_type (str): scaling type [normalize, min_max]
        log_feature (list, optional): features to log. Defaults to DEFAULT_LOG_FEATURE.
        scale_feature (list, optional): feature to scale. Defaults to DEFAULT_SCALE_FEATURE.
    """
    transformer = FunctionTransformer(np.log1p)
    scaler = SCALER_TYPE[scale_type]
    if log_feature is not None:
        df[log_feature] = transformer.transform(df[log_feature])
    df[scale_feature] = scaler.fit_transform(df[scale_feature])
    return df


def extend_feature(row, f):
    """helper for to_visit:
    extend scalar to list

    Args:
        row (pd.Series): _description_
        f (str): feature to extend
    """
    return [row[f] for i in range(len(row["date"]))]


def transform_dates(dates):
    """ Transforms visit dates of a patient from string to 
        time difference from last visit in weeks.
    """
    dates_ = []
    for d in dates:
        if d != 'info':
            dates_.append((datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date(), d))
    dates_ = sorted(dates_, key=lambda x: x[0])
    reference = dates_[-1][0]
    for i, (d, k) in enumerate(dates_):
        # Add time difference in weeks and the original date
        #dates_[i] = ((reference - d).days//7, k)
        dates_[i] = (reference - d).days//7
    return dates_


def truncate_ts(df, feature_concat=DEFAULT_TIME_SERIE):
    """truncate time series before recurrence between [start_idx: end_idx+2]

    Args:
        df (pd.DataFrame): data
    """
    for col in feature_concat:
        df[col] = df.apply(lambda x: x[col][x["start_idx"]: x["end_idx"]+2], axis=1)
    return df