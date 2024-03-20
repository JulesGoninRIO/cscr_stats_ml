import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
import json
import sys
from scipy.stats import pearsonr
sys.path.append("T:/Studies/CSCR/code/cscr/scripts/classification/ML")
from utils import process_data

# CONSTANT_DICT = {
# 'INPUT_STATIC_FEATURES': ["ID", "laterality", "sex", "label"],
# 'INPUT_TIME_SERIES_FEATURES': ["date", "SRF", "IRF", "PED", "RNFL", "GCL+IPL", "INL+OPL", "ONL", "PR+RPE", "CC+CS", "CVI", "DSCORE", "PV_AREA", "age"],
# 'BIOMARKER_NAMES': ["SRF", "IRF", "PED", "RNFL", "GCL+IPL", "INL+OPL", "ONL", "PR+RPE", "CC+CS", "CVI", "DSCORE", "PV_AREA"],
# 'BINARY_FEATURES_MAPPINGS': {'sex': {'M': 1, 'F': 0}},
# 'AGE_FEATURE_NAME': 'age'
# }

DEFAULT_CONFIG_STR = ''

def main(args) -> dict:
    # Parse arguments (either from CLI or config file)
    if args.config == DEFAULT_CONFIG_STR:
        dataset_path = args.dataset_path
        discriminator_name = args.discriminator_name
        alpha = args.alpha
        constant_dict = args.constant_dict # CONSTANT_DICT
        trunc = args.trunc
        visit_level_log_features = args.visit_level_log_features
    else:
        with open(args.config,'r') as config_file:
            config = json.load(config_file)
        dataset_path = config['dataset_path']
        discriminator_name = config['discriminator_name']
        alpha = config['alpha']
        constant_dict = config['constant_dict']
        trunc = config['trunc']
        visit_level_log_features = config['visit_level_log_features']
    # Launch Pipeline
    corr_matrix, mask = statistics_pipeline(dataset_path, discriminator_name, constant_dict, visit_level_log_features, alpha, trunc)
    return corr_matrix, mask

def statistics_pipeline(dataset_path: str, discriminator_name: str,
                        constant_dict: dict, visit_level_log_features, alpha: float=0.05,
                        trunc: bool=False) -> dict:
    measures_df = pd.read_json(dataset_path)
    patients_df = process_data(measures_df.copy(deep=True),
                            scale_type=None,
                            time_serie=constant_dict['INPUT_TIME_SERIES_FEATURES'],
                            other=constant_dict['INPUT_STATIC_FEATURES'],
                            truncate=trunc,
                            visit=False)
    visit_df = process_data(measures_df.copy(deep=True),
                            scale_type=None,
                            time_serie=constant_dict['INPUT_TIME_SERIES_FEATURES'],
                            other=constant_dict['INPUT_STATIC_FEATURES'],
                            truncate=trunc,
                            visit=True)
    # visit_df = visit_df.sample(n=500, random_state=42)
    visit_level_log_features = {"SRF":1, "IRF":2, "PED": 1, "RNFL": 1, "DSCORE":1}
    for biomarker_name, log_iteration in visit_level_log_features.items():
        for _ in range(log_iteration):
            min_value = visit_df[biomarker_name].min()
            if min_value < 0:
                epsilon = 0.01+np.abs(min_value)
            else:
                epsilon = 0.01
            patients_df[biomarker_name] = patients_df[biomarker_name].apply(lambda l: np.log(np.array(l)+epsilon))
            visit_df[biomarker_name] = visit_df[biomarker_name].apply(lambda l: np.log(l+epsilon))
    for biomarker_name in constant_dict['BIOMARKER_NAMES']:
        patients_df[biomarker_name] = patients_df[biomarker_name].apply(lambda l: np.mean(l))

    biomarker_df = patients_df[constant_dict['BIOMARKER_NAMES']]
    corr_matrix = biomarker_correlation(biomarker_df)
    return corr_matrix, correlation_matrix(corr_matrix, corr_matrix.columns, biomarker_df)

def biomarker_correlation(biomarker_df: pd.DataFrame) -> list:
    biomarker_names = biomarker_df.columns

    # Compute corr matrix
    corr_matrix = (biomarker_df.corr(method="pearson")).abs()
    # change diag to one for visualization
    for biomarker in biomarker_names:
        corr_matrix.at[biomarker, biomarker] = 1

    corr_matrix.index = biomarker_names
    corr_matrix.columns = biomarker_names

    return corr_matrix

def correlation_matrix(corr_matrix, biomarker_names, biomarker_df, alpha: float=0.05):
    # Compute p-values
    def pearsonr_pval(x, y):
        return pearsonr(x, y)[1]

    # Compute corr matrix for p-values
    pvalues_matrix = biomarker_df.corr(method=pearsonr_pval)
    pvalues_matrix.index = biomarker_names
    pvalues_matrix.columns = biomarker_names

    # Create a mask for non-significant coefficients
    mask = pvalues_matrix < alpha

    return mask


