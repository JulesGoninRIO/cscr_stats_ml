import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ks_2samp, ttest_ind
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import argparse
import json
import sys
sys.path.append("T:/Studies/CSCR/code/cscr/scripts/classification/ML")
from utils import process_data

DEFAULT_CONFIG_STR = ''

def main(args) -> dict:
    """ 
    Pipeline computing the statistics for a given dataset.
    
    This function compute stats on the data: p-values and Cohen's d effect sizes.
    
    :param args: Argument coming from the parser. See the parser for more detail.
    
    :return: Output a dictionnary with the different stats computed.
    """
    # Parse arguments (either from CLI or config file)
    if args.config == DEFAULT_CONFIG_STR:
        dataset_path = args.dataset_path
        discriminator_name = args.discriminator_name
        alpha = args.alpha
        stats_test = args.stats_test
        correction_method = args.correction_method
        features_order = args.features_order
        constant_dict = args.constant_dict
        per_patient = args.per_patient
        trunc = args.trunc
    else:
        with open(args.config, 'r') as config_file:
            config = json.load(config_file)
        dataset_path = config['dataset_path']
        discriminator_name = config['discriminator_name']
        alpha = config['alpha']
        stats_test = config['stats_test']
        correction_method = config['correction_method']
        features_order = config['features_order']
        constant_dict = config['constant_dict']
        per_patient = config['per_patient']
        trunc = config['trunc']
    # Launch Pipeline
    statistics_results = statistics_pipeline(dataset_path, discriminator_name, constant_dict, alpha, stats_test,
                                             correction_method, features_order, per_patient, trunc)
    return statistics_results

def statistics_pipeline(dataset_path: str, discriminator_name: str,
                        constant_dict: dict, alpha: float=0.05,
                        stats_test: str='ks', correction_method: str='bonferroni',
                        features_order: list=[], per_patient: str="mean",
                        trunc: bool=False) -> dict:
    """ Pipeline computing the statistics for a given dataset.
    
    This function compute stats on the data: p-values and Cohen's d effect sizes.
    
    :param dataset_path: Path to the dataset.
    :param discriminator_name: Name of the feature which will be used to discriminate groups.
    :param constant_dict: Dictionnary containing the constants for input features names.
    :param alpha: Significance level.
    :param correction_method: Method to correct for multiple testing.
    
    :return: Output a dictionnary with the different stats computed.
    """
    measures_df = pd.read_json(dataset_path)
    # add a parameter for Truncate
    patients_df = process_data(measures_df.copy(deep=True), scale_type=None,
                              time_serie=constant_dict['INPUT_TIME_SERIES_FEATURES'],
                              other=constant_dict['INPUT_STATIC_FEATURES'],
                              truncate=trunc, visit=False)
    # visit_df = visit_df.sample(n=500, random_state=42)
    # Compute mean features values across visits
    for biomarker_name in constant_dict['BIOMARKER_NAMES']:
        if per_patient == 'mean':
            patients_df[biomarker_name] = patients_df[biomarker_name].apply(lambda l: np.mean(l))  # l[0] np.mean(l)
        elif per_patient == 'first':
            patients_df[biomarker_name] = patients_df[biomarker_name].apply(lambda l: l[0])
    # Compute Basic Stats
    biomarker_df = patients_df[constant_dict['BIOMARKER_NAMES']]
    # Apply Transforms
    for binary_feature in constant_dict['BINARY_FEATURES']:
        binary_mapping = constant_dict['BINARY_FEATURES_MAPPINGS'][binary_feature]
        patients_df[binary_feature] = patients_df[binary_feature].apply(lambda x: binary_mapping[x])
    patients_df[constant_dict['AGE_FEATURE_NAME']] = patients_df[constant_dict['AGE_FEATURE_NAME']].apply(lambda x: x[0])   # keep first age
    # Normalize
    scaler = StandardScaler()
    patients_df[constant_dict['FEATURE_NAMES']] = scaler.fit_transform(patients_df[constant_dict['FEATURE_NAMES']])
    # Compute and plot effect sizes
    effect_size_df = features_effect_size(patients_df, constant_dict['FEATURE_NAMES'],
                                          discriminator_name, stats_test, correction_method, alpha)
    return effect_size_df

def compute_cohen_d(d1: np.array, d2: np.array) -> float:
    """ Compute the Cohen's d coefficient between d1 and d2. 
    
    :param d1: Numpy array with the numerical values of the first feature.
    :param d1: Numpy array with the numerical values of the second feature.

    :return: Value of the Cohen's d coefficient.
    """
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1), np.var(d2)
    s = (s1 + s2) / 2
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s

def single_feature_effect_size(measures_df: pd.DataFrame, feature_name: str,
                               discriminator_name: str, stats_test: str='ks') -> dict:
    """ Compute the effect size and significance of difference in distribution.
    
    :param measure_df: Pandas DataFrame containing all data features.
    :param feature_name: Name of the feature we want to test.
    :param discriminator_name: Name of the feature which will be used to discriminate groups.

    :return: Dictionnary with the summary of the statistics of difference in distribution.
    """
    # Get numerical data
    discrimator_value = measures_df[discriminator_name].value_counts().index[0]
    data_first_group = measures_df[measures_df[discriminator_name] == discrimator_value][feature_name].values
    data_second_group = measures_df[measures_df[discriminator_name] != discrimator_value][feature_name].values
    # Compute statistics
    cohen_d = compute_cohen_d(data_first_group, data_second_group)
    if stats_test == 'ks':
        test_statistic, p_value = ks_2samp(data_first_group, data_second_group)
        return {'P_Value':p_value, 'Cohen_D':cohen_d}
    elif stats_test == 'ttest':
        test_statistic, p_value = ttest_ind(data_first_group, data_second_group, equal_var=False)    # equal_var=False for Welch's t-test
        return {'P_Value':p_value, 'Cohen_D':cohen_d}

def features_effect_size(measures_df: pd.DataFrame, feature_names: list,
                         discriminator_name: str, stats_test: str='ks',
                         correction_method: str='bonferroni', alpha: float=0.05) -> pd.DataFrame:
    """ Compute the effect size and difference in distribution for all given features. 
    
    :param measure_df: Pandas DataFrame containing all data features.
    :param feature_names: List of features we want to test.
    :param discriminator_name: Name of the feature which will be used to discriminate groups.
    :param correction_method: Method for correcting p-values.
    :param alpha: Significance threshold for statistics.

    :return: Pandas DataFrame with the results (columns) for each features (index).
    """
    output = {'P_Value':[], 'Cohen_D':[]}
    for feature_name in feature_names:
        feature_effect_size_results = single_feature_effect_size(measures_df, feature_name, discriminator_name, stats_test)
        for output_field in output.keys():
            output[output_field].append(feature_effect_size_results[output_field])
    # Correct p-values for multiple testing
    output['Uncorrected_P_Value'] = output['P_Value']
    output['P_Value'] = multipletests(pvals=output['P_Value'], alpha=alpha, method=correction_method)[1]
    # Format output
    return pd.DataFrame(data=output, columns=output.keys(), index=feature_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute basic statistics for the given dataset.')
    config_file_args = parser.add_argument_group('Config File')
    config_file_args.add_argument('--config', type=str, required=False, help='Path to the config file containing all arguments mentionned by the parser.', 
                                  default=DEFAULT_CONFIG_STR)
    required_args = parser.add_argument_group('Required arguments if no config file provided')
    required_args.add_argument('--dataset_path', type=str, required=False, help='Path to the Pandas DataFrame in json format.')
    required_args.add_argument('--discriminator_name', type=str, required=False, 
                               help='Name of the column that is interesting for partitioning the data. Usually a disease label.')
    parser.add_argument('--alpha', type=float, required=False, help='Significance level.', default=0.05)
    parser.add_argument('--correction_method', type=str, required=False,
                        help='Method for multiple testing correction, given to multipletests method of the statsmodels package.',
                        default='bonferonni')
    args = parser.parse_args()
    main(args)

