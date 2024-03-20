import statsmodels.formula.api as smf
import pandas as pd
import sys
sys.path.append("T:\Studies\CSCR\code\cscr\scripts\classification\ML")
from utils import process_data

def logistic_regression(dataset_path, constant_dict, multi=False, trunc=True):  # =FEATURE_LIST
    df = pd.read_json(dataset_path)
    df =  process_data(df.copy(),
                        scale_type="normal",
                        time_serie=constant_dict['INPUT_TIME_SERIES_FEATURES'],
                        other=constant_dict['INPUT_STATIC_FEATURES'],
                        truncate=trunc,
                        visit=True)   # visit=True for all visits

    TS = list(map(lambda x: x.replace('+', '_'), constant_dict['FEATURE_NAMES']))
    df.rename(lambda x: x.replace('+', '_'), axis='columns', inplace=True)
    if multi:
        reg=TS[0]
        for  b in TS[1:]:
            if b=='sex':
                b='C(sex)'
            reg += (" + " + b) 

        mod = smf.logit(formula=constant_dict["DISCRIMINATOR"] + " ~ " + reg, data=df)
        res = mod.fit()
        print(res.summary())
        pvals = list(res.pvalues)[1:]
        # print(pvals)
        coefs = res.params.values[1:]
    else:
        pvals = []
        coefs = []
        for b in TS:
            if b=='sex':
                b='C(sex)'
            mod = smf.logit(formula = constant_dict["DISCRIMINATOR"] + " ~ " + b, data=df)
            res = mod.fit(disp=0)
            pvals.append(res.pvalues[-1])
            coefs.append(res.params.values[-1])

    # print(TS)
    df= pd.DataFrame(data={'P-value': pvals, 'Coefficient': coefs}, index=constant_dict['FEATURE_NAMES'])

    return df

