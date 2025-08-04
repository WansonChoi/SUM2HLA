"""
- Secondary signal simulation 정답확인을 위한 modules들을 여기 모을거임.



"""

import os, sys, re
import numpy as np
import pandas as pd
import json

from datetime import datetime


def check_answer(_fpath_dict, _l_answer_primary: list, _l_answer_secondary: list):
    

    d_SWCA_out = None
    f_correct_primary = None
    f_correct_seconary = None

    with open(_fpath_dict, 'r') as f_json:
        d_SWCA_out = json.load(f_json)



    # print(d_SWCA_out)


    if "ROUND_1" in d_SWCA_out:
        f_correct_primary = pd.Series(d_SWCA_out['ROUND_1']).isin(_l_answer_primary).any()
    else:
        f_correct_primary = np.nan


    if "ROUND_2" in d_SWCA_out:
        f_correct_seconary = pd.Series(d_SWCA_out['ROUND_2']).isin(_l_answer_secondary).any()
    else:
        f_correct_seconary = np.nan



    return f_correct_primary, f_correct_seconary



def check_SSFN(_df_SSFN, _l_answer_primary, _l_answer_secondary):

    sr_OUT_SWCA_dict = _df_SSFN['fpath_OUT'].map(lambda x: x + ".r2pred0.6.ma.SWCA.dict")

    sr_checked = sr_OUT_SWCA_dict.map(lambda x: check_answer(x, _l_answer_primary, _l_answer_secondary))

    df_checked = pd.DataFrame([list(_) for _ in sr_checked], columns=['checked_1st', 'checked_2nd'])

    df_RETURN = pd.concat([_df_SSFN, df_checked], axis=1)


    return df_RETURN



def check_SIMULATIONS(_d_fpath_SSFN, _l_answer_primary, _l_answer_secondary, _nrows=100):


    ## 일부러 procedural하게 짠다?

    d_OUT = {}

    for i, (_ncp_2nd, _fpath_SSFN) in enumerate(_d_fpath_SSFN.items()):
        print(f"=====[{i}]: {_ncp_2nd:+}")


        df_temp_SSFN = pd.read_csv(_fpath_SSFN, sep='\t', header=0, nrows=_nrows)
        df_temp_SSFN = check_SSFN(df_temp_SSFN, _l_answer_primary, _l_answer_secondary)

        # print(df_temp_SSFN)

        d_OUT[_ncp_2nd] = df_temp_SSFN



    return d_OUT