import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd

import math

import src.Util as mod_Util

from threadpoolctl import threadpool_limits


########## Plotting module for result summary
# # %matplotlib inline
# import seaborn as sns
# import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import rc_context
# from matplotlib import rcParams
# from matplotlib import cm
# import matplotlib.colors as mcolors
# from matplotlib.transforms import ScaledTranslation

# # from adjustText import adjust_text

# # FIGSIZE=(3,3)
# # rcParams['figure.figsize']=FIGSIZE
# rcParams['font.family'] = "Arial"
# # rcParams['mathtext.default'] = "Arial"



########## (Deprecated) (1) LD panel의 marker set을 SNP + HLA로 나누기

def partition_LD_marker_set(_sr_summary_marker_set, _sr_LD_marker_set):
    
    # display(_sr_summary_marker_set)
    # display(_sr_LD_marker_set)
    
    ##### flag for SNP markers
    f_SNP = _sr_LD_marker_set.isin(_sr_summary_marker_set)
    
    ##### Subsetting
    sr_LD_SNP_markers = _sr_LD_marker_set[f_SNP]
    sr_LD_HLA_markers = _sr_LD_marker_set[~f_SNP]
    
    
    # display(sr_LD_SNP_markers)
    # display(sr_LD_HLA_markers)
    
    return sr_LD_SNP_markers, sr_LD_HLA_markers



########## (1) LD panel의 marker set (전체 집합)을 Observed vs. Unobserved HLA markers로 나누기. (2025.05.06.)
"""
- 예전에는 SNPs vs. Non-SNPs 들로 분류했고, Non-SNPs == HLA markers들이라 가정했음. (v1)
- 이후 clumped SNPs들만 주어졌을 때를 test할일이 생김.
    - 이때 v1때 처럼 분류하면, clumped SNPs들을 제외한 SNPs들이 Non-SNPs들로 분류됨.
- 이때부터 Observed vs. Unobserved로 나눴고, Unobserved는 HLA markers들만 포함하도록 규정함.
"""

def partition_LD_marker_set_v2(_sr_summary_marker_set, _sr_LD_marker_set):

    # display(_sr_summary_marker_set)
    # display(_sr_LD_marker_set)

    ##### flag for SNP markers
    f_Obs = _sr_LD_marker_set.isin(_sr_summary_marker_set)
    f_is_HLA_locus = mod_Util.is_HLA_locus(_sr_LD_marker_set)

    ##### Subsetting
    sr_LD_Obs_markers = _sr_LD_marker_set[f_Obs]
    sr_LD_Unobs_markers = _sr_LD_marker_set[~f_Obs & f_is_HLA_locus]  # Impute해야할 markers
        # 예전에는 그냥 ~f_Obs이렇게 짜놨음.
        # 그랬더니 Obs SNPs들을 MAF-filtering했을 때, 이 filtered된 SNPs들이 Impute할 대상으로 분류됨.

    # display(sr_LD_Obs_markers)
    # display(sr_LD_Unobs_markers)

    return sr_LD_Obs_markers, sr_LD_Unobs_markers



########## (2) LD를 parition한 SNP + HLA set으로 나누기 => 4 marices

def partition_LD_matrix(_df_LD, _sr_LD_SNP_markers, _sr_LD_HLA_markers):
    
    # if _df_LD.shape[0] != _sr_LD_SNP_markers.shape[0] + _sr_LD_HLA_markers.shape[0]:
    #     print("Wrong dimension!")
    #     return -1
    
    ##### LD: observed
    df_LD_SNP = _df_LD.loc[_sr_LD_SNP_markers, _sr_LD_SNP_markers]
    
    
    ##### LD: covariance part
    df_LD_SNPxHLA = _df_LD.loc[_sr_LD_SNP_markers, _sr_LD_HLA_markers]
    df_LD_HLAxSNP = _df_LD.loc[_sr_LD_HLA_markers, _sr_LD_SNP_markers] # 이거 transpose() 오래 걸릴거 같아서 그냥 이렇게 함.
    
    ##### LD: Unobserved - Imputation해야 하는 애들.
    df_LD_HLA = _df_LD.loc[_sr_LD_HLA_markers, _sr_LD_HLA_markers]
    
    
    return df_LD_SNP, df_LD_SNPxHLA, df_LD_HLAxSNP, df_LD_HLA



########## (3) Conditional mean과 covariance계산하기

def calc_conditional_mean_cov(_df_X1_obs, _df_LD_11, _df_LD_12, _df_LD_21, _df_LD_22):
    
    if _df_X1_obs.shape[0] != _df_LD_11.shape[0]:
        print("Wrong dimension between the summary and its LD matrix!")
        print(f"_df_X1_obs: {_df_X1_obs.shape[0]}")
        print(f"_df_LD_11: {_df_LD_11.shape[0]}")
        return -1
    
    """
    - X1 := Observed Z-scores들이 주어지는 marker set
        - 우리 상황에서는 'SNP'
    - X2 := Imputate해야하는 marker set
        - 우리 상황에서는 'HLA'
        
    - 당연하지만 LD_{11, 12, 21, 22}는 다 주어지는거고.
    
    """

    ##### (1) match: Summary의 SNP order를 LD_11의 order로 맞추기 (***; 아주 아주 중요)
    ## 굳이 무거운 Pandas DataFrame으로 전처리를 진행한 목적 그 자체.
    
    _df_X1_obs_2 = _df_X1_obs \
                        .rename({"SNP_LD": "SNP", "Z_fixed": "Z"}, axis=1) \
                        .set_index("SNP") \
                        .loc[_df_LD_11.index, :]
    
    # print(_df_X1_obs_2)
    # print(_df_LD_11)
    # print(_df_LD_11.shape)
    # _df_LD_11.to_csv(
    #     "/data02/wschoi/_hCAVIAR_v2/SUM2HLA/df_LD_11.export.txt", sep='\t', header=True, index=False, na_rep="NA"
    # )
    
    """
    이렇게 display해서 `.loc[_df_LD_11.index, :]` 이걸로 확실히 match되는거 확인함. (2025.02.18.)
    """
    
    
    ##### (2) calc conditional mean and covariance

    def func_v1(): # v1 / original (최초로 작업했던거 and Deadlock of numpy threads 나던거.)

        arr_LD_11_inv = np.linalg.inv(_df_LD_11)
        print(arr_LD_11_inv)

        conditional_mean = _df_LD_21.values @ arr_LD_11_inv @ _df_X1_obs_2['Z'].values # mu1 and mu2는 0이라서 제외함.
        conditional_mean = pd.DataFrame({"Conditional_mean": conditional_mean}, index=_df_LD_22.index)
        
        conditional_cov = _df_LD_22.values - _df_LD_21.values @ arr_LD_11_inv @ _df_LD_12.values
        conditional_cov = pd.DataFrame(conditional_cov, columns=_df_LD_22.columns, index=_df_LD_22.index)

        return conditional_mean, conditional_cov
    

    def func_v2(): # threadpool_limits 활용

        with threadpool_limits(limits=1, user_api='blas'):
            arr_LD_11_inv = np.linalg.inv(_df_LD_11)
            # print(arr_LD_11_inv)

            conditional_mean = _df_LD_21.values @ arr_LD_11_inv @ _df_X1_obs_2['Z'].values # mu1 and mu2는 0이라서 제외함.
            conditional_mean = pd.DataFrame({"Conditional_mean": conditional_mean}, index=_df_LD_22.index)
            
            conditional_cov = _df_LD_22.values - _df_LD_21.values @ arr_LD_11_inv @ _df_LD_12.values
            conditional_cov = pd.DataFrame(conditional_cov, columns=_df_LD_22.columns, index=_df_LD_22.index)

        return conditional_mean, conditional_cov


    def func_v3(): # solve활용. (얘도 잘 되는거 확인함.)

        conditional_mean = _df_LD_21.values @ np.linalg.solve(_df_LD_11, _df_X1_obs_2['Z'].values) # mu1 and mu2는 0이라서 제외함.
        conditional_mean = pd.DataFrame({"Conditional_mean": conditional_mean}, index=_df_LD_22.index)
        
        conditional_cov = _df_LD_22.values - _df_LD_21.values @ np.linalg.solve(_df_LD_11, _df_LD_12.values)
        conditional_cov = pd.DataFrame(conditional_cov, columns=_df_LD_22.columns, index=_df_LD_22.index)

        return conditional_mean, conditional_cov


    # conditional_mean, conditional_cov = func_v1()
    conditional_mean, conditional_cov = func_v2()
    # conditional_mean, conditional_cov = func_v3()

    """
    - 최종 v2로 가는걸로 함.
        - 어차피 v3의 solve를 적용한다 쳐도, 잠정적으로 Deadlock은 함수 종류보다는 멀티스레딩 자체의 문제이기 때문에 threadpool_limits을 걸어주는게 좋음.
            - (당장 `func_v3()`내에 걸어놓지는 않았음.)
        - 그럴거면 걍 v2로.
        - 추가적으로, solve를 적용하니 아주 살짝 다름. (얘가 오히려 더 정확할 수 있다고 함.)
            - 추후 T1DGC와의 미세하게 replication안되는 것 처럼 보이는 문제를 방지하고자 함.
    
    """
    
    
    return conditional_mean, conditional_cov



def __MAIN__(_fpath_ss_matched, _fpath_ref_LD, _fpath_ref_MAF):
    
    ##### (0) load data
    df_ss_matched = pd.read_csv(_fpath_ss_matched, sep=r'\s+', header=0) \
                        if isinstance(_fpath_ss_matched, str) else _fpath_ss_matched
    
    if df_ss_matched['Z_fixed'].isna().any(): # 사실상 CD를 위한 예외처리
        # display(df_ss_matched)
        df_ss_matched.dropna(subset=["Z_fixed"], axis=0, inplace=True) 
    
    df_LD_PSD = pd.read_csv(_fpath_ref_LD, sep='\t', header=0) \
                    if isinstance(_fpath_ref_LD, str) else _fpath_ref_LD
    df_LD_PSD.index = df_LD_PSD.columns

    df_ref_MAF = pd.read_csv(_fpath_ref_MAF, sep=r'\s+', header=0).drop(['NCHROBS'], axis=1) \
                    if isinstance(_fpath_ref_MAF, str) else _fpath_ref_MAF



    ##### (1) LD panel의 marker set을 SNP + HLA로 나누기
    sr_LD_SNP_markers, sr_LD_HLA_markers = \
        partition_LD_marker_set_v2(df_ss_matched['SNP_LD'], df_LD_PSD.columns.to_series().reset_index(drop=True))



    ##### (2) LD를 parition한 SNP + HLA set으로 나누기 => 4 marices
    LD_partitioned = partition_LD_matrix(df_LD_PSD, sr_LD_SNP_markers, sr_LD_HLA_markers)    
    # print(LD_partitioned)

    
    ##### (3) Conditional mean과 covariance계산하기 (main; the summary imputation)
    df_cond_mean, df_cond_cov = calc_conditional_mean_cov(
        df_ss_matched[['SNP_LD', 'Z_fixed']],
        *LD_partitioned
    )
    # print(df_cond_mean)
    # print(df_cond_cov)



    ##### (4) Conditional variance (uncertainty measure)와 MAF정보 챙겨서 return하기.
    sr_cvar = pd.Series(np.diagonal(df_cond_cov), index=df_cond_cov.index, name='Conditional_var')
    sr_r2_pred = (1 - sr_cvar).rename("r2_pred")

    df_RETURN = pd.concat([df_cond_mean, sr_cvar, sr_r2_pred], axis=1) \
        .rename_axis("SNP", axis=0) \
        .reset_index("SNP", drop=False)

    if isinstance(df_ref_MAF, pd.DataFrame):

        df_ref_MAF = df_ref_MAF.drop(['MAF'], axis=1) # (2025.11.16.) MAF column때고 export

        df_RETURN = df_RETURN.merge(df_ref_MAF, on=['SNP'])


    return df_RETURN