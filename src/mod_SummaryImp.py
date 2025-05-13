import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd

import math

import src.mod_Util as mod_Util


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
    
    # display(_df_X1_obs_2)
    # display(_df_LD_11)
    
    """
    이렇게 display해서 `.loc[_df_LD_11.index, :]` 이걸로 확실히 match되는거 확인함. (2025.02.18.)
    """
    
    
    ##### (2) calc conditional mean and covariance
    
    arr_LD_11_inv = np.linalg.inv(_df_LD_11)

    conditional_mean = _df_LD_21.values @ arr_LD_11_inv @ _df_X1_obs_2['Z'].values # mu1 and mu2는 0이라서 제외함.
    conditional_mean = pd.DataFrame({"Conditional_mean": conditional_mean}, index=_df_LD_22.index)
    
    conditional_cov = _df_LD_22.values - _df_LD_21.values @ arr_LD_11_inv @ _df_LD_12.values
    conditional_cov = pd.DataFrame(conditional_cov, columns=_df_LD_22.columns, index=_df_LD_22.index)
    
    
    return conditional_mean, conditional_cov



########## (4) 시마이 하기
## 정답이 있는 WTCCC 데이터 같은거랑만 활용하는 run하는 함수임. 
## 사실상, 결과 확인 && 분석 용.
def wrap_up_imputation_with_answer(_df_answer, _df_FRQ, _df_cond_mean_imputed, _df_cond_cov, _maf=None, _how='left'):
    
    # display(_df_answer)
    
    ##### Answer and MAF
    df_answer_2 = _df_answer.merge(
        _df_FRQ.drop(['NCHROBS', 'A1'], axis=1), on=['CHR', 'SNP']
    )
    # print(df_answer_2.shape)
    
    if isinstance(_maf, float) and _maf < 1.0:
        f_maf = (df_answer_2['MAF'] < _maf) | (df_answer_2['MAF'] > 1 - _maf)
        df_answer_2 = df_answer_2[~f_maf] # subset
    
    # display(df_answer_2)

    ##### Imputed Z-scores
    _df_cond_mean_imputed_2 = _df_cond_mean_imputed \
                                .rename_axis("SNP", axis=0) \
                                .reset_index(drop=False) \
                                .rename({"Conditional_mean": "Z_imputed"}, axis=1)
    
    
    ##### Imputed P-value
    sr_P_imputed = _df_cond_mean_imputed_2['Z_imputed'] \
                    .abs() \
                    .map(lambda x: 2 * sp.stats.norm.cdf(-x)) \
                    .rename("P_imputed")
    
    sr_P_imputed = pd.concat([_df_cond_mean_imputed_2['SNP'], sr_P_imputed], axis=1)
    
    
    ##### The diagonal elements of the condtional covariance => SE
    sr_cov_diagonal = pd.Series(
        np.sqrt([abs(_) if abs(_) < 1e-3 else _ for _ in _df_cond_cov.values.diagonal()]), 
        name="SE(Z_imputed)"
    )
    ## 결국 1e-3은 음수인 애들을 filtering하고 싶은거임.
    
    sr_cov_diagonal = pd.concat(
        [_df_cond_mean_imputed_2['SNP'], sr_cov_diagonal], axis=1
    )
    # display(sr_cov_diagonal)
    
    
    ##### Left_join
    
    df_mg0 = df_answer_2.merge(_df_cond_mean_imputed_2, on='SNP', how=_how) 
        # COJO input만들려면 어차피 'left' 필요함.
        # 특히, SNP들도 MAF 붙여와야 함.
        
    # display(df_mg0)
    
    df_mg1 = df_mg0 \
                .merge(sr_cov_diagonal, on='SNP', how='left') \
                .merge(sr_P_imputed, on='SNP', how='left')
    # display(df_mg1)
        
    return df_mg1, _df_cond_cov



########## Main (1 ~ 4)

def __MAIN__(_fpath_ss_matched, _fpath_ref_LD, _fpath_MAF=None, _fpath_answer=None):
    
    ##### load data
    df_ss_matched = pd.read_csv(_fpath_ss_matched, sep='\s+', header=0) \
                        if isinstance(_fpath_ss_matched, str) else _fpath_ss_matched
    
    if df_ss_matched['Z_fixed'].isna().any(): # 사실상 CD를 위한 예외처리
        # display(df_ss_matched)
        df_ss_matched.dropna(subset=["Z_fixed"], axis=0, inplace=True) 
    
    df_LD_PSD = pd.read_csv(_fpath_ref_LD, sep='\t', header=0) \
                    if isinstance(_fpath_ref_LD, str) else _fpath_ref_LD
    df_LD_PSD.index = df_LD_PSD.columns
    
    
    ##### (1) LD panel의 marker set을 SNP + HLA로 나누기
    sr_LD_SNP_markers, sr_LD_HLA_markers = \
        partition_LD_marker_set_v2(df_ss_matched['SNP_LD'], df_LD_PSD.columns.to_series().reset_index(drop=True))

    
    ##### (2) LD를 parition한 SNP + HLA set으로 나누기 => 4 marices
    LD_partitioned = partition_LD_matrix(df_LD_PSD, sr_LD_SNP_markers, sr_LD_HLA_markers)    

    
    ##### (3) Conditional mean과 covariance계산하기 (main; the summary imputation)
    imputed_result = calc_conditional_mean_cov(
        df_ss_matched[['SNP_LD', 'Z_fixed']],
        *LD_partitioned
    )

    
    if isinstance(_fpath_answer, str) and os.path.exists(_fpath_answer) and os.path.exists(_fpath_MAF):
        
        """
        - 당장 MAF로 filtering하는 것도 여기서 안하려고.
        - imputed된 애들 일단 그대로 return. 
        """
                
        df_answer = pd.read_csv(_fpath_answer, sep='\s+', header=0)
        df_MAF = pd.read_csv(_fpath_MAF, sep='\s+', header=0)

        imputed_result_2 = wrap_up_imputation_with_answer(df_answer, df_MAF, *imputed_result)        
        
        return imputed_result_2
    
    else:

        """
        - 그냥 그대로 return할 거임, 정답과 함께 결과 분석할게 아니면.
        - 걍 imputed Z-score만 있으면 됨.
        - SE도 필요없음, COJO 파이프라인에서 활용할 때는.
            - 이런 경우, conditioned-cov는 걍 메모리만 쳐먹음, practically.
        
        """
        
        return imputed_result
    
    
    

    
    


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



# def plot_Scatter(_df_imputed_result_2, _figsize=(12, 6), _suptitle=""):
    
#     fig, axes = plt.subplots(1, 2, figsize=_figsize)

#     fig.suptitle(_suptitle)
    
#     ##### (1) Z-score comparision
    
#     _ax = axes[0]
    
#     _df_imputed_result_2.rename({"STAT": "Z_answer"}, axis=1).plot.scatter(x='Z_answer', y='Z_imputed', ax=_ax)

#     _ax.spines['top'].set_visible(False)
#     _ax.spines['right'].set_visible(False)
    
    
#     ##### (2) P-value comparision
    
#     _ax = axes[1]
    
#     _df_imputed_result_2[['P', 'P_imputed']].rename({"P": "P_answer"}, axis=1) \
#         .applymap(lambda x: -np.log10(x)) \
#         .plot.scatter(x="P_answer", y="P_imputed", ax=_ax)

#     _ax.spines['top'].set_visible(False)
#     _ax.spines['right'].set_visible(False)
    
    
    
#     return fig



