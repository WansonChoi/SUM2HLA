# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import join, dirname, basename, exists
import numpy as np
import pandas as pd
import math
import importlib
import json

import src.SUM2HLA_PostCalc_Cov as SUM2HLA_PostCalc_Cov





def remove_MAF(_fpath_Z_imputed, _out_prefix_new):

    if not exists(_fpath_Z_imputed): return -1

    ##### (1) load Z_imputed
    df_Z_imputed = pd.read_csv(_fpath_Z_imputed, sep='\t', header=0) \
                        .drop(['MAF'], axis=1)

    df_Z_imputed.to_csv(_out_prefix_new + ".Z_imputed", sep='\t', header=True, index=False, na_rep="NA")

    return 0



def cp_bash(_fpath_target, _fpath_out):


    CMD = f"cp {_fpath_target} {_fpath_out}"

    r = os.system(CMD)

    return 0



def trim_rounds_up_to_3(d_json):
    """
    d_json: dict loaded from JSON file
    returns: dict that keeps only ROUND_1, ROUND_2, ROUND_3 if present
    """
    out = {}

    for k in ["ROUND_1", "ROUND_2", "ROUND_3"]:
        if k in d_json:
            out[k] = d_json[k]

    return out


def trim_rounds_up_to_3_from_file(fpath_in, fpath_out=None):
    with open(fpath_in, "r") as f:
        d = json.load(f)

    trimmed = trim_rounds_up_to_3(d)

    if fpath_out is not None:
        with open(fpath_out, "w") as f:
            json.dump(trimmed, f, indent=4)

    return trimmed



########## [1] UKB + FG (Deprecated)

## 그냥 싹 다 다시 돌림. SWCA가 'whole'기준으로 작업돼있어서 round 3까지 자르는게 의미가 없었음.

def recalc_LL_UKB_FG(_fpath_wholePP, _out_prefix_new):

    if not exists(_fpath_wholePP): return -1


    ##### (1) load 'whole.PP'
    df_PP_whole = pd.read_csv(_fpath_wholePP, sep='\t', header=0)


    ##### (2) regenerate
    d_temp_CODINGregion = SUM2HLA_PostCalc_Cov.postprepr_LL(df_PP_whole[['SNP', 'LL+Lprior']], _l_type=["AA+HLA"])
    # print(d_temp_CODINGregion)
    
    
    ##### (3) export
    d_temp = {}
    for j, (_key, _df_PP) in enumerate(d_temp_CODINGregion.items()):
        
        # print(f"=====[{j}]: {_key}")
        # print(_df_PP)

        OUT = _out_prefix_new + f".{_key}.PP"

        _df_PP.to_csv(OUT, sep='\t', header=True, index=False, na_rep="NA")


        d_temp[_key] = OUT


    # ToRETURN = d_temp["AA+HLA"] # 당장은 이렇게 하드코딩하듯히 하면 됨.

    return 0



def wrapper_UKB_FG(_df_SSFN, _out_dir="/data02/wschoi/_hCAVIAR_v2/202511120_toRepository"):

    sr_out_prefix_before = _df_SSFN['OUT_SUM2HLA_SPA_P_recalc']

    sr_wholePP = sr_out_prefix_before.map(lambda x: x + ".whole.PP")
    sr_Z_imputed = sr_out_prefix_before.map(lambda x: x + ".Z_imputed")
    sr_SWCA = sr_out_prefix_before.map(lambda x: x + ".r2pred0.6.ma.SWCA.dict")

    sr_flag = sr_wholePP.map(lambda x: exists(x)) & \
                sr_Z_imputed.map(lambda x: exists(x)) & \
                sr_SWCA.map(lambda x: exists(x))


    sr_out_prefix_after = sr_out_prefix_before.map(lambda x: basename(x).split(".")[-1]) \
                    .map(lambda x: join(_out_dir, f"UKB+FG.{x}")) \
                    .rename("OUT_regen_20251120")


    df_ToIter = pd.concat([sr_wholePP, sr_Z_imputed, sr_SWCA, sr_out_prefix_after, sr_flag], axis=1)


    count = 0

    for _index, _whole_PP, _Z_imputed, _SWCA, _out_prefix_after, _flag in df_ToIter.itertuples():

        if _flag:

            print(f"=====[{_index}]: {_out_prefix_after} / {_flag}")

            recalc_LL_UKB_FG(_whole_PP, _out_prefix_after)
            remove_MAF(_Z_imputed, _out_prefix_after)
            trim_rounds_up_to_3_from_file(_SWCA, _out_prefix_after + ".r2pred0.6.ma.SWCA.dict")

            count += 1

    print(count)

    df_RETURN = pd.concat([_df_SSFN, sr_out_prefix_after], axis=1)

    return df_RETURN


def cp_SWCA_UKB_FG(_df_SSFN, _out_dir="/data02/wschoi/_hCAVIAR_v2/202511120_toRepository"):

    df_ToIter = _df_SSFN[['OUT_SUM2HLA_SPA_P_recalc', 'OUT_regen_20251120']]

    for _index, _fpath_old, _fpath_new in df_ToIter.itertuples():

        print(f"=====[{_index}]: {_fpath_old} / {_fpath_new}")

        fpath_SWCA_old = _fpath_old + ".r2pred0.6.ma.SWCA.dict"
        fpath_SWCA_new = _fpath_new + ".r2pred0.6.ma.SWCA.dict"

        if exists(fpath_SWCA_old):

            cmd = f"cp {fpath_SWCA_old} {fpath_SWCA_new}"
            os.system(cmd)



    return 0





########## [2] MVP

def wrapper_MVP(_df_SSFN, _out_dir="/data02/wschoi/_hCAVIAR_v2/202511120_toRepository_MVP"):

    sr_out_prefix_before = _df_SSFN['out_prefix']

    sr_PP_AA_HLA = sr_out_prefix_before.map(lambda x: x + ".AA+HLA.PP")
    sr_Z_imputed = sr_out_prefix_before.map(lambda x: x + ".Z_imputed")
    sr_SWCA = sr_out_prefix_before.map(lambda x: x + ".r2pred0.6.ma.SWCA.dict")

    sr_flag = sr_PP_AA_HLA.map(lambda x: exists(x)) & \
                sr_Z_imputed.map(lambda x: exists(x)) & \
                sr_SWCA.map(lambda x: exists(x))


    """
    - MVP는 마지막에 basename을 이삳적으로 잘 만들어놔서 딱히 건드릴거 없을듯.
    """

    sr_out_prefix_after = sr_out_prefix_before \
                            .map(lambda x: basename(x)) \
                            .map(lambda x: join(_out_dir, x)) \
                            .rename("OUT_regen_20251120")


    df_ToIter = pd.concat([sr_PP_AA_HLA, sr_Z_imputed, sr_SWCA, sr_out_prefix_after, sr_flag], axis=1)

    count = 0

    for _index, _PP_AA_HLA, _Z_imputed, _SWCA, _out_prefix_after, _flag in df_ToIter.itertuples():

        if _flag:

            print(f"=====[{_index}]: {_out_prefix_after} / {_flag}")

            cp_bash(_PP_AA_HLA, _out_prefix_after + ".AA+HLA.PP")
            remove_MAF(_Z_imputed, _out_prefix_after)
            cp_bash(_SWCA, _out_prefix_after + ".r2pred0.6.ma.SWCA.dict")


            count += 1

    print(count)

    df_RETURN = pd.concat([_df_SSFN, sr_out_prefix_after], axis=1)

    return df_RETURN



########## [3] Consortium (Deprecated; 얘도 걍 싹 다시 돌림.)


def wrapper_Consortium(_df_SSFN, _out_dir="/data02/wschoi/_hCAVIAR_v2/202511120_toRepository"):

    """
    얘네들은 out_prefix_before를 우리가 만들어야 함.
    
    """

    sr_accession_id = _df_SSFN['GWAS catalog Study Accession (or URL)'].map(lambda x: "dbGaP" if not x.startswith("GCST") else x)
    sr_trait_acronym = _df_SSFN['Trait (Acronym)']

    df_temp = pd.concat([sr_accession_id, sr_trait_acronym], axis=1)

    sr_out_prefix_before = pd.Series(
        [f"Consortium.{_trait_acronym}.{_accession_id}" for _index, _accession_id, _trait_acronym in df_temp.itertuples()],
        index = _df_SSFN.index
    ) \
        .map(lambda x: join("/data02/wschoi/_hCAVIAR_v2/20251008_reanalyses", x))

    sr_out_prefix_after = pd.Series(
        [f"Consortium-scale.{_accession_id}.{_trait_acronym}" for _index, _accession_id, _trait_acronym in df_temp.itertuples()],
        index = _df_SSFN.index
    ) \
        .map(lambda x: join(_out_dir, x)) \
        .rename("OUT_regen_20251120")


    sr_PP_AA_HLA = sr_out_prefix_before.map(lambda x: x + ".AA+HLA.PP")
    sr_Z_imputed = sr_out_prefix_before.map(lambda x: x + ".Z_imputed")
    sr_SWCA = sr_out_prefix_before.map(lambda x: x + ".r2pred0.6.ma.SWCA.dict")

    sr_flag = sr_PP_AA_HLA.map(lambda x: exists(x)) & \
                sr_Z_imputed.map(lambda x: exists(x)) & \
                sr_SWCA.map(lambda x: exists(x))



    df_ToIter = pd.concat([sr_PP_AA_HLA, sr_Z_imputed, sr_SWCA, sr_out_prefix_after, sr_flag], axis=1)
    # print(df_ToIter)

    count = 0

    for _index, _PP_AA_HLA, _Z_imputed, _SWCA, _out_prefix_after, _flag in df_ToIter.itertuples():

        if _flag:

            print(f"=====[{_index}]: {_out_prefix_after} / {_flag}")

            cp_bash(_PP_AA_HLA, _out_prefix_after + ".AA+HLA.PP")
            remove_MAF(_Z_imputed, _out_prefix_after)
            cp_bash(_SWCA, _out_prefix_after + ".r2pred0.6.ma.SWCA.dict")


            count += 1

    print(count)

    df_RETURN = pd.concat([_df_SSFN, sr_out_prefix_after], axis=1)

    return df_RETURN



########## UKB+FG supple만들기 (Supplementary Table 4)

## 이게 community detection section을 위한 ST인데, 결과적으로 genotype-based랑 겹치긴함.
## 여튼 주요하게 활용한 UKB+FG의 SSFN table을 활용하면 됨.

def make_SupplementaryTable4(_fpath_SSFN_UKB_FG="/data02/wschoi/_hCAVIAR_v2/20250616_UKBB_BBJ_SUM2HLA_v2/UKB+FG.EUR.v10.T153.Diseases.regen_20251120.ssfn"):

    df_SSFN_UKB_FG_2 = pd.read_csv(
        _fpath_SSFN_UKB_FG, sep='\t', header=0
    )
    print(df_SSFN_UKB_FG_2.columns)
    idx_ToExclude = df_SSFN_UKB_FG_2.columns.tolist().index('ToSWCA')
    print(idx_ToExclude)



    f_hasSignal = df_SSFN_UKB_FG_2['OUT_regen_20251120'].map(lambda x: os.path.exists(x + ".AA+HLA.PP")).rename("hasSignal (in T1DGC SNPs)")
    print(f_hasSignal.value_counts())


    df_SSFN_UKB_FG_3 = pd.concat([f_hasSignal, df_SSFN_UKB_FG_2], axis=1)

    ### True가 먼저오게.
    df_SSFN_UKB_FG_3 = df_SSFN_UKB_FG_3.sort_values("hasSignal (in T1DGC SNPs)", ascending=False)

    ### 우리서버에서 쓰는 columns들 제외
    df_SSFN_UKB_FG_3 = df_SSFN_UKB_FG_3.iloc[:, :idx_ToExclude]

    ## 다음 번에 "Trait Type"이랑 "Nealelab ss" 이 두 columns들 빼야함.
    ## "N_eff_UKB", "N_eff_FG"


    return df_SSFN_UKB_FG_3