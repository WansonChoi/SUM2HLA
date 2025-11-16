"""
- 20251015_MVP_whole.ipynb
- /data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/
- 무조건 여기에 기록을 남겨놔야지, 안그러면 기억도 안나고 또 방황할게 뻔함...
"""


# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import basename, dirname, join
import numpy as np
import scipy as sp
import pandas as pd
import json
# import statsmodels.api as sm
import math

import subprocess

from shutil import which

from datetime import datetime

from importlib import reload

import src.INPUT_Raw_GWAS_summary as INPUT_Raw_GWAS_summary
from concurrent.futures import ProcessPoolExecutor
import src.Util as Util


#################### [ MVP ] ####################

"""
(1) MVP_supple과 GWAS_catalog_export와 PheCode를 바탕으로 inner_join함.

- `filter_each_ethnicgroup()` => GWAS_catalog_export를 filtering (N=4331)
    - 일단 meta-analyzed된 mixed ethnicities들의 summaries들 제외
    - 이렇게 하면 each ethnic group의 summaries들의 개수는 다음과 같음:
        - European: 1669
        - African American or Afro-Caribbean: 1368
        - Hispanic or Latin American: 1053
        - East Asian: 241
    - '/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/'에 exported함.
    - 여기서부터 EUR만 초점 맞췄고, EUR에서도 PheCode있는 애들만 남김. (N=1238)
        - (나중에 다른 ethnicity는 export해놓은것들부터 시작하면 됨.)
- MVP supple 과 inner_join하기
    - MVP_supple를 load할때 (1) PheCode있고 (2) EUR만 남김. (N=1847 => 1844)
- PheCode 몇 가지 하드코딩하고, inner_join해서 export함.
    - `transform_PheCode()`
        - `if _x == "Phe_041_8": return "041.8"`
    - .replace("159) (PheCode 159", "159") (정규표현식 처리하기 귀찮아서.)
    - 쨌든 위 두 개만 hard-exception처리해서 inner_join했고, the GWAS_catalog_export의 N=1238개 다 살림.

- 위 사항들을 당장 여기로 export해오진 않을거임. 필요하면 보셈.

(2) 전처리 한번 쭉 수행함.
- Pneumonia하나 찐빠남. (GCST90476022)

wget -c -r -nH --cut-dirs=6 --no-parent --reject index.html* \
        --exclude-directories=/pub/databases/gwas/summary_statistics/GCST90476001-GCST90477000/GCST90476022/harmonised/ \
        ftp://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90476001-GCST90477000/GCST90476022 \
        -P GCST90476022

- Pneumonia가 계속 download에러남. 서버쪽 문제인듯?
        
"""


def get_top_SNP(_fpath_sumstats0):

    _SNP, _BP, _P, _hasSignal = None, None, None, None


    if not os.path.exists(_fpath_sumstats0):
        _SNP = np.nan
        _BP = np.nan
        _P = np.nan
        _hasSignal = np.nan

        return _SNP, _BP, _P, _hasSignal


    df_sumstats0 = pd.read_csv(_fpath_sumstats0, sep='\t', header=0).sort_values("P")

    df_sumstats0 = df_sumstats0.iloc[0, :]

    _SNP = df_sumstats0['SNP']
    _BP = df_sumstats0['BP']
    _P = df_sumstats0['P']

    # _SNP, _BP, _P = df_sumstats0.tolist()
    _hasSignal = _P < 5e-8


    """
    - 최종 아래와 같이 transform해서 썼음.

    df_temp = pd.DataFrame(
        
        [list(_) for _ in df_MVP_PheCode_4['sumstats0'].map(lambda x: _20251016_MVP_UKBBwgs.get_top_SNP(x))],
        columns=['SNP_top', 'BP_top', 'P_top', 'hasSignal']
    )
    display(df_temp)    
    
    """


    return _SNP, _BP, _P, _hasSignal



"""
- 밑에 5개의 함수는 GPU server에서 쓴 함수임.
    - 20251015_MVP_whole_EUR_run.ipynb

"""

def load_and_prepr_SSFN(_fpath_SSFN_ToRun="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/IJOIN.PheCode_T1238.EUR.4.xlsx"):

    """
    - load하고, 전처리하고, 이런거 기록 남기면서 하려고 일부러 이렇게 작업.
    """

    df_SSFN_MVP_EUR = pd.read_excel(
        _fpath_SSFN_ToRun, header=0
    )
    print(df_SSFN_MVP_EUR.shape)


    ### (1) 여기서 summary가 available하지 않는 Pneumonia하나만 제외
    f_Pneumonia = ~df_SSFN_MVP_EUR['success_round1']



    ### (2) GWAS signal in the shared SNPs between T1DGC만 남기기.

    ## P < 5E-08이 없는 애들은 진짜 HLA가 causal이라 가정하기 힘든 diseases들이 대부분이었음. (ex. Exophthalmos - 안구돌출증)
    ## 우연인지모르겠지만 signal 있는애들은 autoimmune, immune-related, skin disease, diabetes관련된 애들이 대부분이긴 하네.
    f_hasSignal = df_SSFN_MVP_EUR['hasSignal']


    f_ToExtract = ~f_Pneumonia & f_hasSignal


    df_SSFN_MVP_EUR = df_SSFN_MVP_EUR[f_ToExtract]

    return df_SSFN_MVP_EUR



def prepare_output_label(_sr_traitname):

    ## trait 이름을 바로 보기 편하게 만들긴 해야함.


    def replace_special_characters(_x):


        new_label = _x \
                .replace(",", " ").replace(".", " ") \
                .replace("(", " ").replace(")", " ") \
                .replace("[", " ").replace("]", " ") \
                .replace("'", " ").replace("/", " ").replace("&", " ")

        new_label = re.sub(r'\s+', '_', new_label)

        return new_label

    sr_RETURN = _sr_traitname.map(lambda x: replace_special_characters(x)).rename("Trait_Description_2")

    return sr_RETURN



def prepare_output_prefix(_df_SSFN, _out_dir="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR_run"):

    sr_Trait_Description_2 = prepare_output_label(_df_SSFN['Trait_Description'])

    df_temp = pd.concat([_df_SSFN['accessionId'], sr_Trait_Description_2], axis=1)

    sr_OUT = pd.Series(
        [f"{_out_dir}/MVP.{_accessionId}.{_Trait_Description_2}" for _index, _accessionId, _Trait_Description_2 in df_temp.itertuples()],
        index=_df_SSFN.index,
        name='out_prefix'
    )

    # sr_OUT = sr_Trait_Description_2.map(lambda x: f"{_out_dir}/MVP.{x}").rename("out_prefix")

    return sr_OUT



def alloc_GPU(_df_SSFN):

    l_GPU_ToUse = [0, 3, 5, 7]
    
    d_GPU_ToMap = {
        0: 2, # 2로 잡아야 0이 나옴.
        3: 1, # 1로 잡아야 3이 나옴.
    }    


    # 원하는 chunk 크기
    # chunksize = 32
    chunksize = math.ceil(_df_SSFN.shape[0] / len(l_GPU_ToUse))
    n = len(_df_SSFN)
    gpus = l_GPU_ToUse  # 예: [0, 3, 5, 7]

    # 앞 (len(gpus)-1)개는 chunksize로 채우고, 마지막은 나머지
    full = min(len(gpus) - 1, n // chunksize)
    counts = [chunksize] * full                      # 앞쪽 full개
    remaining = n - chunksize * full
    # 만약 full이 (len(gpus)-1)보다 작다면 사이에 0 크기의 chunk가 생길 수 있음
    # (보통은 chunksize를 적절히 잡으면 문제 없음)
    if len(gpus) - 1 - full > 0:
        counts += [0] * (len(gpus) - 1 - full)
    counts.append(remaining)                         # 마지막 GPU에 남은 개수

    # 예: n=126, chunksize=32, gpus=4 -> counts = [32, 32, 32, 30]

    # counts에 따라 GPU ID 나열
    gpu_seq = np.repeat(gpus, counts)

    # Series로 붙이기 (네가 원하던 index 유지)
    sr_GPU_ID = pd.Series(gpu_seq, index=_df_SSFN.index, name="gpu_id")

    # 필요 시 GPU ID 매핑 적용
    sr_GPU_ID = sr_GPU_ID.map(lambda x: d_GPU_ToMap.get(x, x))

    return sr_GPU_ID



def run_MVP_SUM2HLA(_args):

    _index, _accessionId, _Trait_Description, _sumstats0, _out_prefix, _gpu_id = _args

    print(f"=====[{_index}]: _accessionId: {_accessionId} / _Trait_Description: {_Trait_Description}")

    cmd = [
        "conda", "run", "-n", "jax_gpu", # 'jax_gpu' 환경에서 실행하도록 지정
        "python",
        "SUM2HLA.py",
        "--sumstats", _sumstats0,
        "--ref", "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "--out", _out_prefix,
        "--gpu-id", str(_gpu_id)
    ]

    # print(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    
    if result.returncode != 0:
        # 에러가 발생한 경우, 에러 메시지를 출력
        return False
    return True



"""
- MVP diseases들 GPU 서버에서 돌리고 난 후, 이제 결과 갈무리는 다음에서 함.
    - 20251015_MVP_whole_EUR_run_2.ipynb


- 빼먹었더 5개 추가해서 PheCode만 있는 T=131짜리로 다시. (2025.10.23.)
- 사실상 이게 Supplementary Table만드는거임.

"""

def load_SSFN_after_GPUrun(_fpath_SSFN="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/20251023.RJOIN.PheCode_T1670.EUR.v5.PheCode_T1245.txt"):

    """
    - v5에는 이미 아래 제외한 T=126에 대한 items들만 있음.
        - (1) Pneumonia제외했고, (2) T1DGC와 shared SNPs들에서 signal있는 애들만 돌렸음. (T=126)
    - v4까지 T=1238임.

    - 예전에는 이걸로 load했음. 
        - "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/IJOIN.PheCode_T1238.EUR.5.GPU_run.txt"
    
    """

    # df_SSFN = pd.read_excel(_fpath_SSFN, header=0)
    df_SSFN = pd.read_csv(_fpath_SSFN, header=0, sep='\t')

    ## supplementary table을 만들기 위한 column subset을 여기서 해야겠음.
    l_ToExclude = [
        "reportedTrait_2", "Ethnicity", "firstAuthor",
    ]

    l_ToExclude += "publicationDate	journal	title".split()

    l_ToExclude += "reportedTrait	efoTraits	bgTraits	discoverySampleAncestry	replicationSampleAncestry	associationCount		ssApiFlag	agreedToCc0	genotypingTechnologies	pubmedId	initialSampleDescription	replicateSampleDescription	fpath_ss_raw	sumstats0	already_done".split()
    l_ToExclude += ['gpu_id']

    df_SSFN = df_SSFN.drop(l_ToExclude, axis=1, errors='ignore')



    return df_SSFN



def get_top_Credible99(_x):

    """
    - 만들고보니 그렇게 쓸모있지는 않네.
    
    """

    l_ToExtract = ['rank', 'SNP', 'PP', 'CredibleSet(99%)']

    df_PP_AA_HLA = pd.read_csv(_x, sep='\t', header=0, usecols=l_ToExtract)

    f_credible99 = df_PP_AA_HLA['CredibleSet(99%)']

    df_RETURN = df_PP_AA_HLA[f_credible99]

    return df_RETURN



def get_top_PP(_x, _f_flatten=True):

    # l_ToExtract = ['rank', 'SNP', 'PP', 'CredibleSet(99%)']
    l_ToExtract = ['SNP', 'PP']

    df_PP_AA_HLA = pd.read_csv(_x, sep='\t', header=0, usecols=l_ToExtract)
        # PP로 sorted되었다 가정.

    ## 당장 CLE 공동 1등 한 애들이 있음.
    PP_top = df_PP_AA_HLA['PP'].iat[0]
    f_co_top = df_PP_AA_HLA['PP'].map(lambda x: abs(x - PP_top) < 1e-3) # 차이가 없으면 같은거.


    df_RETURN = df_PP_AA_HLA[f_co_top]

    if df_RETURN.shape[0] == 1: 
        return df_RETURN
    else:
        df_RETURN_flatten = df_RETURN.iloc[[0], :].copy()

        str_flatten = ','.join(df_RETURN['SNP'].tolist())

        df_RETURN_flatten['SNP'].iat[0] = str_flatten

        return df_RETURN_flatten



def get_top_Pvalue(_fpath_Z_mputed, _top_HLA_variants):

    """
    - imputed P-value도 필요함, 이제.
    - imputed z-score도 필요함 (2025.11.04.)

    """

    _top_HLA_variants = _top_HLA_variants.split(',')

    l_ToExtract = ['SNP', 'Conditional_mean', 'r2_pred']
    
    df_Z_imputed = pd.read_csv(_fpath_Z_mputed, sep='\t', header=0, usecols=l_ToExtract, index_col=['SNP'])


    ## 관심있는 top HLA variants들로 subset.
    df_Z_imputed = df_Z_imputed.loc[_top_HLA_variants, :]

    sr_P_imputed = pd.Series(
        sp.stats.norm.cdf(-df_Z_imputed['Conditional_mean'].abs()) * 2,
        index = df_Z_imputed.index,
        name='P'
    )

    sr_Z_imputed = df_Z_imputed['Conditional_mean'].map(lambda x: str(x))
    sr_P_imputed_2 = sr_P_imputed.map(lambda x: f"{x:.1E}")
    sr_r2pred = df_Z_imputed['r2_pred'].map(lambda x: f"{x:.2f}")

    str_Z_imputed = ','.join(sr_Z_imputed.tolist())
    str_P_imputed_2 = ','.join(sr_P_imputed_2.tolist())
    str_r2pred = ','.join(sr_r2pred.tolist())

    return str_P_imputed_2, str_Z_imputed, str_r2pred



def concat_SUM2HLA_top(_df_SSFN, _func=get_top_PP, _l_ToExtract = ['accessionId', 'Trait_Description', 'out_prefix']):

    df_ToIter = _df_SSFN[_l_ToExtract]


    df_RETURN = pd.concat(
        {(_accessionId, _Trait_Description): _func(_out_prefix + ".AA+HLA.PP") \
            for _index, _accessionId, _Trait_Description, _out_prefix in df_ToIter.itertuples()},
        names=['accessionId', 'Trait_Description', 'index']
    ) \
        .reset_index(['accessionId', 'Trait_Description'], drop=False) \
        .reset_index('index', drop=True) \
        .rename({"SNP": "Top HLA Variant (by PP)"}, axis=1)

    return df_RETURN


def concat_SUM2HLA_top_v2(_df_SSFN_whole, _func=get_top_PP, _l_ToExtract = ['accessionId', 'Trait_Description', 'out_prefix']):

    ## 할일이 많아져서 그냥 하나로 묶어버림.


    ## (1) hasSignal == SUM2HLA 성공 으로 해서 나누기.
    _df_SSFN_hasSignal = _df_SSFN_whole[ _df_SSFN_whole['hasSignal'] ].copy()
    _df_SSFN_NoSignal = _df_SSFN_whole[ ~_df_SSFN_whole['hasSignal'] ].copy()

    # print(_df_SSFN_hasSignal)
    # print(_df_SSFN_NoSignal)

    ## (2) PP 준비
    df_ToIter = _df_SSFN_hasSignal[_l_ToExtract]

    df_top_PP = pd.concat(
        {(_accessionId, _Trait_Description): _func(_out_prefix + ".AA+HLA.PP") \
            for _index, _accessionId, _Trait_Description, _out_prefix in df_ToIter.itertuples()},
        names=['accessionId', 'Trait_Description', 'index']
    ) \
        .reset_index(['accessionId', 'Trait_Description'], drop=False) \
        .reset_index('index', drop=True) \
        .rename({"SNP": "Top HLA Variant (by PP)"}, axis=1) \
        .drop(['accessionId', 'Trait_Description'], axis=1) # 이제는 필요없음.
    
    df_top_PP.index = _df_SSFN_hasSignal.index
    
    # print(df_top_PP)


    ## (3) P-value 준비 (여기만 z-score도 가져울 수 있게 고치면 되겠네; 2025.11.04.)
    df_Pval = pd.concat([_df_SSFN_hasSignal['out_prefix'], df_top_PP['Top HLA Variant (by PP)']], axis=1)
    # print(df_Pval)


    df_Pval = pd.DataFrame(
        [get_top_Pvalue(_out_prefix + ".Z_imputed", _top_HLA_variant) for _index, _out_prefix, _top_HLA_variant in df_Pval.itertuples()],
        columns=['P-value (imputed)', 'Z-score (imputed)', 'r2_pred'],
        index = _df_SSFN_hasSignal.index
    )
    # print(df_Pval)



    ## (4) 갈무리 전 sorting && 갈무리
    _df_SSFN_hasSignal = pd.concat([_df_SSFN_hasSignal, df_top_PP, df_Pval], axis=1)
    _df_SSFN_hasSignal = _df_SSFN_hasSignal.sort_values(
        ['Trait_Type', 'P_top'], ascending=[True, True]
    )
    # print(_df_SSFN_hasSignal)

    _df_SSFN_NoSignal = _df_SSFN_NoSignal.sort_values("P_top")

    df_RETURN = pd.concat([_df_SSFN_hasSignal, _df_SSFN_NoSignal]) \
                    .drop(['out_prefix'], axis=1)


    return df_RETURN



"""
- 초반에 MVP의 GWAS_catalog_export와 Supple간에 inner_join을 좀 잘못한게 있음.
- 전자를 EUR로 남기고 시작해서 아마 ethnicity의 문제는 크게 없을거임.
- 근데 Multiple scelrosis나 Parkinson's dieases이런애들 몇개 빠졌을 거라서, 얘네들 챙기자.
    - 원인은 GWAS_catalog_export에서 하필 PheCode가 몇개 안 적혀있어서 문제임.
"""


def MVP_load_GWAScatalog_export(_fpath_EUR="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/PMID39024449_studies_export.EUR.tsv"):

    """
    - 일단 Each ethnic group별로 나눠놓은 것 부터 시작.
        - 앞서 meta된 애들 제외한거랑, each ethnic group으로 나눠놓은것까지는 ok임.
        - 여기에 문제가 없다 가정하면 확실히 EUR만 남긴게 맞음.
    
    """

    

    df_GWAScatalog_export = pd.read_csv(_fpath_EUR, sep='\t', header=0)


    return df_GWAScatalog_export



def MVP_load_supple(_fpath_EUR_supple="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/science.adj1182_tables_s2.xlsx"):

    """
    - "Trait" column이 Phecode가지고 있는 애들만 남기기
        - 이건 문제없음. Multiple sclerosis만 하더라도 어차피 PheCode있는 애로 해야지 잘 나옴.
    - EUR을 제외한 나머지 ethnic group에서 모두 sample size가 NA여야 함.
        - 이 부분이 이번에 수정해야하는 부분.
    
    """

    df_MVP_supple = pd.read_excel(_fpath_EUR_supple, header=0)

    ## PheCode flag
    f_PheCode = df_MVP_supple['Trait'].str.startswith("Phe_")
    df_MVP_supple = df_MVP_supple[f_PheCode]

    # ## EUR만 남기기 <=> 'Case_EUR', 'Control_EUR' 이 두 column만 NA가 아닌 rows들만 남기기
    # f_Case_AFR = df_MVP_supple["Case_AFR"].isna()
    # f_Control_AFR = df_MVP_supple["Control_AFR"].isna()
    # f_N_AFR = df_MVP_supple["N_AFR"].isna()

    # f_Case_AMR = df_MVP_supple["Case_AMR"].isna()
    # f_Control_AMR = df_MVP_supple["Control_AMR"].isna()
    # f_N_AMR = df_MVP_supple["N_AMR"].isna()

    # f_Case_EAS = df_MVP_supple["Case_EAS"].isna()
    # f_Control_EAS = df_MVP_supple["Control_EAS"].isna()
    # f_N_EAS = df_MVP_supple["N_EAS"].isna()

    # ## 얘네들민 NOT NA여야 함.
    f_Case_EUR = ~df_MVP_supple["Case_EUR"].isna()
    f_Control_EUR = ~df_MVP_supple["Control_EUR"].isna()
    f_N_EUR = ~df_MVP_supple["N_EUR"].isna()

    # f_EUR = (f_Case_AFR & f_Control_AFR & f_N_AFR) & (f_Case_AMR & f_Control_AMR & f_N_AMR) & (f_Case_EAS & f_Control_EAS & f_N_EAS) & \
    #             (f_Case_EUR & f_Control_EUR & f_N_EUR)

    f_EUR = f_Case_EUR & f_Control_EUR & f_N_EUR
    

    ## 저번에 한게 맞았음. EUR을 제외한 ethnic group에서 sample size가 나타난다고 해서 meta-analyzed된 summary만 제공된다는 뜻이 아님.
    ## (ex) CLE => supple상에서는 AFR, EUR있음 => (1) AFR, (2) EUR, and (3) AFR + EUR 이렇게 나가는거임.


    df_MVP_supple = df_MVP_supple[f_EUR]



    return df_MVP_supple



def MVP_inner_join(_df_GWAScatalog, _df_supple):

    _how = "right" # 걍 inner/outer말고 right로 함.

    _df_GWAScatalog = _df_GWAScatalog.copy()
    _df_supple = _df_supple.copy()

    ## supple에서 필요한 column만 캐서 left로
    l_ToExtract = ['Trait', 'Trait_Description', 'Trait_Type', 'Trait_Category', 'Case_EUR', 'Control_EUR',	'N_EUR']
    _df_supple = _df_supple[l_ToExtract]


    ## (PheCode XX) 이것만 때기
    p_PheCode = re.compile(r"(?:\s*\(PheCode [\d.]+\))+\s*$")
    sr_temp = _df_GWAScatalog['reportedTrait'].map(lambda x: p_PheCode.sub("", x)).rename("reportedTrait_2")

    _df_GWAScatalog = pd.concat([sr_temp, _df_GWAScatalog], axis=1)

    df_mg0 = _df_supple.merge(_df_GWAScatalog, left_on='Trait_Description', right_on='reportedTrait_2', how=_how)


    return df_mg0



def MVP_prepr_for_BatchRun(_df_right_join, _out_dir_sumstats0 = "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR"):

    _df_right_join = _df_right_join.copy()

    ## accession ID duplicated된 애들 있음: "Other tests" <= 필요없음. 이거 죽이고 시작.
    f_ACCID_dup = _df_right_join['Trait_Description'] == 'Other tests'
    _df_right_join = _df_right_join[~f_ACCID_dup]


    ## Pneumonia빼기. (저번에 안받아짐.)
    f_Pneumonia = _df_right_join['Trait_Description'] == 'Pneumonia'
    _df_right_join = _df_right_join[~f_Pneumonia]


    ## fpath_ss_raw달기
    def bring_ss_fpath(_accessionid, _out_dir="/data02/MVP_MillionVeteranProgram_EUR"):
        
        out_dir_temp = os.path.join(_out_dir, _accessionid)
        ss_fpath = os.path.join(out_dir_temp, f"{_accessionid}.tsv.gz")
        # print(out_dir_temp)
        # print(ss_fpath)
        
        if os.path.exists(ss_fpath):
            return ss_fpath
        else:
            return np.nan
        
    # bring_ss_fpath("GCST90479320")

    sr_fpath_ss_raw = _df_right_join['accessionId'].map(lambda x: bring_ss_fpath(x)).rename("fpath_ss_raw")

    _df_right_join = pd.concat([_df_right_join, sr_fpath_ss_raw], axis=1)


    ## sumstats0 준비 && 기존에 했던 애들 mask하기 (iteration돌때 피하려고)
    sr_sumstats0 = _df_right_join['accessionId'].map(lambda x: os.path.join(_out_dir_sumstats0, f"MVP.EUR.{x}.sumstats0")).rename("sumstats0")
    sr_sumstats0_done = sr_sumstats0.map(lambda x: os.path.exists(x)).rename("already_done")

    _df_right_join = pd.concat([_df_right_join, sr_sumstats0, sr_sumstats0_done], axis=1)



    return _df_right_join


"""
- 여기서 전처리 iteration돌리고 저번에 안한 애들에 한정해서 sumstats0들을 만듬
- SSFN v3을 export해서 manually 수정좀 하고 옴.
    - Glaucoma (N=315668 European)이 N=430721 이랑 match됨. (전자가 그 PheCode없이 sample size좀 더 적은거임.)
    - 다음애들은 PheCode있는데 찾질 못함.
        - Screening for malignant neoplasms
        - Screening for other diseases and disorders
        - Dermatophytosis



)

"""


## 걍 위에 'load_SSFN_after_GPUrun()' 수정해서 씀.

def MVP_load_and_prepr_SSFN_v2(_fpath_SSFN = "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/20251023.RJOIN.PheCode_T1670.EUR.v4.xlsx"):

    ## GPU run 할때 load한거. 여기서 Phecode만 남겼다?

    df_SSFN = pd.read_excel(_fpath_SSFN, header=0)

    ## PheCode
    f_hasPheCode = ~df_SSFN['Trait'].isna()
    df_SSFN = df_SSFN[f_hasPheCode]


    ## has Signals
    # f_hasSignal = df_SSFN["hasSignal"]
    # df_SSFN = df_SSFN[f_hasSignal]


    return df_SSFN




#################### [ UKBB_WGS ] ####################

"""
- UKBB_WGS도 그냥 여기서
- Supple: 41586_2025_9272_MOESM11_ESM.xlsx (N=1632)
    - Supplementary Table 19 - GWAS Catalog GCST list.
    - 얘네들은 GWAS catalog accessio_ID를 supple에서 줬음. Inner_join하기 편할듯?
- GWAS_catalog_export: PMID40770095_studies_export.tsv (N=1632)
- 얘네들은 아예 처음부터 개수도 완벽하게 match되, .isin()했을때 다 있음. 그냥 GWAS_catalog_export로 놀면 됨.


- 저번에 ethnic_group 나눴었는데, 그냥 한번만 얼른 더하자. (찾는게 더 귀찮다)
- 나눴고, exported함.
    - /data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole
    - UKBBwgs_PMID40770095_studies_export.{EUR,SAS,EAS,ASJ,AFR}.tsv

- 여기서부터는 EUR만, (N=834밖에 안되네)
    - ICD disease: 763 / Non-disease (biomarker): 19
    - The 19 biomarker traits들은 모두 "UKB data field"가 있음.
"""



def get_ethnicity(_df):

    sr_discoverySampleAncestry = _df['discoverySampleAncestry']

    sr_RETURN = sr_discoverySampleAncestry.map(lambda x: ' '.join(x.split()[1:])).rename("Ethnicity")

    """
    확실히 이렇게 해도 문제 없음.

    Ethnicity
    European              834
    SouthAsian            289
    Africanunspecified    252
    Other                 160
    EastAsian              97
    Name: count, dtype: int64

    """

    return sr_RETURN



def get_outdir(_x, _out_dir="/data02/UKBB_WGS_summaries"):

    l_outdirs = os.listdir(_out_dir)

    l_found = [_ for _ in l_outdirs if _.endswith(_x)]

    if len(l_found) == 0:
        return np.nan
    elif len(l_found) > 1:
        return np.nan
    else:
        return l_found[0]



def get_wget_CMD_etc(_df, _out):


    df_temp = _df[['accessionId', 'reportedTrait', 'efoTraits', 'discoverySampleAncestry']]
    sr_ftp = _df['summaryStatistics'].str.replace("http", "ftp")
    sr_out_dir = _df['accessionId']

    df_temp = pd.concat([df_temp, sr_ftp, sr_out_dir], axis=1)

    CMD = \
        "wget -c -r -nH --cut-dirs=6 --no-parent --reject index.html* \\\n" + \
            '\t--exclude-directories={exclude_dir} \\\n' + \
            "\t{ftp} \\\n" + \
            "\t-P {out_dir}"

    ## 시간 없어서 "harmonized/"는 제외.
    ## 이거 제끼는거 시행착오가 존나 많았음;; 결론은 서버기준 절대경로로 하나씩 언급해야 함. '*'같은 정규표현식은 인지를 못함.

    with open(_out, 'w') as f_out:

        for _index, _accessionId, _reportedTrait, _efoTraits, _discoverySampleAncestry, _ftp, _out_dir in df_temp.itertuples():

            str_signpost = f"\n\n##### {_accessionId} / {_reportedTrait} / {_discoverySampleAncestry}"

            f_out.write(str_signpost + "\n")
            f_out.write(CMD.format(ftp=_ftp, out_dir=_out_dir, exclude_dir=_ftp.replace("ftp://ftp.ebi.ac.uk", "") + "/harmonised/"))


    return _out



"""
다음부터는 그냥 함수내에서 디테일한 수행내역들까지 목적과 결론을 요약해서 쓰면 더 좋을듯.

- out_dir들을 retrieve해봤고, 여기서 실패한애들 확인.
    - out_dir에 '/'가 들어가서 뭐 짤린 애들 두명 => 얘네들은 manually 수정함.
    - "Crohn's disease" => "'" 이거때문에 얘부터 이후인애들은 아예 안받아졌음.
    - 이거 목요일에 command 다시 올려서 download 다시 받음. (`get_wget_CMD_etc()`)

"""

def load_SSFN_UKBBwgs_EUR(
        _fpath_SSFN_T834="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/UKBBwgs_PMID40770095_studies_export.EUR.tsv",
        # _fpath_SSFN_D_T763="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/UKBBwgs_PMID40770095_studies_export.EUR.D_T763.tsv",
        # _fpath_SSFN_B_T19="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/UKBBwgs_PMID40770095_studies_export.EUR.B_T19.tsv",
    ):

    df_SSFN_T834 = pd.read_csv(_fpath_SSFN_T834, sep='\t', header=0)
    # df_SSFN_D_T763 = pd.read_csv(_fpath_SSFN_D_T763, sep='\t', header=0)
    # df_SSFN_B_T19 = pd.read_csv(_fpath_SSFN_B_T19, sep='\t', header=0)

    # return df_SSFN_T834, df_SSFN_D_T763, df_SSFN_B_T19

    return df_SSFN_T834



def check_md5sum(_accessionID, out_dir, base="/data02/UKBB_WGS_summaries/"):

    workdir = os.path.join(base, out_dir)
    # md5sum.txt가 있는 디렉토리에서 실행

    cmd = ["md5sum", "-c", "md5sum.txt"]
    # md5sum.txt 자체 실패를 피하려면 "--ignore-missing" 또는 파일 필터링(옵션 C) 사용

    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)

    flag = f"{_accessionID}.tsv.gz: OK" in r.stdout

    return flag



def check_directory_miss(_accessionID, out_dir, base="/data02/UKBB_WGS_summaries/"):

    workdir = os.path.join(base, out_dir)
    # md5sum.txt가 있는 디렉토리에서 실행

    l_files = os.listdir(workdir)

    l_summary = [_ for _ in l_files if _.endswith(".tsv.gz")]

    # f_redundant = len(l_summary) > 1
    # f_got_necessary = f"{_accessionID}.tsv.gz" in l_summary

    return l_summary




"""
## (3-5) miss냈던 놈들에 한정해서 iterate하자
- shell 파일 하나 만들고, 해당 directory내에서 accessionID mismatch되는 items들은 다 remove함.
- 그리고 md5sum.txt파일도 지움. 이게 저번에 잘못받은거 기준으로 다운받은거야서 필요없음. 제대로 받은거의 md5sum.txt가 필요함.

"""


def make_sh_ToRemove_missed_items(_df, _out_sh, base="/data02/UKBB_WGS_summaries/"):

    """
    - 어제 밤에 생각을 해봤는데, 그냥 낱개로 "rm" command를 치는 sh파일 하나 만드는게 낫겟음. (2025.10.18.)
    - 
    
    """

    l_ToExtract = ['accessionId', 'out_dir']

    df_ToIter = _df[l_ToExtract]


    with open(_out_sh, 'w') as f_out:


        for i, (_index, _accessionId, _out_dir) in enumerate(df_ToIter.itertuples()):

            str_ToPrint = f"=====[{i}]: {_accessionId} / {_out_dir}"
            print(str_ToPrint)

            workdir = os.path.join(base, _out_dir)
            l_files = os.listdir(workdir)
            l_files_ToRemove = [_ for _ in l_files if not _.startswith(_accessionId)]

            f_out.write("\n##" + str_ToPrint + "\n")

            for _ in l_files_ToRemove:
                ToRemove = os.path.join(_out_dir, _)
                f_out.write(f"rm {ToRemove}" + "\n")


    return _out_sh



def subset_wget_sh_file(_df_SSFN_DIRmiss, _fpath_sh="/data02/wschoi/_hCAVIAR_v2/20250912_UKBB_WGS/20250914.wget.priority_others.EUR.trial2.sh"):

    """
    - 쓸데 없는 짓일 수도 있지만, 그냥 download할때 썼던 sh파일의 commands들을 subset하려고.
    
    """

    sr_accessionID = _df_SSFN_DIRmiss['accessionId'].tolist()

    with open(_fpath_sh, 'r') as f_sh:
        l_temp = f_sh.readlines()


    l_temp = l_temp[1:] # 첫번째 '\n' line하나만 하드코딩으로 제거.

    ## 이후 6 lines들 씩 읽으면 됨.
    d_items = {}
    chunksize = 6

    for i_chunk in range(0, len(l_temp), chunksize):

        line_2nd = l_temp[i_chunk + 1]
        accessionID = line_2nd.split()[1]

        d_items[accessionID] = l_temp[i_chunk : (i_chunk + chunksize)]

    
    d_RETURN = {k: v for k, v in d_items.items() if k in sr_accessionID}


    return d_RETURN



def write_subsetted_sh(_out_sh, _df_SSFN_DIRmiss, _fpath_sh="/data02/wschoi/_hCAVIAR_v2/20250912_UKBB_WGS/20250914.wget.priority_others.EUR.trial2.sh"):


    d_subsetted = subset_wget_sh_file(_df_SSFN_DIRmiss, _fpath_sh)


    with open(_out_sh, 'w') as f_out:

        for _accessionID, _l_lines in d_subsetted.items():

            f_out.write(''.join(_l_lines))


    return _out_sh



# def func_worker(_index, _accessionId, _out_dir, _reportedTrait, _sumstats0):
def func_worker(_arg):

    _index, _accessionId, _out_dir, _reportedTrait, _sumstats0 = _arg


    _base_dir = "/data02/UKBB_WGS_summaries/"
    _fpath_df_T1DGC_bim = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.bim"
    _hg = 38
    _col_SNP = "ID"
    _fpath_hg38_to_19 = "/data02/wschoi/hg38ToHg19.over.chain"
    _liftover = "/home/wschoi/miniconda3/envs/liftover/bin/liftOver"


    df_T1DGC_bim = pd.read_csv(
        _fpath_df_T1DGC_bim, sep='\t', header=None,
        names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
    )
    # print(df_T1DGC_bim)



    print(f"===[{_index}]: {_accessionId} / {_out_dir} / {_reportedTrait} ({datetime.now()})")
    
    _fpath_ss_raw = os.path.join(_base_dir, _out_dir, f"{_accessionId}.tsv.gz")

    try:
        df_temp_ss = pd.read_csv(_fpath_ss_raw, sep='\t', header=0)
        
        RAW_SS = INPUT_Raw_GWAS_summary.INPUT_Raw_GWAS_summary(
            df_temp_ss, _hg, _sumstats0, _col_SNP,
            _bin_liftOver=_liftover,
            _fpath_hg38_to_19=_fpath_hg38_to_19,
            _df_ref_bim=df_T1DGC_bim
        )
        
        RAW_SS.run()
    except:
        return False
    else:
        return True



def prepr_UKBBwgs(_df_SSFN, _max_workers=6):

    l_ToExtract = ['accessionId', 'out_dir', 'reportedTrait', 'sumstats0']
    _df_ToIter = _df_SSFN[l_ToExtract]

    # print(_df_ToIter)

    with ProcessPoolExecutor(max_workers=_max_workers) as executor:
        results = list(executor.map(func_worker, _df_ToIter.itertuples(name=None)))

    print(all(results))

    return results



def check_gunzip_t(_accessionID, out_dir, base="/data02/UKBB_WGS_summaries/"):

    workdir = os.path.join(base, out_dir)
    # fpath = os.path.join(workdir, f"{_accessionID}.tsv.gz")
    fpath = f"{_accessionID}.tsv.gz"

    cmd = ["gunzip", "-t", fpath]
    # md5sum.txt 자체 실패를 피하려면 "--ignore-missing" 또는 파일 필터링(옵션 C) 사용

    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)

    # flag = f"{_accessionID}.tsv.gz: OK" in r.stdout

    return r



def get_top_SNP_UKBBwgs(_fpath_sumstats0, _df_T1DGC_bim):

    ## 여기는 sumstats0가 존재하지 않는 경우에 대해 예외처리 안해도 됨. (MVP때는 Pneumonia때문에 해야 했음.)

    ## T1DGC reference panel의 SNPs들만. (매우 중요. 안하면 HLA varaint markers들이랑 혼선됨.)
    f_is_HLA = Util.is_HLA_locus(_df_T1DGC_bim['SNP'])
    df_T1DGC_bim_SNP = _df_T1DGC_bim[~f_is_HLA]


    ## sumstats0에서 SNPs들만 남기기
    df_sumstats0 = pd.read_csv(_fpath_sumstats0, sep='\t', header=0)
    # print(df_sumstats0.shape[0])

    f_is_SNP = df_sumstats0['BP'].isin(df_T1DGC_bim_SNP['BP'])
    df_sumstats0 = df_sumstats0[f_is_SNP]
    # print(df_sumstats0.shape[0])
        ## subset 잘 됨.


    ## Top 캐오기
    df_sumstats0 = df_sumstats0.sort_values("P")
    df_sumstats0 = df_sumstats0.iloc[0, :]

    _SNP = df_sumstats0['SNP']
    _BP = df_sumstats0['BP']
    _P = df_sumstats0['P']

    # _SNP, _BP, _P = df_sumstats0.tolist()
    _hasSignal = _P < 5e-8


    """
    - 최종 아래와 같이 transform해서 썼음.

    df_temp = pd.DataFrame(
        
        [list(_) for _ in df_MVP_PheCode_4['sumstats0'].map(lambda x: _20251016_MVP_UKBBwgs.get_top_SNP(x))],
        columns=['SNP_top', 'BP_top', 'P_top', 'hasSignal']
    )
    display(df_temp)    
    
    """


    return _SNP, _BP, _P, _hasSignal



## 여기서 부터 GPU server가서 작업.

def load_and_prepr_SSFN_UKBBwgs(
        _fpath_SSFN = "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole/UKBBwgs_PMID40770095_studies_export.EUR.3.xlsx",
        _out_dir = "/data02/wschoi/_hCAVIAR_v2/20251019_UKBBwgs_whole_EUR_run/"
    ):

    ## load
    df_SSFN = pd.read_excel(_fpath_SSFN, header=0)


    ## GWAS signal 있는애들만.
    f_hasSignal = df_SSFN['hasSignal']
    df_SSFN = df_SSFN[f_hasSignal]


    ## out_prefix 박아놓고 시작하자.
    sr_Trait_Description_2 = prepare_output_label(df_SSFN['efoTraits'])

    df_temp = pd.concat([df_SSFN['accessionId'], sr_Trait_Description_2], axis=1)

    sr_OUT = pd.Series(
        [f"{_out_dir}/UKBBwgs.{_accessionId}.{_Trait_Description_2}" for _index, _accessionId, _Trait_Description_2 in df_temp.itertuples()],
        index=df_SSFN.index,
        name='out_prefix'
    )

    df_RETURN = pd.concat([df_SSFN, sr_OUT], axis=1)

    return df_RETURN



def run_UKBBwgs_SUM2HLA(_args):

    _index, _accessionId, _reportedTrait, _sumstats0, _out_prefix, _gpu_id = _args

    print(f"=====[{_index}]: _accessionId: {_accessionId} / _Trait_Description: {_reportedTrait}")

    cmd = [
        "conda", "run", "-n", "jax_gpu", # 'jax_gpu' 환경에서 실행하도록 지정 (cuda 11)
        "python",
        "SUM2HLA.py",
        "--sumstats", _sumstats0,
        "--ref", "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "--out", _out_prefix,
        "--gpu-id", str(_gpu_id)
    ]

    # print(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    
    if result.returncode != 0:
        # 에러가 발생한 경우, 에러 메시지를 출력
        return False
    return True



"""
- UKBB_WGS diseases들 GPU 서버에서 돌리고 난 후, 이제 결과 갈무리는 다음에서 함.
    - 20251015_MVP_whole_EUR_run_2.ipynb

- 위에서 top PP캐오는 함수들은 reuse해도 될 것 같아서 새로 안만듬.
"""


def extract_ICDcode(_x):

    """
    - 'reportedTrait' column의 each element를 tranform해야 함.
    """

    if _x.startswith("ICD"):

        p_ICD10_code = re.compile(r"ICD10\s+(\w)\d+(-\w\d+)?\:")

        m = p_ICD10_code.match(_x)

        if not bool(m): return "Failed"

        return m.group(1)


    return "Biomarker"




#################### [ MVP - T1D vs. T2D for Seborrheic dermatitis ] ####################

def load_T2D_Suzuki(_fpath="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR_run_2/Suzuki.Nature2024.T2DGGI.EUR.sumstats/EUR_Metal_LDSC-CORR_Neff.v2.txt"):

    d_ToRename = {
        "Chromsome": "CHR", 
        "Position": "BP",
        "EffectAllele": "A1",
        "NonEffectAllele": "A2",
        "Beta": "BETA",
        "SE": "SE",
        "EAF": "AF_A1",
        "Pval": "P"
    }

    df_T2D = pd.read_csv(_fpath, sep='\t', header=0) \
                .rename(d_ToRename, axis=1) \
                .sort_values(["CHR", "BP"]) \
                .query("CHR == 6")


    ## 얘네 'SNP' column이 없음.
    sr_SNP = pd.Series(
        [f"{_CHR}:{_BP}" for _index, _CHR, _BP in df_T2D[['CHR', 'BP']].itertuples()],
        name="SNP",
        index = df_T2D.index
    )

    df_T2D = pd.concat([sr_SNP, df_T2D], axis=1)


    return df_T2D


def prepr_T2D_Suzuki(_df_T2D_suzuki, 
                     _out="/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR_run_2/Suzuki.Nature2024.T2DGGI.EUR.chr6.sumstats0"):
    
    # "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR_run_2/Suzuki.Nature2024.T2DGGI.EUR.chr6.sumstats"

    df_T1DGC_bim = pd.read_csv(
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.bim",
        sep='\t', header=None, names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
    )
    print(df_T1DGC_bim)

    raw_GWAS_summary = INPUT_Raw_GWAS_summary.INPUT_Raw_GWAS_summary(_df_T2D_suzuki, 19, _out, _col_SNP="SNP", _df_ref_bim=df_T1DGC_bim, _d_ToRename_add={"Neff": "N"})


    ## MVP랑 똑같이 하되, recalc_BETA_SE_from_OR_CI() 이것만 안 넣으면 됨.
    raw_GWAS_summary.rename_colnames()
    raw_GWAS_summary.remove_NA_BP()

    raw_GWAS_summary.extract_HLAregion()

    if raw_GWAS_summary.hg != 19:
        raw_GWAS_summary.do_Liftover()

    raw_GWAS_summary.filter_ref_SNPs()

    # raw_GWAS_summary.recalc_BETA_SE_from_OR_CI()
    raw_GWAS_summary.recalc_Pvalue()

    raw_GWAS_summary.sort_columns()

    ### export
    raw_GWAS_summary.df_ss_raw.to_csv(raw_GWAS_summary.out, sep='\t', header=True, index=False, na_rep="NA")



    return raw_GWAS_summary