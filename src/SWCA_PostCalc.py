"""
- COJO result (conditioned BETA, SE, and P)에 Bayesian fine-mapping을 수행하는 module
- 전처리 등등 사소하게 다른 부분들이 있어서 script하나 새로 만듬.


"""

import os
from os.path import basename, dirname, join
import subprocess
import json

import pandas as pd
import numpy as np

import src.Util as mod_Util
from src.SUM2HLA_PostCalc_Cov import postprepr_LL

from datetime import datetime



def make_cma_to_sumstats3(_fpath_cma, _f_onlySNPs=False):

    df_cma = pd.read_csv(_fpath_cma, sep='\t', header=0) if isinstance(_fpath_cma, str) else _fpath_cma
    # display(df_cma)

    if _f_onlySNPs:
        f_HLA = mod_Util.is_HLA_locus(df_cma['SNP'])
        df_cma = df_cma[~f_HLA]

    ##### (1) column 두 개만 챙기면 됨: (1) 'SNP_LD' and (2) 'Z_fixed'
    sr_Z_fixed = (df_cma['bC'] / df_cma['bC_se']).rename("Z")

    df_RETURN = pd.concat([df_cma['SNP'], sr_Z_fixed], axis=1)

    return df_RETURN



def __MAIN__postCalc_SWCA(_fpath_COJOsummary, _fpath_LDmatrix,
                          _batch_size=30, _gamma=0.01, _ncp=5.2, _N_causal=1):

    ##### (0) load data

    df_LDmatrix = pd.read_csv(_fpath_LDmatrix, sep='\t', header=0) \
                        if isinstance(_fpath_LDmatrix, str) else _fpath_LDmatrix
    df_LDmatrix.index = df_LDmatrix.columns

    sr_COJO_summary = pd.read_csv(_fpath_COJOsummary, sep='\t', header=0) \
                            if isinstance(_fpath_COJOsummary, str) else _fpath_COJOsummary
    sr_COJO_summary = sr_COJO_summary \
                          .rename({"SNP_LD": "SNP", "Z_fixed": "Z"}, axis=1) \
                          .loc[:, ['SNP', 'Z']] \
                          .set_index("SNP").squeeze('columns') \
                          .loc[df_LDmatrix.columns.to_series()]  ### (***; 매우 중요) LD matrix의 marker순서로 match시킴.
    ## 앞서 `_fpath_LDmatrix`를 `_fpath_COJOsummary`를 바탕으로 match시켜서 전처리했기 때문에 `.loc[]`만 활용해도 됨.



    ##### (1) Rank-1 optimized LL computation (N_causal=1)
    #
    # l_ix_SNPs=None (all K markers as candidates) → v_c = R[:, c]
    # R^{-1} @ v_c = R^{-1} @ R[:, c] = e_c  →  alpha_c = R[c,c],  beta_c = Z[c]
    #
    # arr_PIP_acc[c] = LL_1(c) - LL_0 + Lprior
    #               = -0.5*log(denom_c) + 0.5*ncp*Z[c]^2/denom_c + Lprior
    # where denom_c = 1 + ncp * R[c, c]

    K      = df_LDmatrix.shape[0]
    Z      = sr_COJO_summary.values                           # (K,)
    R      = df_LDmatrix.values                               # (K, K)
    Lprior = np.log(_gamma) + (K - 1) * np.log(1.0 - _gamma) # N_causal=1

    diag_R      = np.diag(R)                                  # (K,)
    denom       = np.maximum(1.0 + _ncp * diag_R, 1e-12)     # (K,)
    arr_PIP_acc = -0.5 * np.log(denom) + 0.5 * _ncp * (Z ** 2) / denom + Lprior  # (K,)



    ##### (2) RETURN

    df_PP = pd.DataFrame(
        {
            "SNP": df_LDmatrix.columns.tolist(),
            "LL+Lprior": arr_PIP_acc
        }
    )

    df_PP = postprepr_LL(df_PP, _l_type=['whole'])

    return df_PP['whole'] # 흔적기관... "whole"



def __MAIN__(_fpath_cma, _fpath_ref, _fpath_ref_LD, _out_dir,
             _plink, _ncp=5.2):

    ##### (0) load data
    os.makedirs(_out_dir, exist_ok=True)

    ## cma (COJO output)
    df_cma = pd.read_csv(_fpath_cma, sep='\t', header=0) if isinstance(_fpath_cma, str) else _fpath_cma

    ## LD matrix of the reference data
    df_LDmatrix = pd.read_csv(_fpath_ref_LD, sep='\t', header=0) if isinstance(_fpath_ref_LD, str) else _fpath_ref_LD
    df_LDmatrix.index = df_LDmatrix.columns

    # print(df_cma)
    # print(df_LDmatrix)



    ##### (1) prepr - clumping

    ### run the PLINK clumping
    def run_PLINK_clump(_fpath_ToClump, _fpath_LD_SNP_HLA, _out_prefix, _plink):

        cmd = [
            _plink,
            "--clump", _fpath_ToClump,
            "--bfile", _fpath_LD_SNP_HLA,
            "--out", _out_prefix,
            "--allow-no-sex", "--keep-allele-order"
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        except subprocess.CalledProcessError as e:
            print(json.dumps(cmd, indent='\t'))
            raise e

        # except: # 다른 에러들도 마지막 command보여주고 주어진 error 보여주도록
        #     print(json.dumps(cmd, indent='\t'))
        #     raise

        return _out_prefix + ".clumped"        


    ToClump = join(_out_dir, basename(_fpath_cma) + ".ToClump")
    df_cma.rename({"pC": "P"}, axis=1).loc[:, ['SNP', 'P']] \
            .to_csv(ToClump, sep='\t', header=True, index=False, na_rep="NA")

    OUT_clump = run_PLINK_clump(ToClump, _fpath_ref, join(_out_dir, basename(_fpath_cma) + ".CLUMP"), _plink)



    ##### (2) prepr - extract the clumped SNP and HLA markers from the CMA summary.

    df_clumped = pd.read_csv(OUT_clump, sep=r'\s+', header=0)
    f_clumped = df_cma['SNP'].isin(df_clumped['SNP'])

    df_cma_2 = df_cma.loc[f_clumped, :]



    ##### (3) convert the clumped CMA as the sumstats3

    df_cma_sumstats3 = make_cma_to_sumstats3(df_cma_2)

    OUT_cma_clumped = join(_out_dir, basename(_fpath_cma) + ".CLUMP.sumstats3")
    df_cma_sumstats3.to_csv(OUT_cma_clumped, sep='\t', header=True, index=False, na_rep="NA")



    ##### (4) extract the clumped CMA's SNPs from the LDmatrix

    f_ToExtract = df_LDmatrix.columns.to_series().isin(df_cma_2['SNP'])
    df_LDmatrix_2 = df_LDmatrix.loc[f_ToExtract, f_ToExtract] # 얘 fwrite은 잠깐 보류. 가급적 안하고 싶음.

    # print(df_cma_2)
    # print(df_LDmatrix_2)



    ##### (5) Posterior probability

    df_PP_cma = __MAIN__postCalc_SWCA(df_cma_sumstats3, df_LDmatrix_2, _ncp=_ncp)

    OUT_PP_cma = join(_out_dir, basename(_fpath_cma) + ".PP")
    df_PP_cma.to_csv(OUT_PP_cma, sep='\t', header=True, index=False, na_rep="NA")



    return df_PP_cma, OUT_PP_cma




if __name__ == "__main__":

    print("Hello")

    df_PP, OUT_PP = __MAIN__(
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST.HLA.ROUND_1.cma.cojo",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST",
        "/home/wschoi/miniconda3/bin/plink"
    )

    print(OUT_PP)
    print("df_PP:")
    print(df_PP)

    pass