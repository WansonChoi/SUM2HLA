"""
- COJO result (conditioned BETA, SE, and P)에 Bayesian fine-mapping을 수행하는 module
- 전처리 등등 사소하게 다른 부분들이 있어서 script하나 새로 만듬.


"""

import os
from os.path import basename, dirname, join

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax

import src.Util as mod_Util
from SUM2HLA_PostCalc_Cov import yield_configure_batch, generate_LD_matrices_jax, calc_LL_given_batch_jax, postprepr_LL

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

    # (2025.04.17.) main postCalc의 wrapper도 이런식으로 바뀌었으면 좋겠음.

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

    # print(df_LDmatrix)
    # print(sr_COJO_summary)



    ##### (1) main variables

    ### (1-1) LL_0
    eigenvalues_LDmatrix = jnp.linalg.svd(df_LDmatrix.values, compute_uv=False)  # only eigenvalues, not with eigenvectors
    term2_LDmatrix = -0.5 * jnp.sum(jnp.log(eigenvalues_LDmatrix))
    _LL_0 = term2_LDmatrix - 0.5 * (sr_COJO_summary.values.T @ (jnp.linalg.solve(df_LDmatrix.values, sr_COJO_summary.values)))  # term2 + term3

    ### (1-2) Lprior
    M = df_LDmatrix.shape[0]
    Lprior = _N_causal * np.log(_gamma) + (M - _N_causal) * np.log(1 - _gamma)
    # Lprior는 N_causal이 given되는 지금 계산하는게 나을 듯.

    ### (1-3) output
    arr_PIP_acc = np.zeros(M)
    acc_LL_N_causal = 0.0  # `N_causal`이 주어졌을 때의 LL 누적. 나중에 N_causal이 몇일때 가장 LL이 높게 나오는지도 알고싶음.



    ##### (2) Main iteration

    iter_batch_configures = yield_configure_batch(M, _N_causal, _batch_size, _f_as_list=False)

    for i, _batch_configures in enumerate(iter_batch_configures):

        if i % 100 == 0:
            print("=====[{}]: {}-th batch / First 5 items: {} ({})".format(i, i, _batch_configures[:5], datetime.now()))
            # print("First 5 items: {}".format(_batch_configures[:5]))
            # display(_batch_configures)

        ### (2-1) calc LL for the configure batch.

        GWAS_summary_jax = jnp.array(sr_COJO_summary.values)

        # ✅ (1) 벡터 연산으로 LD matrices 생성
        t_start_1 = datetime.now()

        _LDmatrix_jax = generate_LD_matrices_jax(
            jnp.array(_batch_configures), jnp.array(df_LDmatrix.values), None, _ncp
        )
            # None을 주어 SNPs들만 extraction 하지 않음. (<=> the main fine-mapping)

        # display(_LDmatrix_jax[:10, :10])

        # print("✅ (1) LD matrices 생성 (R + R @ diagC @ R) && Subset: {}".format(datetime.now() - t_start_1))

        # ✅ (2) LL 계산.
        t_start_2 = datetime.now()

        l_LL_batch = calc_LL_given_batch_jax(GWAS_summary_jax, _LDmatrix_jax, Lprior, _LL_0, _ncp=_ncp)
        # display(l_LL_batch)

        # print("✅ (2) LL 계산: {}".format(datetime.now() - t_start_2))

        # print("Total Time of this batch: {}".format(datetime.now() - t_start_1))

        ### (2-2) accumulation for PIP.
        for j, (_configure, _LL) in enumerate(zip(_batch_configures, l_LL_batch)):

            # print("===[{}]: {} / {}".format(j, _configure, _LL))

            for _idx_SNP in _configure:
                # print(_idx_SNP)

                arr_PIP_acc[_idx_SNP] += _LL

        # display(arr_PIP_acc)

        ### (2-3) acc for the prob. of `N_causal`
        acc_LL_N_causal += np.sum(l_LL_batch)

        # display(arr_PIP_acc[:10])
        # display(acc_LL_N_causal)

        # if i >= 1: break



    ##### (3) RETURN

    ## 당장은 `arr_PIP_acc`만 필요함.
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

    ToClump = join(_out_dir, basename(_fpath_cma) + ".ToClump")
    df_cma.rename({"pC": "P"}, axis=1).loc[:, ['SNP', 'P']] \
            .to_csv(ToClump, sep='\t', header=True, index=False, na_rep="NA")

    OUT_clump = mod_Util.run_PLINK_clump(ToClump, _fpath_ref, join(_out_dir, basename(_fpath_cma) + ".CLUMP"), _plink)



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