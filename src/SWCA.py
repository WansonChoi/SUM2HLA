import os, sys, re
from os.path import dirname
import numpy as np
import scipy as sp
import pandas as pd

import math

from datetime import datetime

import subprocess
import json

import src.SWCA_SummaryImp as mod_SummaryImp
import src.SWCA_fine_mapping as fine_mapping_SWCA


########## (1) Summary imputation

## 위 `mod_SummaryImp` import함.



########## (2) make the COJO input

def make_COJO_input(_df_ss, _df_MAF, _N, _df_ss_SNP=None, _maf=0.01):
    
    ##### Required columns: [SNP A1 A2 freq b se p N]
    l_header_final = ["SNP", "A1", "A2", "freq", "b", "se", "p", "N"]

    
    
    ##### (1) MAF
    ## 그니께, ['A1', 'A2'] 때문에라도 어차피 df_MAF에 join해야함.
    df_MAF_2 = _df_MAF.drop(['CHR', 'NCHROBS'], axis=1, errors='ignore').rename({"MAF": "freq"}, axis=1)
    # display(df_MAF_2)
    
    
    
    ##### (2) Imputed (HLA loci)

    df_imputed = _df_ss \
                    .rename_axis("SNP", axis=0) \
                    .reset_index(drop=False) \
                    .rename({"Conditional_mean": "Z_imputed"}, axis=1)
    
    df_imputed = df_imputed.merge(df_MAF_2, on='SNP')

    sr_P_imputed = df_imputed['Z_imputed'] \
                    .abs() \
                    .map(lambda x: 2 * sp.stats.norm.cdf(-x)) \
                    .rename("p")
    
    sr_SE_imputed = pd.Series([1.0]*df_imputed.shape[0], index = df_imputed.index, name='se')
    sr_N = pd.Series([_N]*df_imputed.shape[0], name='N')
    
    df_imputed = pd.concat(
        [
            df_imputed,
            sr_P_imputed,
            sr_SE_imputed,
            sr_N
        ],
        axis=1
    ) \
        .rename({"Z_imputed": "b"}, axis=1)

    # print(df_imputed.shape)

    
    if isinstance(_maf, float) and (0.0 <_maf < 1.0):
        f_MAF = (df_imputed['freq'] < _maf) | (df_imputed['freq'] > (1 - _maf))
        df_imputed = df_imputed[~f_MAF]
        
        """
        일단, Imputed HLA loci에만 MAF thresholding 하는걸로.
        """
    
    # print(df_imputed.shape)
    
    df_RETURN = df_imputed
    
    
    
    ##### (3) Unimputed (the clumped SNPs)
    
    ## SNPs들의 summary가 주어지면 얘도 concat
    if isinstance(_df_ss_SNP, pd.DataFrame):
        
        l_ToExtract = ['SNP_LD', 'A1_LD', 'A2_LD', 'BETA', 'SE', 'P']
        d_ToRename = {"SNP_LD": "SNP", 
                      'A1_LD': "A1", 'A2_LD': "A2", 
                      "BETA": 'b', 'SE': 'se', "P": 'p', "MAF": "freq"}

        df_unimputed = pd.concat([
            _df_ss_SNP[l_ToExtract], 
            pd.Series([_N]*_df_ss_SNP.shape[0], index=_df_ss_SNP.index, name='N')
        ], axis=1) \
            .rename(d_ToRename, axis=1)

        ## `_df_ss`와 MAF간 shared SNPs들만 남김.
        df_unimputed = df_unimputed.merge(
            df_MAF_2, on=['SNP', 'A1', 'A2']
        )
        
        # display(df_unimputed)
        df_RETURN = pd.concat([df_unimputed, df_imputed])

    
    
    ##### (4) Wrap-up
    df_RETURN = df_MAF_2[['SNP', 'A1', 'A2']] \
                    .merge(df_RETURN, on=['SNP', 'A1', 'A2']) \
                    .loc[:, l_header_final]
    ## 사실상 row랑 column의 sorting
    
    return df_RETURN



########## (3) Manual Stepwise Conditional regression with COJO.

def sort_COJO_result_cond(_fpath_COJO_cond):
    
    df_COJO_cond = pd.read_csv(_fpath_COJO_cond, sep='\t', header=0)
    
    sr_zC = (df_COJO_cond['bC'] / df_COJO_cond['bC_se']).rename("zC")
    
    
    df_ToSort = pd.concat([df_COJO_cond['pC'], -sr_zC.abs()], axis=1).sort_values(['pC', 'zC'])
    
    df_RETURN = pd.concat([df_COJO_cond, sr_zC], axis=1).loc[df_ToSort.index, :]
    
    
    return df_RETURN



def get_single_residue_markers(_l_top_signal_markers, _df_MAF, _df_cma_ROUND_N):

    ##### make sure the `_df_ma_ROUND_N` is sorted by 'p'
    # if _top_signal_markers == None:
    #     _df_cma_ROUND_N = _df_cma_ROUND_N.sort_values("p")
    #     print(_df_cma_ROUND_N)
    #
    #     _top_signal_markers = _df_cma_ROUND_N['SNP'].iat[0]

    # display(_top_signal_markers)

    df_cma_ROUND_N = pd.read_csv(_df_cma_ROUND_N, sep='\t', header=0) if isinstance(_df_cma_ROUND_N, str) else _df_cma_ROUND_N
    df_cma_ROUND_N.sort_values("p", inplace=True)

    df_MAF = pd.read_csv(_df_MAF, sep=r'\s+', header=0) if isinstance(_df_MAF, str) else _df_MAF



    ##### MAF
    df_MAF_2 = df_MAF[['SNP', 'A1', 'MAF']]
    sr_MAF_2 = df_MAF['MAF'].map(lambda x: x if x < 0.5 else 1 - x).rename("MAF_2")
    df_MAF_2 = pd.concat([df_MAF_2, sr_MAF_2], axis=1)



    ##### pattern for AA and intragenic SNPs
    p_AA = re.compile(r'^(AA_\S+_-?\d+_)\d+_(\S+)$')
    # p_HLA = re.compile(r'HLA_DRB1_04')
    p_SNPS = re.compile(r'^(SNP_\S+_\d+)_(\S+)$')
    # p_INS = re.compile()



    l_OUT_single_factor_markers = [] # reference factor is excluded
    l_OUT_df_ToCondition = [] # for check

    for _top_signal_markers in _l_top_signal_markers:

        ### factor가 2개 이상인 AA or intragenic SNP marker
        if bool(p_AA.match(_top_signal_markers)):

            m = p_AA.match(_top_signal_markers)
            prefix_temp = m.group(1)

            f_AA_locus = df_cma_ROUND_N['SNP'].str.startswith(prefix_temp)
            _df_cma_ROUND_N_2 = df_cma_ROUND_N[f_AA_locus]

            f_is_single_factor = _df_cma_ROUND_N_2['SNP'].map(lambda x: p_AA.match(x).group(2)).map(lambda x: len(x) == 1)
            _df_cma_ROUND_N_3 = _df_cma_ROUND_N_2[f_is_single_factor]


        elif bool(p_SNPS.match(_top_signal_markers)):

            m = p_SNPS.match(_top_signal_markers)
            prefix_temp = m.group(1)
            # print(prefix_temp)

            f_SNPS_locus = df_cma_ROUND_N['SNP'].str.startswith(prefix_temp)
            _df_cma_ROUND_N_2 = df_cma_ROUND_N[f_SNPS_locus]

            f_is_single_factor = _df_cma_ROUND_N_2['SNP'].map(lambda x: p_SNPS.match(x).group(2)).map(lambda x: len(x) == 1)
            _df_cma_ROUND_N_3 = _df_cma_ROUND_N_2[f_is_single_factor]


        else:

            ## (ex) "HLA_DRB1_0401"
            l_OUT_single_factor_markers.extend([_top_signal_markers])
            # No job for `l_OUT_df_ToCondition`
            continue

        """
        - 해당 locus의 single factor markers들만 남긴다.
        - 얘네들의 MAF정보로 제일 prevalent한 놈만 제외한다. => colinearity를 반영하기 위해.
        """

        # df_ToCondition = _df_cma_ROUND_N_3[['SNP', 'p', 'A1']]
        df_ToCondition = _df_cma_ROUND_N_3.rename({"refA": "A1"}, axis=1).loc[:, ['SNP', 'p', 'A1']]
            # 최초 '*.ma'파일은 A1인데 (ROUND_0), cojo파일상에서는 'refA'임 (ROUND_1 ~)

        df_ToCondition = df_ToCondition.merge(df_MAF_2, on=['SNP', 'A1'])
        df_ToCondition = df_ToCondition.sort_values("MAF_2", ascending=True)
        # display(df_ToCondition)

        if df_ToCondition.shape[0] > 1:
            l_OUT_single_factor_markers.extend(df_ToCondition['SNP'].iloc[:-1].tolist()) # co-linearity 방지.
        else:
            l_OUT_single_factor_markers.append(df_ToCondition['SNP'].iloc[0])
            """
            이런 예외가 있었음.
            
                                  SNP             p A1     MAF   MAF_2
            0  AA_DRB1_112_32657542_H  2.185140e-30  P  0.9883  0.0117
            
            - single residue marker가 1개인 경우, `.iloc[:-1]`이렇게 하면 아무것도 추가가 안됨.  
            - 걍 얘만 extend 해줘야 함.
            
            """

        l_OUT_df_ToCondition.append(df_ToCondition)



    df_ToRefer = pd.concat(l_OUT_df_ToCondition) if len(l_OUT_df_ToCondition) > 0 else None

    return l_OUT_single_factor_markers, df_ToRefer



def iterate_GCTA_COJO(_fpath_ma, _initial_top_signal, _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF, _out_prefix,
                      _N_max_iter=100, _f_use_finemapping=True, _f_single_factor_markers=True,
                      _gcta="/home/wschoi/bin/gcta64", _plink="/home/wschoi/miniconda3/bin/plink"):
    
    ## 얘가 main varaible

    ## (2025.04.25.) initial signal도 single residue marker로 전환해줘야 함.
    if _f_single_factor_markers:
        l_secondary_signals, df_ToRefer = get_single_residue_markers([_initial_top_signal], _fpath_ref_MAF, _fpath_ma)
    else:
        l_secondary_signals = [_initial_top_signal] if isinstance(_initial_top_signal, str) else _initial_top_signal



    ##### Main iteration
    
    for i in range(1, _N_max_iter+1):
        
        print("===[{}]: ROUND_{}.".format(i, i))
        # print("Current conditions:\n{}".format(l_secondary_signals))
        
        snplist_temp = os.path.join(_out_prefix + "{}.snplist".format(i))
        out_prefix_temp = re.sub(r'.snplist$', '', snplist_temp)

        sr_snplist = pd.Series(l_secondary_signals, name="snptlist")
        # display(sr_snplist)
        sr_snplist.to_csv(snplist_temp, header=False, index=False, na_rep="NA")
        
        
        # print("snplist: ", snplist_temp)
        # print("out_prefix: ", out_prefix_temp)
        
        
        cmd = [
            _gcta,
            "--bfile", _fpath_ref_bfile,
            "--cojo-file", _fpath_ma,
            "--cojo-cond", snplist_temp,
            "--out", out_prefix_temp
        ]
        # print("\nExecuting command:\n")
        # print(json.dumps(cmd, indent='\t'))
        
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        except subprocess.CalledProcessError as e:
            # 에러 메시지를 Jupyter Notebook에 출력
            # print(f"Error occurred: {e}")

            # stderr을 파일로 기록
            with open(out_prefix_temp + ".stderr", 'w') as f_stderr:
                f_stderr.write(f"Error occurred while running GCTA64:\n{e}\n")
                
            """
            추후 여기서 마지막 fail한 gcta execution log에서 "colinearity" 관련 단어가 확인되면 "Done"이라고 출력해주는 코드 몇줄만 추가하셈.
            """

            break  # 에러 발생 시 루프 종료

        else:

            OUT_CMA = out_prefix_temp + ".cma.cojo"
                        
            df_out_sort = sort_COJO_result_cond(OUT_CMA)
            df_out_sort.to_csv(OUT_CMA + ".sort", sep='\t', header=True, index=False, na_rep="NA")
            # display(df_out_sort.head(5))



            ### To use fine-mapping for the next top?
            if _f_use_finemapping:

                df_PP_cma, OUT_PP_cma = fine_mapping_SWCA.__MAIN__(
                    OUT_CMA,
                    _fpath_ref_bfile, _fpath_ref_ld,
                    dirname(OUT_CMA),
                    _plink
                )

                print(df_PP_cma.head(10))

                ## credible set에 있는 애들 확인.
                df_PP_cma = df_PP_cma[ df_PP_cma['CredibleSet'] ] # 2개 이상의 rows들일 수 있음.

                l_next_top_signal = df_PP_cma['SNP'].tolist()
                next_top_signal_pC = \
                        df_out_sort.loc[ df_out_sort['SNP'].isin(l_next_top_signal) , 'pC'].min()

            else:
                l_next_top_signal = [df_out_sort['SNP'].iat[0]]
                next_top_signal_pC = df_out_sort['pC'].iat[0]



            ### the top signal as single-factor markers? (without co-linearity)

            if _f_single_factor_markers:

                l_next_top_signal, df_ToRefer = get_single_residue_markers(l_next_top_signal, _fpath_ref_MAF, df_out_sort)
                print(f"df_ToRefer:\n{df_ToRefer}")


            ### End 조건: No more significant signals?
            if next_top_signal_pC < 5e-8:
            
                l_secondary_signals.extend(l_next_top_signal)
                print("\nNext conditions: ", l_secondary_signals)
                
            else:
                
                print("No more significant loci!")
                break
        
        # if i >= 2: break
    
    return l_secondary_signals



def __MAIN__(_fpath_ss, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _fpath_PP, _out_prefix, _N,
             _f_include_SNPs=False, _f_use_finemapping=True, _f_single_factor_markers=True,
             _maf=0.01,
             _gcta="/home/wschoi/bin/gcta64", _plink="/home/wschoi/miniconda3/bin/plink"):
    
    """
    - Main wrapper for 'SWCA.py'
    """

    ##### (0) load data

    df_LDmatrix = pd.read_csv(_fpath_ref_ld, sep='\t', header=0) \
                        if isinstance(_fpath_ref_ld, str) else _fpath_ref_ld
    df_LDmatrix.index = df_LDmatrix.columns
        # next top 찾을 때 fine-mapping 활용하게 되면 LDmatrix 한 번 load하고 재사용하게 해야 함.

    df_MAF = pd.read_csv(_fpath_ref_MAF, sep='\s+', header=0) \
                    if isinstance(_fpath_ref_MAF, str) else _fpath_ref_MAF

    df_ss_SNP = pd.read_csv(_fpath_ss, sep='\t', header=0) \
                    if isinstance(_fpath_ss, str) else _fpath_ss


    
    ##### (1) Summary Imputation

    df_Z_imputed, _ = mod_SummaryImp.__MAIN__(df_ss_SNP, df_LDmatrix)
    
    """
    (No axis name)    Conditional_mean
    SNP_A_30018316	-1.328578
    AA_A_-22_30018317_I	-1.328578    
    """



    ##### (2) make the COJO input

    df_ma = make_COJO_input(df_Z_imputed, df_MAF, _N, _maf=_maf,
                            _df_ss_SNP=(df_ss_SNP if _f_include_SNPs else None))

    OUT_ma = _out_prefix + ".ma"
    df_ma.to_csv(OUT_ma, sep='\t', header=True, index=False, na_rep="NA")

    """
    - HLA loci들에 한정해서 Z_imputed가 나옴.
        - 필요하면 `_fpath_ss`의 SNPs들도 함께 넣을 수 있음.
    - MAF filtering은 `make_COJO_input()`함수 내에서 적용함.
    """


    
    ##### (3) iterate GCTA COJO "--cojo-cond"
    
    ### (3-1) the top PP locus to start
    df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0) \
                .sort_values("PP", ascending=False) \
                .iloc[:5, :]
    
    # display(df_PP)
    
    initial_top_signal = df_PP['SNP'].iat[0]
    print("initial_top_signal: {}".format(initial_top_signal))

    
    ### (3-2) new out_prefix for COJO
    basename_temp = os.path.basename(_out_prefix) + ".ROUND_"
    dirname_temp = os.path.dirname(_out_prefix)
    
    out_dir_COJO = os.path.join(dirname_temp, basename_temp)
    out_prefix_COJO = os.path.join(out_dir_COJO, basename_temp)
    
    os.makedirs(out_dir_COJO, exist_ok=True)
    
    l_secondary_signals = iterate_GCTA_COJO(
        OUT_ma, initial_top_signal,
        _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF,
        out_prefix_COJO,
        _f_use_finemapping=_f_use_finemapping, _f_single_factor_markers=_f_single_factor_markers,
        _gcta=_gcta, _plink=_plink
    )

    
    return l_secondary_signals, OUT_ma



if __name__ == '__main__':

    ##### `iterate_GCTA_COJO` 테스트

    ### HLA only

    # r_temp = iterate_GCTA_COJO(
    #     "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST.HLA.ma",
    #     "AA_DRB1_13_32660109_HF",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq",
    #     "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST_2.HLA.ROUND_",
    #     _f_use_finemapping=True, _f_single_factor_markers=True,
    #     _N_max_iter=2
    # )
    #
    # print("Final secondary signals:")
    # print(r_temp)



    ##### `__MAIN__` test.

    r_temp = __MAIN__(
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.M613.sumstats3",
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.LD.whole.M3703.ld",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.PP",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250418_TEST_2.HLA",
        58284,
        _f_include_SNPs=False
    )
    print("Final secondary signals:")
    print(r_temp)



    pass



"""
(Usage example)

r = __MAIN__(
    d_RA_yuki_clumped['sumstats'], d_RA_yuki_clumped['ld'], d_RA_yuki_clumped['MAF'],
    "20250218_ImpG_HLA_v3/RA_yuki.EUR.CovModel.whole.whole.PP",
    "20250218_ImpG_HLA_v3/TEST.RA_yuki", 
    58284
)

"""