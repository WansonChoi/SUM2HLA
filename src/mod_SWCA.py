import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd

import math

from datetime import datetime

import subprocess
import json

import mod_SummaryImp



########## (1) Summary imputation

## 위 `mod_SummaryImp` import함.



########## (2) make the COJO input

def make_COJO_input(_df_ss_imputed_HLA, _df_MAF, _N, _df_ss_SNP=None, _maf=0.01):
    
    ##### Required columns: [SNP A1 A2 freq b se p N]
    l_header_final = ["SNP", "A1", "A2", "freq", "b", "se", "p", "N"]

    
    
    ##### (1) MAF
    ## 그니께, ['A1', 'A2'] 때문에라도 어차피 df_MAF에 join해야함.
    df_MAF_2 = _df_MAF.drop(['CHR', 'NCHROBS'], axis=1).rename({"MAF": "freq"}, axis=1)
    # display(df_MAF_2)
    
    
    
    ##### (2) Imputed (HLA loci)
    # display(_df_ss_imputed_HLA)
    
    df_imputed = _df_ss_imputed_HLA \
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
    
    
    
    ##### (3) Unimputed (a Clumped GWAS summary)
    
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



def iterate_GCTA_COJO(_fpath_ma, _initial_top_signal, _fpath_ld_bfile, _out_prefix, _N_max_iter=100, 
                      _f_follow_HLAmarkers=True,
                     _gcta="/home/wschoi/bin/gcta64"):
    
    ## 얘가 main varaible
    l_top_signals = [_initial_top_signal] if isinstance(_initial_top_signal, str) else _initial_top_signal
        
        
    ##### Main iteration
    
    for i in range(1, _N_max_iter+1):
        
        print("===[{}]: ROUND_{}.".format(i, i))
        # print("Current conditions:\n{}".format(l_top_signals))
        
        snplist_temp = os.path.join(_out_prefix + "{}.snplist".format(i))
        out_prefix_temp = re.sub(r'.snplist$', '', snplist_temp)

        sr_snplist = pd.Series(l_top_signals, name="snptlist")
        # display(sr_snplist)
        sr_snplist.to_csv(snplist_temp, header=False, index=False, na_rep="NA")
        
        
        # print("snplist: ", snplist_temp)
        # print("out_prefix: ", out_prefix_temp)
        
        
        cmd = [
            _gcta,
            "--bfile", _fpath_ld_bfile,
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
                        
            df_out_sort = sort_COJO_result_cond(out_prefix_temp + ".cma.cojo")
            df_out_sort.to_csv(out_prefix_temp + ".cma.cojo.sort", sep='\t', header=True, index=False, na_rep="NA")

            ### 당분간은 ma에 HLA locus markers들만 주어질 거기 때문에 mute하는 거임. 엄밀하게 필요한 부분임.
            # if _f_follow_HLAmarkers:

            #     f_AA = df_out_sort['SNP'].str.startswith("AA")
            #     f_SNP_intra = df_out_sort['SNP'].map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x))
            #     f_HLA = df_out_sort['SNP'].str.startswith("HLA")
            #     f_INS = df_out_sort['SNP'].str.startswith("INS_")

            #     f_HLAs = f_AA | f_SNP_intra | f_HLA | f_INS

            #     df_out_sort = df_out_sort[f_HLAs]
            
            # display(df_out_sort.head(5))

            next_top_signal = df_out_sort['SNP'].iat[0]
            next_top_signal_pC = df_out_sort['pC'].iat[0]
            
            if next_top_signal_pC < 5e-8:
            
                l_top_signals.append(next_top_signal)
                # print("\nNext conditions: ", l_top_signals)
                
            else:
                
                print("No more significant loci!")
                break
        
        # if i >= 2: break
    
    return l_top_signals



def __MAIN__(_fpath_ss, _fpath_ld_matrix, _fpath_ld_bfile, _fpath_ld_MAF, _fpath_PP, _out_prefix, _N, 
             _df_ss_SNP=None, _maf=0.01, _f_follow_HLAmarkers=True, _gcta="/home/wschoi/bin/gcta64"):
    
    """
    - Main wrapper for 'mod_SWCA.py'
    """
    
    ##### (1) Summary Imputation
    df_Z_imputed, _ = mod_SummaryImp.__MAIN__(_fpath_ss, _fpath_ld_matrix)
    
    """
    (No axis name)    Conditional_mean
    SNP_A_30018316	-1.328578
    AA_A_-22_30018317_I	-1.328578    
    """
    
    ##### (2) make the COJO input
    df_MAF = pd.read_csv(_fpath_ld_MAF, sep='\s+', header=0)
    df_ma = make_COJO_input(df_Z_imputed, df_MAF, _N, _maf=_maf, _df_ss_SNP=_df_ss_SNP)

    """
    - HLA loci들에 한정해서 Z_imputed가 나옴.
        - SNPs들도 포함하고 싶으면 `_df_ss_SNP`에 summary주면 됨.
    - MAF filtering은 `make_COJO_input()`함수 내에서 적용해서 return됨.
    - 요약하면, `df_ma`에 (1) HLA locus markers들만 있고 (2) MAF thresholding된 결과임.
    """
    
    out_ma = _out_prefix + ".ma"
    df_ma.to_csv(out_ma, sep='\t', header=True, index=False, na_rep="NA")


    
    ##### (3) iterate GCTA COJO "--cojo-cond"
    
    ### (3-1) the top PP locus to start
    df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0) \
                .sort_values("PP", ascending=False) \
                .iloc[:5, :]
    
    # display(df_PP)
    
    initial_top_signal = df_PP['SNP'].iat[0]
    print("initial_top_signal: {}".format(initial_top_signal))

    
    ### (3-2) new out_prefix for COJO
    basemae_temp = os.path.basename(_out_prefix) + ".ROUND_"
    dirname_temp = os.path.dirname(_out_prefix)
    
    out_dir_COJO = os.path.join(dirname_temp, basemae_temp)
    out_prefix_COJO = os.path.join(out_dir_COJO, basemae_temp)
    
    os.makedirs(out_dir_COJO, exist_ok=True)
    
    l_secondary_signals = iterate_GCTA_COJO(
        out_ma, initial_top_signal, _fpath_ld_bfile, out_prefix_COJO,
        _f_follow_HLAmarkers=_f_follow_HLAmarkers,
        _gcta=_gcta
    )

    
    return l_secondary_signals, out_ma



"""
(Usage example)

r = __MAIN__(
    d_RA_yuki_clumped['sumstats'], d_RA_yuki_clumped['ld'], d_RA_yuki_clumped['MAF'],
    "20250218_ImpG_HLA_v3/RA_yuki.EUR.CovModel.whole.whole.PP",
    "20250218_ImpG_HLA_v3/TEST.RA_yuki", 
    58284
)

"""