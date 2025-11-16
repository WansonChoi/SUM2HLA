# %load_ext autoreload
# %autoreload 2


import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd

import json

import subprocess

from shutil import which

from datetime import datetime

"""
- 이 파일이 마지막으로 touch된게 2025.10.07.임.
- 내가 WTCCC 재분석한것도 이 날짜임.
    - '20251007_WTCCC_rerun_v2/'
- 그때 ipynb에 남겨놨던거 여기로 백업해놓은거같음.


"""



def SWCA_GT_naive(_trait:str, _bfile:str, _a1_allele:str, _pheno:str, _out_prefix:str,
                 _top_signal_ROUND_0=None, _fpath_assoc_ROUND_0=None, _extract=None,
                 _N_max_iter=5, _plink="/home/wschoi/miniconda3/envs/jax_gpu/bin/plink"):

    ##### Main variables
    l_top_signals = []
    out_dir = os.path.dirname(_out_prefix)
    os.makedirs(out_dir, exist_ok=True)

    
    print(f"out_dir: {out_dir}")


    
    ##### (1) ROUND_0 갈무리.

    if isinstance(_top_signal_ROUND_0, str) and (_fpath_assoc_ROUND_0 == None):
        l_top_signals = [_top_signal_ROUND_0]
        
    elif (_top_signal_ROUND_0 == None) and isinstance(_fpath_assoc_ROUND_0, str):
        df_assoc = pd.read_csv(_fpath_assoc_ROUND_0, sep='\t', header=0) \
                    .dropna() \
                    .sort_values("P")
        print(df_assoc.head(10))
        
        l_top_signals = [df_assoc.iloc[0, 1]] # 'SNP' column이 2번째 column으로 고정됐다 가정.
    elif isinstance(_top_signal_ROUND_0, list):
        l_top_signals = _top_signal_ROUND_0
    else:
        raise ValueError("Invalid assoc result or top signal label!")

    
    print(f"l_top_signals: {l_top_signals}")

    
    
    ##### (2) iteration starting from ROUND_1.
    
    for i in range(1, _N_max_iter+1):
        
        print("===[{}]: ROUND_{}.".format(i, i))
        # print("Current conditions:\n{}".format(l_top_signals))

        ### condition list and output prefix of the iteration.
        snplist_temp = os.path.join(_out_prefix + "{}.snplist".format(i))
        out_prefix_temp = re.sub(r'.snplist$', '', snplist_temp)

        ### export
        sr_snplist = pd.Series(l_top_signals, name="snptlist")
        # display(sr_snplist)
        sr_snplist.to_csv(snplist_temp, header=False, index=False, na_rep="NA")
        
        
        # print("snplist: ", snplist_temp)
        # print("out_prefix: ", out_prefix_temp)
        
        
        cmd = [
            _plink,
            "--logistic", "hide-covar", "--ci", "0.95",
            "--bfile", _bfile,
            "--a1-allele", _a1_allele,
            "--pheno", _pheno,
            "--pheno-name", _trait,
            "--condition-list", snplist_temp,
            "--out", out_prefix_temp,
            "--allow-no-sex",
            "--keep-allele-order"
        ]
        # print("\nExecuting command:\n")
        # print(json.dumps(cmd, indent='\t'))

        if isinstance(_extract, str):
            cmd.extend(["--extract", _extract])
        
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        except subprocess.CalledProcessError as e:

            # stderr을 파일로 기록
            with open(out_prefix_temp + ".stderr", 'w') as f_stderr:
                f_stderr.write(f"Error occurred while running GCTA64:\n{e}\n")
                
            """
            추후 여기서 마지막 fail한 gcta execution log에서 "colinearity" 관련 단어가 확인되면 "Done"이라고 출력해주는 코드 몇줄만 추가하셈.
            """

            break  # 에러 발생 시 루프 종료

        except Exception as e_else:
            print(f"{e_else}")

        else:

            ##### 다음 iteration 준비.

            df_assoc_sort = pd.read_csv(
                out_prefix_temp + ".assoc.logistic", sep=r'\s+', header=0
            ) \
                .sort_values("P")

            ## export
            df_assoc_sort.to_csv(out_prefix_temp + ".assoc.logistic.sort", sep='\t', header=True, index=False, na_rep="NA")


            next_top_signal = df_assoc_sort['SNP'].iat[0]
            next_top_signal_P = df_assoc_sort['P'].iat[0]
            
            if next_top_signal_P < 5e-8:
            
                l_top_signals.append(next_top_signal)
                # print("\nNext conditions: ", l_top_signals)
                
            else:
                
                print("No more significant loci!")
                break
        
        # if i >= 2: break
    
    return l_top_signals


"""
Usage example

### RA
SWCA_GT_naive(
    "RA", 
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb",
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb.a1_allele",
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb.phes",
    "/data02/wschoi/_hCAVIAR_v2/20250715_WTCCC_rerun/20250929.PLINK.SWCA_n.WTCCC.RA.AA_DRB1_11_32660115_SPG.ROUND_",
    _top_signal_ROUND_0="AA_DRB1_11_32660115_SPG", # hCAVIAR top
    # _top_signal_ROUND_0="AA_DRB1_13_32660109_HF", # PLINK top
    _extract="/data02/wschoi/_hCAVIAR_v2/20250715_WTCCC_rerun/REF_T1DGC.hg19.HLA.M3090.ToExtract",
    _plink="/home/wschoi/miniconda3/bin/plink"
)

### T1D
SWCA_GT_naive(
    "T1D", 
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb",
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb.a1_allele",
    "20250408_SWCA-GT/IMPUTED.WTCCC.58C+NBS+RA+CD+T1D.hg19.chr6.29-34mb.phes",
    "/data02/wschoi/_hCAVIAR_v2/20250715_WTCCC_rerun/20250929.PLINK.SWCA_n.WTCCC.RA.AA_DQB1_57_32740666_D.ROUND_",
    _top_signal_ROUND_0="AA_DQB1_57_32740666_D", # hCAVIAR의 top은 얘.
    # _top_signal_ROUND_0="SNP_DQB1_32740666_G", # PLINK logistic은 얘가 top
    _extract="/data02/wschoi/_hCAVIAR_v2/20250715_WTCCC_rerun/REF_T1DGC.hg19.HLA.M3090.ToExtract",
    _plink="/home/wschoi/miniconda3/bin/plink"
)

"""