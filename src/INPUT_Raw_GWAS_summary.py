import os, sys, re
from os.path import exists, dirname, basename, join
import numpy as np
import scipy as sp
import pandas as pd
import math

from shutil import which
import subprocess

from datetime import datetime

import src.Util as Util

"""
- GWAS catalog에서 download한 raw summary가 주어졌을 때,
    - 이 raw summary의 기본 정보들 확인 (ex. columns, HLA region내 marker 몇 개인지)
    - 전처리를 수행해서 SUM2HLA에서 쓸 수 있도록 (최소한 header는 손 봐야함.)


- 필요한 전처리들만 나열하면 됨.
    - liftover
    - HLA region
    - Duplicated BP warning
    - export

    
%load_ext autoreload
%autoreload 2

/data02/wschoi/hg38ToHg19.over.chain
"""



class INPUT_Raw_GWAS_summary():


    def __init__(self, _df_ss_raw, _hg, _out, _col_SNP=None,
                 _bin_liftOver=which("liftOver"), _fpath_hg38_to_19=None, _fpath_hg18_to_19=None,
                 _df_ref_bim=None):
        

        self.required_columns = ["CHR", "SNP", "BP", "A1", "A2", "BETA", "SE", "Z", "P", "N"]
        
        ### OUTPUT
        self.out = _out
        self.out_dir = dirname(_out)
    
        self.hg = _hg

        self.df_ss_raw = _df_ss_raw



        ### labels to rename
        self.d_ToRename = {
            "chromosome": "CHR",
            "base_pair_location": "BP",
            "effect_allele": "A1",
            "other_allele": "A2",
            'n': "N",
            "beta": "BETA",
            "standard_error": "SE",
            "effect_allele_frequency": "AF_A1",
            "p_value": "P_original",

            "odds_ratio": "OR", # 'ci_upper', 'ci_lower' 이렇게 두 개가 있어야 함.

        }
        if isinstance(_col_SNP, str):
            self.d_ToRename.update({_col_SNP: "SNP"})


        ### Duplication removal
        self.df_ref_bim = _df_ref_bim # 이거 SNP이든 SNP + HLA이든 받고 SNP만 캐오도록 바꿔야 할듯.



        ### external software
        self.bin_liftOver = _bin_liftOver
        self.out_UCSCbed = _out + f".UCSC.hg{self.hg}.bed"


        self.out_UCSCbed_hg19 = _out + f".UCSC.hg19.bed"
        self.out_UCSCbed_hg19_unmapped = _out + f".UCSC.hg19.bed.unmapped"

        self.hg38_to_19 = _fpath_hg38_to_19
        self.hg18_to_19 = _fpath_hg18_to_19
    


    @classmethod # Factory method 1
    def from_paths(cls, _fpath_ss_raw, **kwargs):

        df_ss_raw = pd.read_csv(_fpath_ss_raw, sep='\t', header=0)

        return cls(df_ss_raw, **kwargs)



    def rename_colnames(self):

        """
        - Required columns: ["CHR", "SNP", "BP", "A1", "N", "SE", "Z", "P", "A2"]

        - rename하고 시작해야 downstream이 편함.

        """



        self.df_ss_raw.rename(self.d_ToRename, axis=1, inplace=True)
            ## inplace로 처리해서 input으로 받은애가 바뀌는거 잊지마셈.

        return self.df_ss_raw



    def remove_NA_BP(self):

        """
        ## MVP summary의 경우 BP가 NA인 items들도 있었음.

                chromosome  base_pair_location effect_allele other_allele  odds_ratio  standard_error  effect_allele_frequency  ...  num_cases control_af  num_controls      r2 q_pval  i2  direction
        228597             1                 NaN             G            A      4.2380             NaN                  0.99490  ...        607     0.9949        121151  0.9550    NaN NaN        NaN
        411704             1                 NaN             C            A      0.8876             NaN                  0.98730  ...        607     0.9873        121151  0.7918    NaN NaN        NaN
        427728             1                 NaN             A            G      1.7020             NaN                  0.99640  ...        607     0.9963        121151  0.4096    NaN NaN        NaN
        1563267            1                 NaN             G            A      0.5917             NaN                  0.96110  ...        607     0.9612        121151  0.4177    NaN NaN        NaN
        2464027            1                 NaN             A            T      0.8708             NaN                  0.98940  ...        607     0.9894        121151  0.8645    NaN NaN        NaN
        3392696            2                 NaN             C            T      1.0260             NaN                  0.97020  ...        607     0.9702        121151  0.6609    NaN NaN        NaN
        4095236            2                 NaN             C            T      1.2140             NaN                  0.99780  ...        607     0.9978        121151  0.7646    NaN NaN        NaN

        ...
        
        """

        self.df_ss_raw.dropna(subset=['BP'], inplace=True)
        self.df_ss_raw['BP'] = self.df_ss_raw['BP'].astype(int)

        return self.df_ss_raw



    def check_CHR(self):

        ### 잠정적으로 "CHR"이 chr6 이런식으로 주어질 때가 있음.

        return 0



    def extract_HLAregion(self):

        ## hg 상관없이 chr6:28-35mb로 자르기.

        f_chr6 = self.df_ss_raw['CHR'] == 6
        f_HLAregion = (self.df_ss_raw['BP'] >= 28_000_000) & (self.df_ss_raw['BP'] < 35_000_000)

        self.df_ss_raw = self.df_ss_raw[f_chr6 & f_HLAregion]

        return self.df_ss_raw
    


    def recalc_BETA_SE_from_OR_CI(self):

        f_OR = "OR" in self.df_ss_raw.columns
        f_CI_upper = "ci_upper" in self.df_ss_raw.columns
        f_CI_lower = "ci_lower" in self.df_ss_raw.columns

        if not (f_OR and f_CI_upper and f_CI_lower):
            print("failed to calculate BETA and SE based on OR and CIs.")
            return -1

        ### (1) BETA based on log(OR)
        sr_BETA = pd.Series(np.log(self.df_ss_raw['OR']), name='BETA', index=self.df_ss_raw.index)


        ### (2) SE based on CIs
        arr_SE = (np.log(self.df_ss_raw["ci_upper"]) - np.log(self.df_ss_raw["ci_lower"])) / (2 * 1.96)
        sr_SE = pd.Series(arr_SE, name='SE', index=self.df_ss_raw.index)


        l_ToDrop = ['BETA', 'SE'] # 쓰레기 값으로 column이 이미 준비되어 있는 경우.

        self.df_ss_raw = pd.concat([self.df_ss_raw.drop(l_ToDrop, axis=1, errors='ignore'), sr_BETA, sr_SE], axis=1)
        
        return self.df_ss_raw



    def recalc_Pvalue(self):

        sr_Z = (self.df_ss_raw['BETA'] / self.df_ss_raw['SE']).rename("Z")

        sr_P = pd.Series(
            2 * sp.stats.norm.cdf(-sr_Z.abs()),
            name='P',
            index = self.df_ss_raw.index
        )

        self.df_ss_raw = pd.concat([self.df_ss_raw, sr_Z, sr_P], axis=1)

        return self.df_ss_raw



    def do_Liftover(self):

        if self.bin_liftOver == None:
            print(f"No 'liftOver' binary file found! ({self.bin_liftOver})")
            return -1
        
        if isinstance(self.bin_liftOver, str) and not exists(self.bin_liftOver):
            print(f"No 'liftOver' binary file exist! ({self.bin_liftOver})")
            return -1

        if self.hg == 38 and self.hg38_to_19 == None:
            print("No proper chain file for hg38 to hg19 liftOver!")
            return -1

        if self.hg == 18 and self.hg18_to_19 == None:
            print("No proper chain file for hg18 to hg19 liftOver!")
            return -1



        ##### (1) UCSC bed 파일 만들기.
        sr_CHR = self.df_ss_raw['CHR'].map(lambda x: f"chr{x}")
        sr_BP = self.df_ss_raw['BP'].astype(int)
        sr_SNP = self.df_ss_raw['SNP']

        df_UCSC_bed = pd.concat(
            [
                sr_CHR,
                (sr_BP - 1).rename("start"),
                sr_BP.rename("end"),
                sr_SNP
            ],
            axis=1
        )

        df_UCSC_bed \
            .drop_duplicates(subset=['CHR', 'end', 'SNP']) \
            .to_csv(self.out_UCSCbed, sep='\t', header=False, index=False, na_rep="NA")

        ## (cf) drop_duplicates()안하면 바로 다음 merge()하고 나서 duplicated items()들이 2배로 뿔어남. (2 x 2 = 4)


        ##### (2) liftOver run.

        CMD = f"{self.bin_liftOver} {self.out_UCSCbed} {self.hg38_to_19} {self.out_UCSCbed_hg19} {self.out_UCSCbed_hg19_unmapped}"

        result_liftover = subprocess.run(CMD.split(), capture_output=True, text=True)


        if result_liftover.returncode != 0:
            print("Failed to run liftOver!")
            print(result_liftover)
            return -1
        

        ##### (3) applying hg19 BP
        df_UCSC_bed_hg19 = pd.read_csv(
            self.out_UCSCbed_hg19, sep='\t', header=None, 
            names=['CHR', 'BP_start', 'BP', 'SNP'], usecols=['SNP', 'BP']
        )

        ## unmapped BPs들 발생.
        # if self.df_ss_raw.shape[0] != df_UCSC_bed_hg19.shape[0]:
        #     print("There are unmapped BPs!")
        #     return -1
        
        self.df_ss_raw = df_UCSC_bed_hg19.merge(
            self.df_ss_raw.rename({"BP": f"BP_hg{self.hg}"}, axis=1), 
            on=['SNP'], how='outer') \
                .sort_values("BP")


        return self.df_ss_raw



    def filter_ref_SNPs(self):

        ## duplication removal을 위해 main 전처리 몇 개를 여기서 미리 수행.
        ## 이 함수를 call하지 않는다면 main 전처리에서 수행할 작업들임.

        if isinstance(self.df_ref_bim, pd.DataFrame):
            pass
        elif isinstance(self.df_ref_bim, str):
            self.df_ref_bim = pd.read_csv(
                self.df_ref_bim, sep='\t', header=None, names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])
        else:
            print("Wrong input for `df_ref_bim` !")
            return -1
        

        ### SNP만 남기기
        f_isHLA = Util.is_HLA_locus(self.df_ref_bim.iloc[:, 1])
        self.df_ref_bim = self.df_ref_bim[~f_isHLA]
        

        def complement_base(_x):
            if _x == 'A': return 'T'
            elif _x == 'C': return 'G'
            elif _x == 'G': return 'C'
            elif _x == 'T': return 'A'
            else:
                return -1

        sr_allele_pairs = self.df_ss_raw[['A1', 'A2']].apply(lambda x: set(x), axis=1).rename("allele_pair1")
        sr_allele_pairs_2 = self.df_ss_raw[['A1', 'A2']].applymap(lambda x: complement_base(x)) \
                                .apply(lambda x: set(x), axis=1).rename("allele_pair2")

        sr_BP_allele_pairs = pd.concat([self.df_ss_raw['BP'], sr_allele_pairs], axis=1).apply(lambda x: tuple(x), axis=1)
        sr_BP_allele_pairs_2 = pd.concat([self.df_ss_raw['BP'], sr_allele_pairs_2], axis=1).apply(lambda x: tuple(x), axis=1)


        sr_REF_allele_pairs = self.df_ref_bim[['A1', 'A2']].apply(lambda x: set(x), axis=1).rename("REF_allele_pair1")
        sr_REF_BP_allele_pairs = pd.concat([self.df_ref_bim['BP'], sr_REF_allele_pairs], axis=1).apply(lambda x: tuple(x), axis=1)



        f_isin_1 = sr_BP_allele_pairs.isin(sr_REF_BP_allele_pairs)
        f_isin_2 = sr_BP_allele_pairs_2.isin(sr_REF_BP_allele_pairs)
        f_isin = f_isin_1 | f_isin_2

        self.df_ss_raw = self.df_ss_raw[f_isin]

        return self.df_ss_raw



    def sort_columns(self):

        self.df_ss_raw = pd.concat(
            [
                self.df_ss_raw.loc[:, self.required_columns],
                self.df_ss_raw.drop(self.required_columns, axis=1)
            ],
            axis=1
        ) \
            .sort_values("BP")

        return self.df_ss_raw



    def run(self):

        self.rename_colnames()
        self.extract_HLAregion()

        if self.hg != 19:
            self.do_Liftover()

        self.recalc_Pvalue()

        self.sort_columns()

        ### export
        self.df_ss_raw.to_csv(self.out, sep='\t', header=True, index=False, na_rep="NA")

        return self.df_ss_raw



    def run_MVP(self):

        self.rename_colnames()
        self.remove_NA_BP()

        self.extract_HLAregion()

        if self.hg != 19:
            self.do_Liftover()

        self.filter_ref_SNPs()

        self.recalc_BETA_SE_from_OR_CI()
        self.recalc_Pvalue()

        self.sort_columns()

        ### export
        self.df_ss_raw.to_csv(self.out, sep='\t', header=True, index=False, na_rep="NA")

        return self.df_ss_raw
