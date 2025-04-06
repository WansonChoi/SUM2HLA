import os, sys, re
import numpy as np
import pandas as pd
import math

from datetime import datetime

from src.mod_LDmatrix_class import LDmatrix
from src.mod_Util import get_N_of_ss



class GWAS_summary():

    def __init__(self, _fpath, _LDmatrix:LDmatrix):

        ########## Main variables
        self.sr_GWAS_summary = None
        self.N = None


        ########## Main

        ##### (1) load the GWAS summary
        
        if isinstance(_fpath, str):

            self.sr_GWAS_summary = \
                pd.read_csv(_fpath, sep='\t', header=0) \
                    .loc[:, ['SNP_LD', 'Z_fixed']] \
                    .rename({"SNP_LD": "SNP", "Z_fixed": "Z"}, axis=1) \
                    .set_index("SNP") \
                    .squeeze('columns')

            self.N = get_N_of_ss(_fpath)

        elif isinstance(_fpath, pd.DataFrame):

            self.sr_GWAS_summary = \
                _fpath \
                    .loc[:, ['SNP_LD', 'Z_fixed']] \
                    .rename({"SNP_LD": "SNP", "Z_fixed": "Z"}, axis=1) \
                    .set_index("SNP") \
                    .squeeze('columns')

            self.N = _fpath['N'].iat[0]

        else:
            raise ValueError("Wrong GWAS input!")


        
        ##### (2) match the SNPs to that of the LD matrix
        sr_SNP_ToExtract = _LDmatrix.df_LD_SNP.columns.intersection(self.sr_GWAS_summary.index)
        self.sr_GWAS_summary = self.sr_GWAS_summary.loc[sr_SNP_ToExtract]
            ## 여기서 LDmatrix의 SNPs들로 subset 되고, LDmatrix SNP order로 맞춰짐.





    def __repr__(self):

        str_ss = f"Loaded GWAS summary: {self.sr_GWAS_summary}"
        str_N  = f"N: {self.N}"

        l_str = [str_ss, str_N]

        return '\n'.join(l_str)