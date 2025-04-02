import os, sys, re
import numpy as np
import pandas as pd
import math

from datetime import datetime



class LDmatrix():
    
    def __init__(self, _fpath):
        
        ##### Main variables
        
        ### dim: SNP + HLA
        self.fpath = _fpath
        self.df_LD = None
        
        
        ### dim: SNP
        self.l_ix_SNPs = []
        self.df_LD_SNP = None
        
        self.df_LD_SNP_inv = None
        
        self.eigenvalues = None
        self.eigenvectors = None
        
        self.term1 = None # No longer use.
        self.term2 = None
        
        
        ##### (1) load raw LD matrix
        
        if isinstance(_fpath, str):
            self.df_LD = pd.read_csv(_fpath, sep='\t', header=0)
            self.df_LD.index = self.df_LD.columns
        elif isinstance(_fpath, pd.DataFrame):
            self.fpath = "DataFrame Given."
            self.df_LD = _fpath.copy()
            self.df_LD.index = self.df_LD.columns
        else:
            # print("Wrong input!")
            raise ValueError("Incorrect input was given: {}".format(_fpath))

        
        
        ##### (2) subset SNPs and their integer index (for numpy `np.ix_`)
        
        sr_SNP_HLA = self.df_LD.columns.to_series().reset_index(drop=True)

        ### SNPs들의 위치 파악.
        f_AA = sr_SNP_HLA.str.startswith("AA")
        f_SNP_intra = sr_SNP_HLA.map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x))
        f_HLA = sr_SNP_HLA.str.startswith("HLA")
        f_INS = sr_SNP_HLA.str.startswith("INS_")

        f_HLAs = f_AA | f_SNP_intra | f_HLA | f_INS
        f_SNPs = ~f_HLAs
        
        self.l_ix_SNPs = sr_SNP_HLA[f_SNPs].index.tolist()
        
        
        
        ##### (3) subset SNP + HLA to SNP (No HLA type markers)
        self.df_LD_SNP = self.df_LD.iloc[self.l_ix_SNPs, self.l_ix_SNPs]        

        
        
        ##### (4) calc eigen-things
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.df_LD_SNP.values)

        
        
        ##### (5) log-likelihood 계산을 위한 term 두 개
        """
        term1 = -0.5 * (ld_matrix.col_idx_SNP.size()) * std::log(2 * arma::datum::pi);
        term2 = -0.5 * arma::sum(arma::log(ld_matrix.eigenvalues_SNPs));
        """

        self.term1 = -0.5 * (self.df_LD_SNP.shape[0]) * math.log(2 * math.pi)
        self.term2 = -0.5 * np.sum(np.log(self.eigenvalues)) # sum(log(determinant))를 eigenvalues로 미리 계산하고 시작

        
                
    def print_info(self):
        
        print("df_LD:")
        display(self.df_LD)
        print(self.df_LD.shape)
        
        print("df_LD_SNP:")
        display(self.df_LD_SNP)
        print(self.df_LD_SNP.shape)
        
        # print("df_LD_SNP_inv:")
        # display(self.df_LD_SNP_inv)
        
        # print("term1: ", self.term1)
        
        print("All eigenvalues positive? (is PSD?): ", np.all(self.eigenvalues > 0))
        print("term2: ", self.term2)        
        
        return 0


        
    def __repr__(self):
        
        return "(loaded) {}".format(self.fpath)
    
    
        # str_RETURN = \
        #     "df_LD:\n{_df_LD}\n" \
        #     "df_LD_SNP:\n{_df_LD_SNP}\n" \
        #     "df_LD_SNP_inv:\n{_df_LD_SNP_inv}\n" \
        #     "term1: {_term1}\n" \
        #     "term2 (eigenvalues): {_term2}\n" \
        #     .format(
        #         _df_LD = self.df_LD,
        #         _df_LD_SNP = self.df_LD_SNP,
        #         _df_LD_SNP_inv = self.df_LD_SNP_inv,
        #         _term1 = self.term1,
        #         _term2 = self.term2
        #     )