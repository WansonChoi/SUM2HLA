import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd
import math
import json
from datetime import datetime
from shutil import which

import src.INPUT_prepr as INPUT_prepr
from src.INPUT_LDmatrix import INPUT_LDmatrix
from src.INPUT_GWAS_summary import INPUT_GWAS_summary
import src.hCAVIAR_PostCal_Cov as mod_PostCal_Cov
import src.SWCA as SWCA



class hCAVIAR_batch(): # a single run (batch) of hCAVIAR.

    def __init__(self, _ss_raw, _ref_prefix, _out_prefix,
                 _batch_size=30, _f_run_SWCR=True,
                 _out_json=None, _bfile_ToClump=None, _f_do_clump=True, # Utility arguments for testing.
                 _plink=None, _gcta=None
    ):

        ##### output path and prefix
        self.out_prefix = _out_prefix
        self.out_dir = os.path.dirname(_out_prefix)
        # self.out_basename = os.path.basename(_out_prefix)
        self.out_prefix_LD = None


        ##### The reference dataset.
        self.d_fpath_LD = {"whole": _ref_prefix + ".NoNA.PSD.ld"} # 예전에 HLA sub-region 별로 짤라서 활용하던대로 둠.
        self.fpath_LD_SNP_HLA = _ref_prefix if _bfile_ToClump is None else _bfile_ToClump
        self.fpath_LD_MAF = _ref_prefix + ".FRQ.frq"
        self.out_prefix_LD = _out_prefix + ".LD"

        
        ##### Raw GWAS summary file
        self.sumstats1 = _ss_raw


        ##### Intermediate output of the curation. ('mod_hCAVIAR_prepr')
        self.out_matched = None
        self.out_ToClump = None
        self.out_clumped = None # sumstats2
        self.out_json = _out_json # sumstats3 (***; ".hCAVIAR_input.json")


        ##### Curated input (LD and GWAS summary)
        self.LDmatrix:INPUT_LDmatrix = None
        self.GWAS_summary:INPUT_GWAS_summary = None

        self.LL_0 = None # LL-baseline

        
        ##### Model Parameters
        self.ncp = 5.2
        self.gamma = 0.01
        self.N_causal = 1 # 웬만하면 건드리지 마라.
        self.batch_size = _batch_size
        self.engine = "jax"


        ##### External software
        self.plink = which("plink") if bool(which("plink")) else _plink
        self.gcta64 = which("gcta") if bool(which("gcta")) else _gcta

        
        ##### Output (Accumulator)
        self.OUT_PIP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        self.OUT_LL_N_causal = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        
        self.OUT_PIP_PP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)} # PIP를 posterior로 normalize and export.
        self.OUT_PIP_PP_fpath = {_N_causal: {'whole': None} for _N_causal in range(1, self.N_causal + 1)} # filepath

        ##### SWCR with GCTA "--cojo-cond"
        # self.N = _N   # `GWAS_summary` class 내에서 가져오도록 바꿈.
        self.ma = None
        self.l_secondary_signals = []
        self.fpath_secondary_signals = None
        self.f_run_SWCR = _f_run_SWCR

        # self.out_cojo_slct = None


        ##### Clumping
        self.f_do_clump = _f_do_clump

        ##### Types of markers to calculate PP.
        self.l_type = ('whole', 'SNP', 'HLAtype', 'HLA', 'AA', 'intraSNP', 'AA+HLA')




    def run_hCAVIAR_prepr(self):
        
        ##### Interface with the `mod_hCAVIAR_prepr.hCAVIAR_prepr()`.
        ## 얘는 나중에 조건부로 안돌리게 만들어야 할 수도 있을 것 같아서 따로 떼어내놓게.

        self.out_matched, self.out_ToClump, self.out_clumped, self.out_json = \
            INPUT_prepr.__MAIN__(
                _fpath_ss=self.sumstats1,
                _d_fpath_LD=self.d_fpath_LD, _fpath_LD_SNP_bim=None, _fpath_LD_SNP_HLA=self.fpath_LD_SNP_HLA,
                _out_prefix_ss=self.out_prefix, _out_prefix_LD=self.out_prefix_LD,
                _f_do_clump=self.f_do_clump,
                _plink=self.plink
            )

        return self.out_matched, self.out_ToClump, self.out_clumped, self.out_json



    def run_SWCA(self, _N_causal=1, _type='whole'): # '_N_causal=1' 일 때만 한다 가정.

        if not os.path.exists(self.out_json):
            print("The '*.hCAVIAR_input.json' file not found! ('*.hCAVIAR_input.json'")
            return -1

        if not os.path.exists(self.OUT_PIP_PP_fpath[_N_causal][_type]):
            print("The posterior probability file not found!")
            return -1

        
        ### load the '*.hCAVIAR_input.json'
        with open(self.out_json, 'r') as f_json:
            d_curated_input = json.load(f_json)


        self.l_secondary_signals, self.ma = SWCA.__MAIN__(
            _fpath_ss=d_curated_input['whole']['sumstats'],
            _fpath_ref_ld=d_curated_input['whole']['ld'],
            _fpath_ref_bfile=self.fpath_LD_SNP_HLA,
            _fpath_ref_MAF=self.fpath_LD_MAF,
            _fpath_PP=self.OUT_PIP_PP_fpath[_N_causal]['whole'],
            _out_prefix=self.out_prefix,
            _N=self.GWAS_summary.N,
            _gcta=self.gcta64
        )

        pd.Series(self.l_secondary_signals, name='secondary_signal') \
            .to_csv(self.ma + ".SWCA.snplist", header=False, index=False, na_rep='NA')
        
        self.fpath_secondary_signals = self.ma + ".SWCA.snplist"

        return self.l_secondary_signals, self.ma, self.fpath_secondary_signals

        # self.out_cojo_slct = mod_SWCA.__MAIN__(
        #     _fpath_ss=d_curated_input['whole']['sumstats'],
        #     _fpath_ld_matrix=d_curated_input['whole']['ld'],
        #     _fpath_ld_bfile=self.fpath_LD_SNP_HLA,
        #     _fpath_ld_MAF=self.fpath_LD_MAF,
        #     _out_prefix=self.out_prefix,
        #     _N=self.N,
        #     _gcta=self.gcta64
        # )

        # return self.out_cojo_slct

    
    
    def __MAIN__(self):

        ##### has the curated main input?
        if self.out_json == None:
            print("\n\n==========[0]: preprocessing hCAVIAR input")
            self.run_hCAVIAR_prepr()


        
        ##### load the curated input (LD and GWAS summary)
        with open(self.out_json, 'r') as f_json:
            d_curated_input = json.load(f_json)

        self.LDmatrix = INPUT_LDmatrix(d_curated_input['whole']['ld'])
        self.GWAS_summary = INPUT_GWAS_summary(d_curated_input['whole']['sumstats'], self.LDmatrix)



        ##### calculate LL_0
        self.LL_0 = \
            self.LDmatrix.term2 \
            -0.5*( self.GWAS_summary.sr_GWAS_summary.values.T @ (np.linalg.solve(self.LDmatrix.df_LD_SNP.values, self.GWAS_summary.sr_GWAS_summary.values)) )
                # (2025.05.14.) 얘 잠정적으로 `mod_PostCal_Cov.__MAIN__()` 함수 안으로 집어넣었으면 좋겠음.
                # 'fine-mapping_SWCA.py'에서는 문제없이 집어넣었음.


        
        ##### run
        print("\n\n==========[1]: calculating LL (for each batch)")
        print("Batch size: {}".format(self.batch_size))
        for _N_causal in range(1, self.N_causal + 1):

            t_start_calcLL = datetime.now()
            
            self.OUT_PIP[_N_causal], self.OUT_LL_N_causal[_N_causal] = \
                mod_PostCal_Cov.__MAIN__(
                    _N_causal, self.GWAS_summary, self.LDmatrix, self.LL_0,
                    _batch_size=self.batch_size, _gamma=self.gamma, _ncp=self.ncp, _engine=self.engine
                )
        
            t_end_calcLL = datetime.now()

            print("\nTotal time for the LL when `N_causal`={}: {}".format(_N_causal, t_end_calcLL - t_start_calcLL))


        
        ##### Postprocessing
        print("\n\n==========[2]: postprocessing and export")
        for _N_causal in range(1, self.N_causal + 1):
            
            df_PP = pd.DataFrame({
                "SNP": self.LDmatrix.df_LD.columns,
                "LL+Lprior": self.OUT_PIP[_N_causal]
            })

            # df_PP.to_csv(self.out_prefix + ".before_postprepr.txt", sep='\t', header=True, index=False, na_rep="NA")
    
            self.OUT_PIP_PP[_N_causal] = \
                mod_PostCal_Cov.postprepr_LL(df_PP, _l_type=self.l_type) # 여기서 return되는건 DataFrame의 dictionary임.

            for _type, _df_PP in self.OUT_PIP_PP[_N_causal].items():

                out_temp = self.out_prefix + ".{}.PP".format(_type)
                _df_PP.to_csv(out_temp, sep='\t', header=True, index=False, na_rep="NA")

                ## `_df_PP`를 fwrite했으면 그냥 file path를 저장.
                self.OUT_PIP_PP_fpath[_N_causal][_type] = out_temp



        ##### SWCA
        if self.f_run_SWCR:
            print("\n\n==========[3]: Step-Wise Conditional Analysis with GCTA COJO.")
            self.run_SWCA() # `self.fpath_secondary_signals` 가 여기서 채워짐.
        
        
        
        return self.OUT_PIP, self.OUT_LL_N_causal, self.fpath_secondary_signals


    
    def __repr__(self):

        str_raw_ss = \
            "- GWAS summary: {}".format(self.sumstats1)

        str_ref_LD = \
            "- Reference LD file: {}".format(self.d_fpath_LD)

        str_ref_GT = \
            "- Reference genotype: {}".format(self.fpath_LD_SNP_HLA) if self.f_do_clump else \
            "- No Clumping!"

        
        str_curated_input = \
            "- Curated input: {}".format(self.out_json) if bool(self.out_json) else ""

        str_PP = \
            "- Output Posterior probability: {}".format(self.OUT_PIP_PP_fpath)

        ### external software
        str_plink = \
            "- plink: {}".format(self.plink)
        str_gcta64 = \
            "- gcta64: {}".format(self.gcta64)


        ### Types of markers to calculate PP
        str_l_type = \
            f"- Types of markers to calculate PP: {self.l_type}"
        
        l_RETURN = [str_raw_ss, str_ref_LD, str_ref_GT, str_curated_input, str_PP,
                   str_plink, str_gcta64, str_l_type]

        return '\n'.join(l_RETURN)



    def set_PLINK_path(self, _plink):
        self.plink = _plink
        return 0

    def set_GCTA_path(self, _gcta):
        self.gcta64 = _gcta
        return 0