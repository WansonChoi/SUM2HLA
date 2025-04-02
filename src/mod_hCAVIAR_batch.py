import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd
import math
import json
from datetime import datetime
from shutil import which

import mod_hCAVIAR_prepr
# import mod_Util
import mod_LDmatrix_class
import mod_GWAS_summary

from mod_LDmatrix_class import LDmatrix
from mod_GWAS_summary import GWAS_summary
import mod_PostCal_Cov

import mod_SWCA


### use GPU
import jax
from jax.lib import xla_bridge

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # a singleimport jax
# print(jax.devices())
# print(jax.__version__)
# print(xla_bridge.get_backend().platform)



class hCAVIAR_batch(): # a single run of hCAVIAR on Linux.

    def __init__(self, _ss_raw, _out_prefix, _ethnicity, _N, _out_json=None,
                _batch_size=50, _bfile_ToClump=None,
                 _f_run_SWCR=True, _f_do_clump=True,
                ):

        ##### output path and prefix
        self.out_prefix = _out_prefix
        self.out_dir = os.path.dirname(_out_prefix)
        # self.out_basename = os.path.basename(_out_prefix)
        self.out_prefix_LD = None


        ##### Raw LD files (before curation)
        ## 어차피 load해야할 대상이 정해져 있음. (최종 publish할 때는 clumping부분을 어떻게 해줄지 모르겠네.)
        if _ethnicity == 'EUR':

            ### T1DGC
            self.d_fpath_LD = {"whole": "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/LD.REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld"}
            self.fpath_LD_SNP_bim = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.bim"
            self.fpath_LD_SNP_HLA = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA" if _bfile_ToClump is None else _bfile_ToClump
            self.fpath_LD_MAF = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq"

            self.out_prefix_LD = _out_prefix + ".LD.REF_T1DGC.hg19.SNP+HLA.NoNA.PSD"

            ### 1kG - EUR
            # self.d_fpath_LD = {"whole": "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/LD.REF_1kG.EUR.hg19.SNP+HLA.NoNA.PSD.ld"}
            # self.fpath_LD_SNP_bim = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_1kG.EUR.hg19.bim"
            # self.fpath_LD_SNP_HLA = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_1kG.EUR.hg19.SNP+HLA" if _bfile_ToClump is None else _bfile_ToClump
            # self.fpath_LD_MAF = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_1kG.EUR.hg19.SNP+HLA.FRQ.frq"

            # self.out_prefix_LD = _out_prefix + ".LD.1kG_EUR.hg19.SNP+HLA.NoNA.PSD"

        
        elif _ethnicity == 'EAS':

            self.d_fpath_LD = {"whole": "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/LD.REF_HAN_China.hg19.SNP+HLA.NoNA.PSD.ld"}
            self.fpath_LD_SNP_bim = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_HAN_China.hg19.bim"
            self.fpath_LD_SNP_HLA = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_HAN_China.hg19.SNP+HLA" if _bfile_ToClump is None else _bfile_ToClump
            self.fpath_LD_MAF = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_HAN_China.hg19.SNP+HLA.FRQ.frq"

            self.out_prefix_LD = _out_prefix + ".LD.REF_HAN_China.hg19.SNP+HLA.NoNA.PSD"
        else:
            raise ValueError("Wrong `_ethnicity` value! ({})".format(_ethnicity))

        
        ##### Raw GWAS summary file
        self.sumstats1 = _ss_raw


        ##### Intermediate output of the curation. ('mod_hCAVIAR_prepr')
        self.out_matched = None
        self.out_ToClump = None
        self.out_clumped = None # sumstats2
        self.out_json = _out_json # sumstats3 (***; ".hCAVIAR_input.json")


        ##### Curated input (LD and GWAS summary)
        self.LDmatrix:LDmatrix = None
        self.GWAS_summary:GWAS_summary = None

        self.LL_0 = None # LL-baseline

        
        ##### Model Parameters
        self.ncp = 5.2
        self.gamma = 0.01
        self.N_causal = 1 # 웬만하면 건드리지 마라.
        self.batch_size = _batch_size
        self.engine = "jax"


        ##### External software
        self.plink = "/home/wschoi/miniconda3/bin/plink"
        self.gcta64 = "/data02/wschoi/_ClusterPhes_v4/gcta-1.94.3-linux-kernel-3-x86_64/gcta64"

        
        ##### Output (Accumulator)
        self.OUT_PIP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        self.OUT_LL_N_causal = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        
        self.OUT_PIP_PP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)} # PIP를 posterior로 normalize and export.
        self.OUT_PIP_PP_fpath = {_N_causal: {'whole': None} for _N_causal in range(1, self.N_causal + 1)} # filepath

        ##### SWCR with GCTA "--cojo-cond"
        self.N = _N # sample size
        self.ma = None
        self.l_secondary_signals = []
        self.fpath_secondary_signals = None
        self.f_run_SWCR = _f_run_SWCR

        # self.out_cojo_slct = None


        ##### Clumping
        self.f_do_clump = _f_do_clump


    

    
    def run_hCAVIAR_prepr(self):
        
        ##### Interface with the `mod_hCAVIAR_prepr.hCAVIAR_prepr()`.
        ## 얘는 나중에 조건부로 안돌리게 만들어야 할 수도 있을 것 같아서 따로 떼어내놓게.

        self.out_matched, self.out_ToClump, self.out_clumped, self.out_json = \
            mod_hCAVIAR_prepr.__MAIN__(
                _fpath_ss=self.sumstats1,
                _d_fpath_LD=self.d_fpath_LD, _fpath_LD_SNP_bim=self.fpath_LD_SNP_bim, _fpath_LD_SNP_HLA=self.fpath_LD_SNP_HLA,
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


        self.l_secondary_signals, self.ma = mod_SWCA.__MAIN__(
            _fpath_ss=d_curated_input['whole']['sumstats'],
            _fpath_ld_matrix=d_curated_input['whole']['ld'],
            _fpath_ld_bfile=self.fpath_LD_SNP_HLA,
            _fpath_ld_MAF=self.fpath_LD_MAF,
            _fpath_PP=self.OUT_PIP_PP_fpath[_N_causal]['whole'],
            _out_prefix=self.out_prefix,
            _N=self.N,
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

    
    
    def compute(self):

        ##### has the curated main input?
        if self.out_json == None:
            print("\n\n==========[0]: preprocessing hCAVIAR input")
            self.run_hCAVIAR_prepr()


        
        ##### load the curated input (LD and GWAS summary)
        with open(self.out_json, 'r') as f_json:
            d_curated_input = json.load(f_json)

        self.LDmatrix = LDmatrix(d_curated_input['whole']['ld'])
        self.GWAS_summary = GWAS_summary(d_curated_input['whole']['sumstats'], self.LDmatrix)


        
        ##### calculate LL_0
        self.LL_0 = \
            self.LDmatrix.term2 \
            -0.5*( self.GWAS_summary.sr_GWAS_summary.values.T @ (np.linalg.solve(self.LDmatrix.df_LD_SNP.values, self.GWAS_summary.sr_GWAS_summary.values)) )


        
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

            df_PP.to_csv(self.out_prefix + ".before_postprepr.txt", sep='\t', header=True, index=False, na_rep="NA")
    
            self.OUT_PIP_PP[_N_causal] = \
                mod_PostCal_Cov.postprepr_LL(df_PP, _l_type=['whole']) # 여기서 return되는건 DataFrame의 dictionary임.

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
            "- Raw input GWAS summary: {}".format(self.sumstats1)

        str_raw_LD = \
            "- Raw LD file: {}".format(self.d_fpath_LD)

        str_ToClump = \
            "- Clumping genotype: {}".format(self.fpath_LD_SNP_HLA) if self.f_do_clump else \
            "- No Clumping!"

        
        str_curated_input = \
            "- Curated input: {}".format(self.out_json)

        str_PP = \
            "- Output - Posterior probability: {}".format(self.OUT_PIP_PP_fpath)

        ### external software
        str_plink = \
            "- plink: {}".format(self.plink)
        str_gcta64 = \
            "- gcta64: {}".format(self.gcta64)
        
        
        l_RETURN = [str_raw_ss, str_raw_LD, str_ToClump, str_curated_input, str_PP,
                   str_plink, str_gcta64]

        return '\n'.join(l_RETURN)