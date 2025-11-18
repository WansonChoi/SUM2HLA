import os, sys, re
from os.path import basename, dirname, exists, join
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
import src.SUM2HLA_PostCalc_Cov as mod_PostCal_Cov
import src.SWCA as SWCA

import logging
logger_SUM2HLA_batch = logging.getLogger(__name__)


class SUM2HLA_batch(): # a single run (batch) of SUM2HLA.

    def __init__(self, _ss_raw, _ref_prefix, _out_prefix,
                 _batch_size=30, 
                 _f_run_SWCR=True, _N_max_iter=3, _r2_pred=0.6, 
                 _ncp=5.2,
                 _out_json=None, _bfile_ToClump=None, _f_do_clump=True, # Utility arguments for testing.
                 _plink=which("plink"), _gcta=None
    ):

        ##### INPUT

        ### Raw GWAS summary file
        self.sumstats1 = _ss_raw


        ### The reference dataset.
        self.d_fpath_LD = {"whole": _ref_prefix + ".NoNA.PSD.ld" if exists(_ref_prefix + ".NoNA.PSD.ld") else _ref_prefix + ".NoNA.PSD.ld.gz"} # 예전에 HLA sub-region 별로 짤라서 활용하던대로 둠.
        self.fpath_LD_SNP_HLA = _ref_prefix if _bfile_ToClump is None else _bfile_ToClump
        self.fpath_LD_MAF = _ref_prefix + ".FRQ.frq"
        self.out_prefix_LD = _out_prefix + ".LD"


        ### output path and prefix
        self.out_prefix = _out_prefix
        self.out_dir = dirname(_out_prefix)
        if self.out_dir != '':
            os.makedirs(self.out_dir, exist_ok=True)


        ### Clumping
        self.f_do_clump = _f_do_clump


        ### Model Parameters
        self.ncp = _ncp
        # print(f"NCP: {self.ncp}")
        self.gamma = 0.01
        self.N_causal = 1 # 웬만하면 건드리지 마라.
        self.batch_size = _batch_size
        self.engine = "jax"



        ##### OUTPUT / states

        ### 'mod_SUM2HLA_prepr()' => 얘가 return하는 main output 4개.
        self.out_matched = None
        self.out_ToClump = None
        self.out_clumped = None # sumstats2
        self.out_json = _out_json # sumstats3 (***; ".SUM2HLA_input.json")


        ### 'SUM2HLA_PostCalc_Cov.py'
        self.LDmatrix:INPUT_LDmatrix = None
        self.GWAS_summary:INPUT_GWAS_summary = None
        
        self.LL_0 = None # LL-baseline

        self.OUT_PIP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        self.OUT_LL_N_causal = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)}
        
        self.OUT_PIP_PP = {_N_causal: None for _N_causal in range(1, self.N_causal + 1)} # PIP를 posterior로 normalize and export.
        self.OUT_PIP_PP_fpath = {_N_causal: {'whole': None} for _N_causal in range(1, self.N_causal + 1)} # filepath

        ## Types of markers to calculate PP.
        # self.l_type = ('whole', 'SNP', 'HLAtype', 'HLA', 'AA', 'intraSNP', 'AA+HLA')
        self.l_type = ('AA+HLA',)


        ### SWCR
        # self.N = _N   # `GWAS_summary` class 내에서 가져오도록 바꿈.
        # self.ma = None
        # self.l_conditional_signals = [] # 여차하면 없애도 됨.
        # self.d_conditional_signals = {}
        self.f_run_SWCR = _f_run_SWCR
        self.N_max_iter = _N_max_iter
        self.r2_pred = _r2_pred

        self.SWCA_batch:SWCA.SWCA = None


        ### External software
        self.plink = which("plink") if bool(which("plink")) else _plink
        self.gcta64 = which("gcta") if bool(which("gcta")) else _gcta

    

    ########## Main helper funcitons ##########

    def __repr__(self):
        
        # 클래스 이름을 가져옵니다.
        class_name = self.__class__.__name__
        
        # self.__dict__ 의 모든 아이템을 "key = value" 형태의 문자열로 만듭니다.
        # 데이터프레임처럼 내용이 긴 속성은 shape만 출력하도록 간단히 처리할 수 있습니다.
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                items.append(f"{key}=DataFrame(shape={value.shape})")
            else:
                # repr(value)를 사용해 문자열에 따옴표가 붙도록 합니다.
                items.append(f"{key}={repr(value)}")
        
        # 모든 속성을 쉼표와 줄바꿈으로 연결하여 보기 좋게 만듭니다.
        return f"{class_name}(\n  " + ",\n  ".join(items) + "\n)"



    def run_SUM2HLA_prepr(self):

        print("\n\n==========[0]: preprocessing SUM2HLA input")

        ##### Interface with the `mod_SUM2HLA_prepr.SUM2HLA_prepr()`.
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
    


    def calc_posterior_probabilities(self):

        if self.out_json == None:
            logger_SUM2HLA_batch.error("You must run the `run_SUM2HLA_prepr()` member function first!")
            return -1
        


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
                    # (2025.06.28.) No. 혹시나 나중에 N_causal >= 2 할때 여기 있는게 더 나을 듯.
                # 'fine-mapping_SWCA.py'에서는 문제없이 집어넣었음.

        # print("LL_0: ", self.LL_0)

        Lprior_0 = (0 * np.log(self.gamma) + (self.LDmatrix.df_LD.shape[0] - 0) * np.log(1 - self.gamma))
        # print(Lprior_0)
        
        self.LL_0 += Lprior_0
        # print("LL_0: ", self.LL_0)



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


        return self.OUT_PIP_PP_fpath



    def run_SWCA(self, _N_causal=1, _type='whole'): # '_N_causal=1' 일 때만 한다 가정.

        if self.out_json == None:
            logger_SUM2HLA_batch.error("You must run the `run_SUM2HLA_prepr()` member function first!")
            return -1

        if self.OUT_PIP_PP_fpath == None:
            logger_SUM2HLA_batch.error("You must run the `calc_posterior_probabilities()` member function first!")
            return -1

        # if self.GWAS_summary == None: # 얘는 어차피 바로 위 조건문과 redundant한 것 같음.



        print("\n\n==========[3]: Step-Wise Conditional Analysis.")
        
        ### load the '*.SUM2HLA_input.json'
        with open(self.out_json, 'r') as f_json:
            d_curated_input = json.load(f_json)


        # self.SWCA_batch = SWCA.SWCA.from_paths(
        #     _fpath_sumstats3=d_curated_input['whole']['sumstats'], 
        #     _fpath_ref_ld=d_curated_input['whole']['ld'],
        #     _fpath_ref_bfile=self.fpath_LD_SNP_HLA, 
        #     _fpath_ref_MAF=self.fpath_LD_MAF,
        #     _fpath_PP=self.OUT_PIP_PP_fpath[_N_causal]['whole'], 
        #     _out_prefix=self.out_prefix, 
        #     _N=self.GWAS_summary.N,
        #     _r2_pred=self.r2_pred, _ncp=self.ncp, _N_max_iter=self.N_max_iter,
        #     _plink=self.plink)

        ## 이제 codingAA+HLA로 fix시킬거임.
        self.SWCA_batch = SWCA.SWCA.from_paths(
            _fpath_sumstats3=d_curated_input['whole']['sumstats'], 
            _fpath_ref_ld=d_curated_input['whole']['ld'],
            _fpath_ref_bfile=self.fpath_LD_SNP_HLA, 
            _fpath_ref_MAF=self.fpath_LD_MAF,
            _fpath_PP=self.OUT_PIP_PP_fpath[_N_causal]['AA+HLA'], 
            _out_prefix=self.out_prefix, 
            _N=self.GWAS_summary.N,
            _r2_pred=self.r2_pred, _ncp=self.ncp, _N_max_iter=self.N_max_iter,
            _plink=self.plink,
            _f_use_codingAA_HLA=True
        )


        return self.SWCA_batch.run()
    


    def run(self):

        self.run_SUM2HLA_prepr()

        self.calc_posterior_probabilities()

        if self.f_run_SWCR:
            self.run_SWCA()

        return 0

    
    
    # def __MAIN__(self):

    #     ##### has the curated main input?
    #     if self.out_json == None:
    #         print("\n\n==========[0]: preprocessing SUM2HLA input")
    #         self.run_SUM2HLA_prepr()


        
    #     ##### load the curated input (LD and GWAS summary)
    #     with open(self.out_json, 'r') as f_json:
    #         d_curated_input = json.load(f_json)

    #     self.LDmatrix = INPUT_LDmatrix(d_curated_input['whole']['ld'])
    #     self.GWAS_summary = INPUT_GWAS_summary(d_curated_input['whole']['sumstats'], self.LDmatrix)



    #     ##### calculate LL_0
    #     self.LL_0 = \
    #         self.LDmatrix.term2 \
    #         -0.5*( self.GWAS_summary.sr_GWAS_summary.values.T @ (np.linalg.solve(self.LDmatrix.df_LD_SNP.values, self.GWAS_summary.sr_GWAS_summary.values)) )
    #             # (2025.05.14.) 얘 잠정적으로 `mod_PostCal_Cov.__MAIN__()` 함수 안으로 집어넣었으면 좋겠음.
    #                 # (2025.06.28.) No. 혹시나 나중에 N_causal >= 2 할때 여기 있는게 더 나을 듯.
    #             # 'fine-mapping_SWCA.py'에서는 문제없이 집어넣었음.
    #     print("LL_0: ", self.LL_0)

    #     Lprior_0 = (0 * np.log(self.gamma) + (self.LDmatrix.df_LD.shape[0] - 0) * np.log(1 - self.gamma))
    #     print(Lprior_0)
        
    #     self.LL_0 += Lprior_0
    #     print("LL_0: ", self.LL_0)

    #     ##### run
    #     print("\n\n==========[1]: calculating LL (for each batch)")
    #     print("Batch size: {}".format(self.batch_size))
    #     for _N_causal in range(1, self.N_causal + 1):

    #         t_start_calcLL = datetime.now()
            
    #         self.OUT_PIP[_N_causal], self.OUT_LL_N_causal[_N_causal] = \
    #             mod_PostCal_Cov.__MAIN__(
    #                 _N_causal, self.GWAS_summary, self.LDmatrix, self.LL_0,
    #                 _batch_size=self.batch_size, _gamma=self.gamma, _ncp=self.ncp, _engine=self.engine
    #             )
        
    #         t_end_calcLL = datetime.now()

    #         print("\nTotal time for the LL when `N_causal`={}: {}".format(_N_causal, t_end_calcLL - t_start_calcLL))


        
    #     ##### Postprocessing
    #     print("\n\n==========[2]: postprocessing and export")
    #     for _N_causal in range(1, self.N_causal + 1):
            
    #         df_PP = pd.DataFrame({
    #             "SNP": self.LDmatrix.df_LD.columns,
    #             "LL+Lprior": self.OUT_PIP[_N_causal]
    #         })

    #         # df_PP.to_csv(self.out_prefix + ".before_postprepr.txt", sep='\t', header=True, index=False, na_rep="NA")
    
    #         self.OUT_PIP_PP[_N_causal] = \
    #             mod_PostCal_Cov.postprepr_LL(df_PP, _l_type=self.l_type) # 여기서 return되는건 DataFrame의 dictionary임.

    #         for _type, _df_PP in self.OUT_PIP_PP[_N_causal].items():

    #             out_temp = self.out_prefix + ".{}.PP".format(_type)
    #             _df_PP.to_csv(out_temp, sep='\t', header=True, index=False, na_rep="NA")

    #             ## `_df_PP`를 fwrite했으면 그냥 file path를 저장.
    #             self.OUT_PIP_PP_fpath[_N_causal][_type] = out_temp



    #     ##### SWCA
    #     if self.f_run_SWCR:
    #         print("\n\n==========[3]: Step-Wise Conditional Analysis.")
    #         self.run_SWCA() # `self.fpath_secondary_signals` 가 여기서 채워짐.
        
        
        
    #     return self.OUT_PIP, self.OUT_LL_N_causal
