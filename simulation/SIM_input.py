import os, sys, re
from os.path import join, dirname, basename, exists
import json
import numpy as np
import scipy as sp
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Union

import jax

from datetime import datetime

"""
- a priamry causal marker 선택했을 때 (ex. "AA_DRB1_11_32660115_SPG"), 이 marke와 특정 r/r2값을 가지는 marker 찾는건 보류.
    - r/r2값을 우리가 assign할 수가 없어서, 우리가 원하는 r/r2값을 가지는 marker를 찾아야 함. (그리고 나서 이 marker에 true ncp값들을 가정해서 simulation해야함.)

- (0) 다음들이 input 주어짐: 
    - df_LD
    - a list of causal markers (1개 or 2개 이상; 몇개가 주어지든 상관 없음. the matrix operation이 composition형태로 additive하게 만들어놓을거임.)
    - ncp (expected true association z-score of each of the causal markers)
- (1) expected mean 구하기.
    - 여기서 matrix operation으로 composition
    - export
- (2) the expected mean으로 MVN geneartion (with JAX)
    - float64여야 함. (32면 fail함.)
    - 5,225 SNPs들의 z-scores들만 extract한 후, input simulated summary로 export하기.

"""



class SIM_input:

    """
    - secondary causal marker까지만 가정할거임.
    - 그 이상으로 가면 진짜 `@dataclass`이런거 도입해야 할듯.
    
    """


    def __init__(self, _out_prefix, _df_ref_LD, _df_ref_bim_SNP, _primary_marker, _primary_ncp, _secondary_marker = None, _seconary_ncp = None,
                 _N_sim=1_000, _N_sample=50_000, _seed=0):

        ### input 
        self.out_prefix = _out_prefix

        self.df_ref_LD = _df_ref_LD
        self.df_ref_bim_SNP = _df_ref_bim_SNP # 얘가 잠정적으로 factory function이 많이 필요할 수 있음. 일단 header = ['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
        
        self.primary_marker = _primary_marker
        self.primary_ncp = _primary_ncp

        self.secondary_marker = _secondary_marker
        self.secondary_ncp = _seconary_ncp

        self.N_sim = _N_sim
        self.N_sample = _N_sample
        self.seed = _seed

        ### output / states

        self.out_mean = self.out_prefix + ".mean"
        self.out_SSFN = self.out_prefix + ".SSFN"

        self.out_dir_OUT = self.out_prefix + ".OUT"
        os.makedirs(self.out_dir_OUT, exist_ok=True)

        self.out_dir_SUMSTATS = self.out_prefix + ".SUMSTATS"
        os.makedirs(self.out_dir_SUMSTATS, exist_ok=True)

        self.sr_mean_expected = None
        self.df_SSFN = None



    # @classmethod # Factory method 1
    # def from_paths(cls, _fpath_SSFN:str, _col_output:str, _l_answer_primary: list, _l_answer_secondary: list, **kwargs):

    #     df_SSFN = pd.read_csv(_fpath_SSFN, sep='\t', header=0)

    #     return cls(df_SSFN, _col_output, _l_answer_primary, _l_answer_secondary, **kwargs)
    

    def __repr__(self):
        
        # 클래스 이름을 가져옵니다.
        class_name = self.__class__.__name__
        
        # self.__dict__ 의 모든 아이템을 "key = value" 형태의 문자열로 만듭니다.
        # 데이터프레임처럼 내용이 긴 속성은 shape만 출력하도록 간단히 처리할 수 있습니다.
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                items.append(f"{key}=DataFrame(shape={value.shape})")
            elif isinstance(value, pd.Series):
                items.append(f"{key}=Series(shape={value.shape})")
            else:
                # repr(value)를 사용해 문자열에 따옴표가 붙도록 합니다.
                items.append(f"{key}={repr(value)}")
        
        # 모든 속성을 쉼표와 줄바꿈으로 연결하여 보기 좋게 만듭니다.
        return f"{class_name}(\n  " + ",\n  ".join(items) + "\n)"
    


    ########## Main helper functions ##########

    def calc_ncp_expected_mean(self):

        sr_header = self.df_ref_LD.columns.to_series()

        sr_configuration = sr_header.map(
            lambda x: self.primary_ncp if x == self.primary_marker else self.secondary_ncp if x == self.secondary_marker else 0.0
        )

        ### for checking
        # print(sr_configuration)
        # print(sr_configuration.sort_values(ascending=False))
        
        
        self.sr_mean_expected = pd.Series(
            self.df_ref_LD.values @ sr_configuration,
            index = self.df_ref_LD.index, name='mean_exp'
        )

        self.sr_mean_expected.to_csv(
            self.out_mean, sep='\t', header=False, index=True, na_rep="NA"
        )
            
        return self.sr_mean_expected



    def make_ssfn(self):

        self.df_SSFN = pd.concat(
            [
                pd.Series([join(self.out_dir_SUMSTATS, f"Sim_No.{i+1}.sumstats") for i in np.arange(1000)], name='fpath_IN'),
                pd.Series([join(self.out_dir_OUT, f"Sim_No.{i+1}") for i in np.arange(1000)], name='fpath_OUT')
            ],
            axis=1
        )

        self.df_SSFN.to_csv(self.out_SSFN, sep='\t', header=True, index=False, na_rep="NA")

        return self.df_SSFN
    


    def generate_simulated_GWAS_summaries(self, _M_break=None):


        jax.config.update("jax_enable_x64", True)
        print(f"기본 jax_enable_x64 설정: {jax.config.jax_enable_x64}")        

        # @jax.jit
        def sampling_with_MVN(_key, _mean_vec, _cov_mat, M=self.N_sim):
            """
            JAX 키를 인자로 받아 다변수 정규분포 샘플을 생성.
            """
            return jax.random.multivariate_normal(_key, _mean_vec, _cov_mat, shape=(M,))
        

        sampling_with_MVN_JAX = jax.jit(sampling_with_MVN, static_argnums=(3,))
        

        arr_ss_generated = sampling_with_MVN_JAX(jax.random.PRNGKey(self.seed), self.sr_mean_expected.values, self.df_ref_LD.values, self.N_sim)
        

        
        for j, (_fpath_SUMSTATS, _arr_ss) in enumerate(zip(self.df_SSFN['fpath_IN'], arr_ss_generated)):
            
            # print("===[{}]: ({})".format(j, datetime.now()))
            
            if j % 100 == 0:
                print("===[{}]: ({})".format(j, datetime.now()))
                # display(_arr_ss, _out)
            
            df_temp = pd.DataFrame({
                "SNP": self.df_ref_LD.columns, 
                "Z": _arr_ss,
                "P": sp.stats.norm.cdf(-np.abs(_arr_ss)) * 2,
                "N": [self.N_sample] * self.df_ref_LD.shape[0],
                "SE": [1.0] * self.df_ref_LD.shape[0]
            })

            if j == 0:        
                print(df_temp.head(10))
                # display(df_temp.shape)
            
            df_temp = self.df_ref_bim_SNP.merge(df_temp, on='SNP') # (2025.07.15.) 걍 join으로 sorting하면서 SNP만 남기기
            # display(df_temp)
            
            df_temp.to_csv(_fpath_SUMSTATS, sep='\t', header=True, index=False, na_rep='NA')

            if isinstance(_M_break, int) and j >= _M_break: break



        jax.config.update("jax_enable_x64", False)
        print(f"종료 후 jax_enable_x64 설정: {jax.config.jax_enable_x64}")        



        return 0



    def run(self):

        self.calc_ncp_expected_mean()
        self.make_ssfn()

        self.generate_simulated_GWAS_summaries()


        return 0