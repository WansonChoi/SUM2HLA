import os, sys, re
import json
import numpy as np
import pandas as pd

from datetime import datetime

import src.Util as Util


class SIM_output:

    """
    - For a ncp_2nd value, a SSFN file with `_N` simulations is given.
    - multiple SSFN files들이 주지는건 이 class의 composition으로 짤거임.

    - (cf) ROUND_1은 clumping과 r2값을 반영할 수 없음. 얘는 최초에 SNPs들이 clumping된걸로 알아내는거기 때문에.
    - 의도한건 아니었지만, 그래서 ROUND_1은 list, ROUND_2는 dict로 주어짐.
    
    """

    def __init__(self, _df_SSFN:pd.DataFrame, _col_output:str, _l_answer_primary: list, _l_answer_secondary: list,
                 _N=None, _suffix = ".r2pred0.6.ma.SWCA.dict"):

        self.df_SSFN = _df_SSFN
        self.N = _N
        if isinstance(_N, int): self.df_SSFN = self.df_SSFN.iloc[:_N, :]

        self.col_output = _col_output
        self.sr_out_dicts = self.df_SSFN[_col_output].map(lambda x: x + _suffix).rename("OUT_dictionary")

        self.l_answer_primary = _l_answer_primary
        self.l_answer_secondary = _l_answer_secondary


    @classmethod # Factory method 1
    def from_paths(cls, _fpath_SSFN, _col_output, **kwargs):

        df_SSFN = pd.read_csv(_fpath_SSFN, sep='\t', header=0)

        return cls(df_SSFN, _col_output, **kwargs)
    

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


    ##### Helper functions

    def check_answers(self, _l_answer_primary=None, _l_answer_secondary=None):

        """
        - 딱 strict하게 정답 marker가 나타났는지 확인. (in primary and secondary)
        
        """

        ##### [0] Thising to reset or override for a function call.
        _l_answer_primary = _l_answer_primary if isinstance(_l_answer_primary, list) else self.l_answer_primary
        _l_answer_secondary = _l_answer_secondary if isinstance(_l_answer_secondary, list) else self.l_answer_secondary


        ## the function to mask answer given an output dictionary of a simulation
        def check_answer(_fpath_dict, _l_answer_primary: list, _l_answer_secondary: list):
            

            d_SWCA_out = None
            f_correct_primary = None
            f_correct_seconary = None

            with open(_fpath_dict, 'r') as f_json:
                d_SWCA_out = json.load(f_json)


            # print(d_SWCA_out)


            # (cf) `d_SWCA_out['ROUND_1']` and `d_SWCA_out['ROUND_2']` 둘 다 single item이라 가정. (SUM2HLA 돌릴 때 the top만 가져왔다 가정.)
            if "ROUND_1" in d_SWCA_out:
                l_ROUND_1 = d_SWCA_out['ROUND_1'] # 얘는 무조건 list임.
                f_correct_primary = pd.Series(l_ROUND_1).isin(_l_answer_primary).any()
            else:
                f_correct_primary = np.nan


            if "ROUND_2" in d_SWCA_out:
                d_ROUND_2 = d_SWCA_out['ROUND_2'] # 'ROUND_2'는 무조건 dict임.
                l_ROUND_2 = list(d_ROUND_2.keys())
                f_correct_seconary = pd.Series(l_ROUND_2).isin(_l_answer_secondary).any()
            else:
                f_correct_seconary = np.nan



            return f_correct_primary, f_correct_seconary 


        sr_ToMap = self.sr_out_dicts.map(lambda x: check_answer(x, _l_answer_primary, _l_answer_secondary))

        df_RETURN = pd.DataFrame(sr_ToMap.tolist(), columns=['correct_primary', 'correct_secondary'])

        return df_RETURN


    def check_answers_with_r_r2(self, _l_answer_secondary=None, _l_r2_threshold = (0.5, 0.7, 0.9)):

        """
        - ROUND_2 signal을 several r2 threshold values들로 확인.
        
        """

        ##### [0] Thising to reset or override for a function call.
        _l_answer_secondary = _l_answer_secondary if isinstance(_l_answer_secondary, list) else self.l_answer_secondary


        def check_answer(_fpath_dict, _l_answer_secondary: list, _l_r2_threshold: list):

            d_SWCA_out = None

            with open(_fpath_dict, 'r') as f_json:
                d_SWCA_out = json.load(f_json)


            # print(d_SWCA_out)

            if "ROUND_2" in d_SWCA_out:


                def subset_dict_with_r2(_d_ROUND_2, _r2_threshold=0.9):

                    d_RETURN = {
                        _SNP_clumped: _d_r_r2 \
                            for _SNP_clumped, _d_r_r2 in _d_ROUND_2.items() \
                                if 'r' in _d_r_r2 and 'r2' in _d_r_r2 and _d_r_r2['r2'] >= _r2_threshold
                    }

                    return d_RETURN

                l_keys = list(d_SWCA_out['ROUND_2'].keys()) # 당연히 a single item이라 가정.

                d_ROUND_2 = d_SWCA_out['ROUND_2'][l_keys[0]]

                iter_d_subset = (subset_dict_with_r2(d_ROUND_2, _threshold) for _threshold in _l_r2_threshold)
                iter_l_subset = (list(_.keys()) for _ in iter_d_subset)
                iter_mask = (pd.Series(_l_answer_secondary).isin(_).any() for _ in iter_l_subset)

                return tuple(iter_mask)                
            else:
                return (np.nan, ) * 3




        sr_ToMap = self.sr_out_dicts.map(lambda x: check_answer(x, _l_answer_secondary, _l_r2_threshold))

        df_RETURN = pd.DataFrame(sr_ToMap.tolist(), columns=[f"correct_secondary_r2_{str(_thresh).replace(".", "_")}" for _thresh in _l_r2_threshold])

        return df_RETURN


    def run(self):
        
        """
        - Main dish 1
        """

        df_RETURN_1 = self.check_answers()
        df_RETURN_2 = self.check_answers_with_r_r2()

        df_RETURN = pd.concat([df_RETURN_1, df_RETURN_2], axis=1)

        return df_RETURN



"""
- 나중에 임의의 simulation의 plot을 그릴일이 있나?
- 혹은 multiple simulations들의 결과를 취합해서 뭘 그릴일이있나?

그게 아니라면 여가까지면 된듯? 당장 정답 check를 위한 뭐시기는.

- 그 다음 얘를 multple ncp_2nd values들에 대해 compoisiton한 class를 만들어야 함.
    - 여기서 return된 채점 결과를 여러가지 기준으로 요리해먹는 짓을 해야 함.


    
객체들을 inpu으로 받아서 활요하는 함수 만들기 vs. class 내에서 객체들을 만들고 무언가를 만들기.
composiiton하려니까 얘기가 또 살짝 달라지네.
"""




    # def check_answers_v0(self, _suffix = ".r2pred0.6.ma.SWCA.dict", _l_answer_primary=None, _l_answer_secondary=None): # Deprecated

    #     _l_answer_primary = _l_answer_primary if isinstance(_l_answer_primary, list) else self.l_answer_primary
    #     _l_answer_secondary = _l_answer_secondary if isinstance(_l_answer_secondary, list) else self.l_answer_secondary

    #     """
    #     - additional arguments들이 주어지는건 기본적으로 개별적으로 활용해보기 위함임.
        
    #     """


    #     sr_ToMap = self.sr_out_prefix.map(lambda x: x + _suffix)


    #     def check_answer(_fpath_dict, _l_answer_primary: list, _l_answer_secondary: list):
            

    #         d_SWCA_out = None
    #         f_correct_primary = None
    #         f_correct_seconary = None

    #         with open(_fpath_dict, 'r') as f_json:
    #             d_SWCA_out = json.load(f_json)


    #         # print(d_SWCA_out)


    #         if "ROUND_1" in d_SWCA_out:
    #             f_correct_primary = pd.Series(d_SWCA_out['ROUND_1']).isin(_l_answer_primary).any()
    #         else:
    #             f_correct_primary = np.nan


    #         if "ROUND_2" in d_SWCA_out:
    #             f_correct_seconary = pd.Series(d_SWCA_out['ROUND_2']).isin(_l_answer_secondary).any()
    #         else:
    #             f_correct_seconary = np.nan



    #         return f_correct_primary, f_correct_seconary 


    #     sr_ToMap = sr_ToMap.map(lambda x: check_answer(x, _l_answer_primary, _l_answer_secondary))

    #     df_RETURN = pd.DataFrame(sr_ToMap.tolist(), columns=['checked_1st', 'checked_2nd'])

    #     return df_RETURN