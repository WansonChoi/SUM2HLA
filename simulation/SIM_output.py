import os, sys, re
import json
import numpy as np
import pandas as pd

from datetime import datetime

import src.Util as Util


class SIM_output_scenario_1:

    def __init__(self, _df_SSFN:pd.DataFrame, _col_output:str, _l_answer_primary: list,
                 _N=None, _suffix = ".whole.PP", _suffix_ma = ".ma", _suffix_ma_r2pred = ".r2pred0.6.ma", _label_failed = "Failed"):

        self.df_SSFN = _df_SSFN.copy()
        self.N = _N
        if isinstance(_N, int): self.df_SSFN = self.df_SSFN.iloc[:_N, :]

        self.l_answer_primary = _l_answer_primary

        self.col_output = _col_output


        ## SUM2HLA
        self.sr_out_PP = self.df_SSFN[_col_output].map(lambda x: x + _suffix).rename("OUT")

        self.sr_identified_markers = None
        self.sr_masked_answer = None
        self.sr_masked_answer_2 = None # without failed cases


        ## ma
        self.sr_out_ma = self.df_SSFN[_col_output].map(lambda x: x + _suffix_ma).rename("OUT_ma")

        self.sr_identified_markers_ma = None
        self.sr_masked_answer_ma = None
        self.sr_masked_answer_2_ma = None # without failed cases


        ## ma_r2pred
        self.sr_out_ma_r2pred = self.df_SSFN[_col_output].map(lambda x: x + _suffix_ma_r2pred).rename("OUT_ma_r2pred")

        self.sr_identified_markers_ma_r2pred = None
        self.sr_masked_answer_ma_r2pred = None
        self.sr_masked_answer_2_ma_r2pred = None # without failed cases


        self.label_failed = _label_failed





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





    def check_all_exsit(self): return self.sr_out_PP.map(lambda x: os.path.exists(x)).rename("exist_all")




    def mask_answer(self):
        
        """
        - scenario 2와 consistent하게 하려면 SWCA dictionary로 해야하는데, 당장 SWCA가 왜인지 실패하는 경우가 있음.
        - 이거 때문에 scenario 1은 걍 ".whole.PP"만 활용하는걸로.
        - 마찬가지로, r2 >= 0.9이상으로 같은거로 정답처리할거는 미리 정해서 '_l_answer_primary'로 준다 가정함.
        
        """

        ##### (1) transform to the identifeid variant lables
        def check_identifed(_fpath_output):

            if not os.path.exists(_fpath_output):
                return self.label_failed # 아예 fail한 경우.

            df_temp = pd.read_csv(_fpath_output, sep='\t', header=0, nrows=10).sort_values("PP", ascending=False)

            return df_temp['SNP'].iat[0]
        

        self.sr_identified_markers = self.sr_out_PP.map(lambda x: check_identifed(x)).rename("identified")


        ##### (2) mask answer
        def check_answer(_x):

            if _x == self.label_failed: return _x

            return _x in self.l_answer_primary

        self.sr_masked_answer = self.sr_identified_markers.map(lambda x: check_answer(x)).rename("answer_masked")

        ##### (3) exclude failed cases
        f_failed = self.sr_masked_answer == self.label_failed
        self.sr_masked_answer_2 = self.sr_masked_answer[~f_failed]

        recall_rate = self.sr_masked_answer_2.value_counts()[True] / self.sr_masked_answer_2.shape[0]


        return self.sr_identified_markers, self.sr_masked_answer, self.sr_masked_answer_2, recall_rate
    


    def mask_answer_Z_imputed(self):

        ##### (1) transform to the identifeid variant lables
        def check_identifed(_fpath_output):

            if not os.path.exists(_fpath_output):
                return self.label_failed # 아예 fail한 경우.

            df_temp = pd.read_csv(_fpath_output, sep='\t', header=0)

            ## sorting 1 - |Z|

            # idx_sort = df_temp['b'].abs().sort_values(ascending=False).index
            # df_temp = df_temp.loc[idx_sort, :]


            ## sorting 2 - ['p', Z_abs]

            # df_temp['Z_abs'] = df_temp['b'].abs().rename("Z_abs")
            # df_temp = df_temp.sort_values(['p', 'Z_abs'], ascending=[True, False])

            ## sorting 3 - ['p']
            df_temp = df_temp.sort_values('p')


            return df_temp['SNP'].iat[0]
        

        self.sr_identified_markers_ma = self.sr_out_ma.map(lambda x: check_identifed(x)).rename("identified")


        ##### (2) mask answer // 얜 고대로 쓰면 되네
        def check_answer(_x):

            if _x == self.label_failed: return _x

            return _x in self.l_answer_primary

        self.sr_masked_answer_ma = self.sr_identified_markers_ma.map(lambda x: check_answer(x)).rename("answer_masked")


        ##### (3) exclude failed cases
        f_failed = self.sr_masked_answer_ma == self.label_failed
        self.sr_masked_answer_2_ma = self.sr_masked_answer_ma[~f_failed]


        recall_rate = self.sr_masked_answer_2_ma.value_counts()[True] / self.sr_masked_answer_2_ma.shape[0]


        return self.sr_identified_markers_ma, self.sr_masked_answer_ma, self.sr_masked_answer_2_ma, recall_rate



    def mask_answer_Z_imputed_r2pred(self):

        ##### (1) transform to the identifeid variant lables
        def check_identifed(_fpath_output):

            if not os.path.exists(_fpath_output):
                return self.label_failed # 아예 fail한 경우.

            df_temp = pd.read_csv(_fpath_output, sep='\t', header=0)

            ## sorting 1 - |Z|

            # idx_sort = df_temp['b'].abs().sort_values(ascending=False).index
            # df_temp = df_temp.loc[idx_sort, :]


            ## sorting 2 - ['p', Z_abs]

            # df_temp['Z_abs'] = df_temp['b'].abs().rename("Z_abs")
            # df_temp = df_temp.sort_values(['p', 'Z_abs'], ascending=[True, False])


            ## sorting 3 - ['p']
            df_temp = df_temp.sort_values('p')


            return df_temp['SNP'].iat[0]
        
        

        self.sr_identified_markers_ma_r2pred = self.sr_out_ma_r2pred.map(lambda x: check_identifed(x)).rename("identified")


        ##### (2) mask answer // 얜 고대로 쓰면 되네
        def check_answer(_x):

            if _x == self.label_failed: return _x

            return _x in self.l_answer_primary

        self.sr_masked_answer_ma_r2pred = self.sr_identified_markers_ma_r2pred.map(lambda x: check_answer(x)).rename("answer_masked")


        ##### (3) exclude failed cases
        f_failed = self.sr_masked_answer_ma_r2pred == self.label_failed
        self.sr_masked_answer_2_ma_r2pred = self.sr_masked_answer_ma_r2pred[~f_failed]


        recall_rate = self.sr_masked_answer_2_ma_r2pred.value_counts()[True] / self.sr_masked_answer_2_ma_r2pred.shape[0]


        return self.sr_identified_markers_ma_r2pred, self.sr_masked_answer_ma_r2pred, self.sr_masked_answer_2_ma_r2pred, recall_rate




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


        self.df_identified_markers = None
        self.df_masked_answer = None


    @classmethod # Factory method 1
    def from_paths(cls, _fpath_SSFN:str, _col_output:str, _l_answer_primary: list, _l_answer_secondary: list, **kwargs):

        df_SSFN = pd.read_csv(_fpath_SSFN, sep='\t', header=0)

        return cls(df_SSFN, _col_output, _l_answer_primary, _l_answer_secondary, **kwargs)
    

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

    ### 데이터 프레임으로 무슨 marker가 정답으로 나왔는지까지 확인할 수 있게 짠거

    def transform_Dictionaries_to_Identified_Marker(self, _answer_secondary, _l_r2_threshold = (1.0, 0.95, 0.9)):

        def load_dict(_fpath_dict):
            with open(_fpath_dict, 'r') as f_dict:
                return json.load(f_dict)


        def transform_dict_to_R1_R2_markers(_d_OUTPUT):

            # 일단 간단하게 정말 SUM2HLA가 찍어온 values들만으로 DataFrame만들어봐.

            marker_ROUND_1 = _d_OUTPUT['ROUND_1'][0] if "ROUND_1" in _d_OUTPUT else np.nan

            marker_ROUND_2 = list(_d_OUTPUT['ROUND_2'].keys())[0] if "ROUND_2" in _d_OUTPUT else np.nan

            return marker_ROUND_1, marker_ROUND_2

        # transform_dict_to_R1_R2_markers(d_temp)


        def transform_dict_to_R2_markers_with_r2(_d_OUTPUT, _answer_secondary, _r2_threshold=0.9):

            # display(_d_OUTPUT)

            if "ROUND_2" not in _d_OUTPUT:
                return np.nan

            marker_ROUND2 = list(_d_OUTPUT["ROUND_2"].keys())[0] # single top이라 가정.

            if marker_ROUND2 == _answer_secondary:
                return marker_ROUND2

            d_temp = _d_OUTPUT["ROUND_2"][marker_ROUND2]

            for _marker_clumped, _d_r_r2 in d_temp.items():

                if _marker_clumped == _answer_secondary and _d_r_r2['r2'] >= _r2_threshold:

                    return _marker_clumped

            return marker_ROUND2 # 결국 못 찾으면 원래대로 return

        # transform_dict_to_R2_markers_with_r2(d_temp, "AA_B_9_31432689_D")


        ### main

        sr_dict = self.sr_out_dicts.map(lambda x: load_dict(x))

        sr_R1_R2_markers = sr_dict.map(lambda x: transform_dict_to_R1_R2_markers(x))
        df_R1_R2_markers = pd.DataFrame(sr_R1_R2_markers.tolist(), columns=["ROUND_1", "ROUND_2"])

        sr_R2_r2_0_99 = sr_dict.map(lambda x: transform_dict_to_R2_markers_with_r2(x, _answer_secondary, 0.99)).rename("ROUND_2_r2_0_99")
        sr_R2_r2_0_90 = sr_dict.map(lambda x: transform_dict_to_R2_markers_with_r2(x, _answer_secondary, 0.90)).rename("ROUND_2_r2_0_90")

        self.df_identified_markers = df_RETURN = pd.concat(
            [df_R1_R2_markers, sr_R2_r2_0_99, sr_R2_r2_0_90],
            axis=1
        )

        return df_RETURN
    

    def mask_answer(self, _l_answer_primary=None, _l_answer_secondary=None):

        if not isinstance(self.df_identified_markers, pd.DataFrame):
            print("Run `transform_Dictionaries_to_Identified_Marker()` first!")
            return -1
        
        l_answer_primary = _l_answer_primary if isinstance(_l_answer_primary, list) else self.l_answer_primary
        l_answer_secondary = _l_answer_secondary if isinstance(_l_answer_secondary, list) else self.l_answer_secondary

        ### (1) primary
        df_primary = self.df_identified_markers.iloc[:, [0]].map(lambda x: x in l_answer_primary)


        ### (2) secondary
        df_secondary = self.df_identified_markers.iloc[:, 1:].map(lambda x: x in l_answer_secondary)

        self.df_masked_answer = df_RETURN = pd.concat([df_primary, df_secondary], axis=1)


        return df_RETURN



    def run_v2(self, _label_r2:str):

        self.transform_Dictionaries_to_Identified_Marker(_answer_secondary=_label_r2)

        df_RETURN = self.mask_answer()

        return df_RETURN




    ### procedural하게 짠 구버전 (추후 deprecate하고 싶음.)
    ## (2025.08.12.) 아래 `check_answers()`와 `check_answers_with_r_r2()`는 deprecate하기로 함. 당장 지우지만 않음.

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

        df_RETURN = pd.DataFrame(sr_ToMap.tolist(), columns=[f"correct_secondary_r2_{str(_thresh).replace('.', '_')}" for _thresh in _l_r2_threshold])

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

##### Util 함수

"""
- 필연적으로 multiple `ncp_2nd` values들에 대해 `SIM_output` instances들을 만들거란 말이지.
- 결국 each instance가 run()함수로 return하는 DataFrames들을 concat해야함.
- 이걸위해 새로운 class를 도입하자니, 당장 함수 하나로 해결 가능할 것 같음.
    - 어차피 a SSFN file => `SIM_output` instance => DataFrame by the run() 의 sequence of transformation이라서.
- 일단 당분간은 이렇게 쓰고, multiple `ncp_2nd` values들에 대해 또 다른 작업을 해야하면 그때 class로 짜자.
    - class하나 도입하는 것도 은근히 손이 많이 가네.
"""

def concat_SIM_output_scneario_1(_d_SSFN: dict, _col_output:str="fpath_OUT", 
                      _l_answer_primary:list = ["AA_DRB1_11_32660115_SPG", "AA_DRB1_13_32660109_HF", "SNP_DRB1_32660115_GC"], 
                      _suffix = ".whole.PP"):

    # print(_d_SSFN)

    l_sr_answer_masked = []

    for i, (_ncp_1, _fpath_SSFN) in enumerate(_d_SSFN.items()):

        print(f"=====[ ncp {_ncp_1} ]")

        df_SSFN_temp = pd.read_csv(_fpath_SSFN, sep='\t', header=0)

        SIM_output_scenario_1_temp = SIM_output_scenario_1(df_SSFN_temp, _col_output, _l_answer_primary)

        f_all_exsit = SIM_output_scenario_1_temp.check_all_exsit()
        print(f_all_exsit)

        print(f"All exist?: {f_all_exsit.all()}")

        if not f_all_exsit.all(): 
            print(f_all_exsit[~f_all_exsit])
            return -1

        sr_temp = SIM_output_scenario_1_temp.mask_answer()

        l_sr_answer_masked.append(sr_temp.rename(f"{_ncp_1:+}"))


    df_RETURN = pd.concat(l_sr_answer_masked, axis=1)


    return df_RETURN



def concat_SIM_output(_d_SSFN: dict, _col_output:str, 
                      _l_answer_primary: list, _l_answer_secondary: list,
                      _N=None, _suffix = ".r2pred0.6.ma.SWCA.dict"):

    # print(_d_SSFN)


    def transform_to_masked_DataFrame(_i, _ncp_2nd, _fpath_SSFN):

        print(f"=====[{_i}]: ncp_2nd {_ncp_2nd:+}")

        SIM_output_temp = SIM_output.from_paths(_fpath_SSFN, _col_output, _l_answer_primary, _l_answer_secondary, _N=_N)

        return SIM_output_temp.run()
    

    df_RETURN = pd.concat(
        {_ncp_2nd: transform_to_masked_DataFrame(i, _ncp_2nd, _fpath_SSFN) for i, (_ncp_2nd, _fpath_SSFN) in enumerate(_d_SSFN.items())},
        names=['ncp_2nd', 'Sim_No']
    )


    return df_RETURN




def concat_SIM_output_v2(_d_SSFN: dict, _col_output:str, 
                      _l_answer_primary: list, _l_answer_secondary: list,
                      _label_r2 = str,
                      _N=None, _suffix = ".r2pred0.6.ma.SWCA.dict"):

    # print(_d_SSFN)


    def transform_to_masked_DataFrame(_i, _ncp_2nd, _fpath_SSFN):

        print(f"=====[{_i}]: ncp_2nd {_ncp_2nd:+}")

        SIM_output_temp = SIM_output.from_paths(_fpath_SSFN, _col_output, _l_answer_primary, _l_answer_secondary, _N=_N)

        return SIM_output_temp.run_v2(_label_r2=_label_r2)
    

    df_RETURN = pd.concat(
        {_ncp_2nd: transform_to_masked_DataFrame(i, _ncp_2nd, _fpath_SSFN) for i, (_ncp_2nd, _fpath_SSFN) in enumerate(_d_SSFN.items())},
        names=['ncp_2nd', 'Sim_No']
    )


    return df_RETURN