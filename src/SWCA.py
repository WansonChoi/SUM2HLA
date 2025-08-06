import os, re
from os.path import basename, dirname, join, exists
import json
import scipy as sp
import pandas as pd

import src.SWCA_SummaryImp as mod_SummaryImp
from src.SWCA_COJO_GCTA import iterate_GCTA_COJO
from src.SWCA_FineMapping import iterate_BayesianFineMapping
import src.SWCA_calc_r_r2 as SWCA_calc_r_r2

import logging

logger = logging.getLogger(__name__)



def transform_imputed_Z_to_ma(_df_imputed_Z, _df_ref_MAF, _N):
    ### Required columns of the ma : "SNP	A1	A2	freq	b	se	p	N"

    ##### (1) The imputed result
    df_imputed_Z_2 = _df_imputed_Z \
                         .loc[:, ['SNP', 'Conditional_mean']] \
        .rename({"Conditional_mean": "Z"}, axis=1)

    # display(df_imputed_Z_2)

    ##### (2) MAF
    df_imputed_Z_3 = df_imputed_Z_2.merge(_df_ref_MAF.drop(['CHR'], axis=1), on=['SNP'])
    # display(df_imputed_Z_3)

    ##### (3) SE, P, and N
    sr_SE = pd.Series([1.0] * df_imputed_Z_2.shape[0], name='SE', index=df_imputed_Z_3.index)
    sr_N = pd.Series([_N] * df_imputed_Z_2.shape[0], name='N', index=df_imputed_Z_3.index)
    sr_P = df_imputed_Z_3['Z'].abs().map(lambda x: 2 * sp.stats.norm.cdf(-x)).rename("P")

    df_RETURN = pd.concat(
        [
            df_imputed_Z_3,
            sr_SE, sr_N, sr_P
        ],
        axis=1
    ) \
                    .rename({"Z": 'b', 'MAF': 'freq', 'SE': 'se', 'P': 'p'}, axis=1) \
                    .loc[:, ['SNP', 'A1', 'A2', 'freq', 'b', 'se', 'p', 'N']]

    return df_RETURN



def find_input_items(_out_prefix):

    """
    - The `from_input_prefix` factory function의 핵심 util함수.
    - 나중에 더 추가해도 됨.

    - item을 찾아오는 방식이 이게 best인지 모르겠음. 근데 당장은 시간없어서 일부러라도 이렇게 짬.
    """

    out_dir = dirname(_out_prefix)
    l_items = [_ for _ in os.listdir(out_dir) if _.startswith(basename(_out_prefix))] # 걍 reduandant한 채로 해.

    # print(l_items)


    _fpath_sumstats3 = None
    _fpath_PP = None

    ### (1) sumstats3
    l_temp = [_ for _ in l_items if _.endswith("sumstats3")]
    if len(l_temp) == 0:
        print("No sumstats3")
    else:
        _fpath_sumstats3 = join(out_dir, l_temp[0])

    l_temp = [_ for _ in l_items if _.endswith(".whole.PP")]
    if len(l_temp) == 0:
        print("No whole PP")
    else:
        _fpath_PP = join(out_dir, l_temp[0])

    

    return _fpath_sumstats3, _fpath_PP



class SWCA:

    def __init__(self, 
                 _df_sumstats3: pd.DataFrame, 
                 _df_ref_ld: pd.DataFrame, _df_ref_MAF: pd.DataFrame, _fpath_ref_bfile: str, 
                 _fpath_PP: str, 
                 _out_prefix:str, _N: int,
                 _module_CMVN="Bayesian", 
                 _f_include_SNPs=False, _f_use_finemapping=True,
                 _r2_pred=0.6, _ncp=5.2, _N_max_iter=5, 
                 _gcta=None, _plink="/home/wschoi/miniconda3/bin/plink"):
        
        # logger.info("SWCARunner initialized with pre-loaded DataFrames.")        
        
        ##### INPUT
        self.df_sumstats3 = _df_sumstats3
        self.df_ref_ld = _df_ref_ld
        self.df_ref_MAF = _df_ref_MAF
        self.fpath_ref_bfile = _fpath_ref_bfile
        self.fpath_PP = _fpath_PP

        self.out_prefix = _out_prefix
        self.N = _N
        self.module_CMVN = _module_CMVN
        self.f_include_SNPs = _f_include_SNPs
        self.f_use_finemapping = _f_use_finemapping
        self.r2_pred = _r2_pred
        self.ncp = _ncp # variance of the ncp
        self.N_max_iter = _N_max_iter


        ##### OUTPUT
        self.out_dir = dirname(self.out_prefix) # out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.df_Z_imputed = None
        self.df_Z_imputed_r2pred = None

        self.fpath_Z_imputed = None
        self.fpath_Z_imputed_r2pred = None


        self.df_ma = None
        self.df_ma_r2pred = None

        self.fpath_ma = None
        self.fpath_ma_r2pred = None

        self.out_prefix_SWCA = join(self.out_dir, basename(self.out_prefix) + ".ROUND_", basename(self.out_prefix))
        # os.makedirs(dirname(self.out_prefix_SWCA), exist_ok=True)
        self.out_SWCA_dict = None

        self.l_conditions = [] # 얘는 잠정적으로 deprecate할거임. (2025.08.05.)
        self.d_conditions = {}
        self.d_conditions_clumped = {}
        self.d_conditions_clumped_r2 = {}


        ##### External software
        self.gcta = _gcta
        self.plink = _plink



    ########## Factory functions

    @classmethod
    def from_paths(cls, _fpath_sumstats3, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _fpath_PP, _out_prefix, _N,
                   **kwargs):

        # logger.info("Loading data from file paths to create SWCARunner instance.")

        df_sumstats3 = pd.read_csv(_fpath_sumstats3, sep='\t', header=0) if isinstance(_fpath_sumstats3, str) else _fpath_sumstats3

        df_ref_ld = pd.read_csv(_fpath_ref_ld, sep='\t', header=0) if isinstance(_fpath_ref_ld, str) else _fpath_ref_ld
        df_ref_ld.index = df_ref_ld.columns

        df_ref_MAF = pd.read_csv(_fpath_ref_MAF, sep=r'\s+', header=0).drop(['NCHROBS'], axis=1) if isinstance(_fpath_ref_MAF, str) else _fpath_ref_MAF

        return cls(
            df_sumstats3, df_ref_ld, df_ref_MAF, _fpath_ref_bfile, _fpath_PP, _out_prefix, _N, **kwargs
        )


    @classmethod
    def from_input_prefix(cls, _input_prefix, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _out_prefix, _N, **kwargs):

        fpath_SUMSTATS3, fpath_PP = find_input_items(_input_prefix)

        return cls.from_paths(
            fpath_SUMSTATS3, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, fpath_PP, _out_prefix, _N, **kwargs
        )



    @classmethod
    def from_config(cls, ): 

        pass



    ########## Magic functions

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
    


    ########## Helper functions

    def perform_Z_imputation(self):

        ##### (1) Summary Imputation

        self.df_Z_imputed = mod_SummaryImp.__MAIN__(self.df_sumstats3, self.df_ref_ld, self.df_ref_MAF)
        self.df_Z_imputed_r2pred = self.df_Z_imputed[ self.df_Z_imputed['r2_pred'] >= self.r2_pred ]

        self.fpath_Z_imputed = self.out_prefix + ".Z_imputed"
        self.df_Z_imputed.to_csv(self.fpath_Z_imputed, sep='\t', header=True, index=False, na_rep="NA")

        return 0


    def prepare_COJO_input(self):

        ##### (2) make the COJO input

        ### 여기서 sumstats3으로 SE recalc하고, BETA 재계산하는 단계가 추가되어야 함.

        self.df_ma = transform_imputed_Z_to_ma(self.df_Z_imputed, self.df_ref_MAF, self.N)
        self.df_ma_r2pred = transform_imputed_Z_to_ma(self.df_Z_imputed_r2pred, self.df_ref_MAF, self.N)

        self.fpath_ma = self.out_prefix + ".ma"
        self.fpath_ma_r2pred = self.out_prefix + f".r2pred{self.r2_pred}.ma"

        self.df_ma.to_csv(self.fpath_ma, sep='\t', header=True, index=False, na_rep="NA")
        self.df_ma_r2pred.to_csv(self.fpath_ma_r2pred, sep='\t', header=True, index=False, na_rep="NA")

        return 0
    

    # def load_generated_ma(self, _fpath_ma, _fpath_ma_r2pred):

    #     """
    #     - 'ma', 'ma_r2pred'가 필요함 perform_SWCA_Bayesian()를 낱개로 돌리려면.
    #     - 이 둘은 특별함. '_out_prefix', '_l_top_signals' 이런 애들이랑 결이 다름.
    #         - 후자 둘은 확실히 INPUT임. 
    #         - 반면에, 'ma', 'ma_r2pred'는 기본적으로 OUTPUT임. 근데 다시 INPUT으로 써야 하는 상황.

    #     - 가급적 Initializer의 argument set을 건드리고 싶지 않았음.
    #         - 딱 Z_imputed와 ma, ma_r2pred가 None으로 시작하고, main orchestrator돌려서 mid-output으로 주어지고, 
    #             최종적으로 SWCA결과 나오는 pipeline형태로 두고 싶었음.

    #     - 그래서 마지막으로 generated한 ma, ma_r2pred를 manually load할 수 있도록 helper funciton을 도입함.
    #     - (분명히 input prefix이용해서 한방에 준비 끝내는 방법도 있을거임.)
        
    #     """

    #     self.fpath_ma = _fpath_ma
    #     self.fpath_ma_r2pred = _fpath_ma_r2pred

    #     self.df_ma = pd.read_csv(self.fpath_ma, sep='\t', header=0)
    #     self.df_ma_r2pred = pd.read_csv(self.fpath_ma_r2pred, sep='\t', header=0)

    #     return 0

    #### 따로 돌릴라 했는데, 벌써 state가 복잡해짐. 그냥 통으로 다 돌려.
    #### 당장 이전의 ma, ma_r2pred의 out_prefix를 써서 input으로 받은 out_prefix랑 따로 놀아서 눈에 안보임.


    def perform_SWCA_Bayesian(self, _out_prefix=None, _l_top_signals=None):

        ##### `l_top_signals`

        l_top_signals = []

        if isinstance(_l_top_signals, list):
            l_top_signals = _l_top_signals
        else:

            df_PP = pd.read_csv(self.fpath_PP, sep='\t', header=0) # 예전에 `.sort_values("PP", ascending=False)` 이렇게했는데 이제 안함.

            df_PP = df_PP[ df_PP['SNP'].isin(self.df_ma_r2pred['SNP']) ] # (2025.05.28.) r2pred로 filter하고 남은 애들만.

            # df_PP = df_PP[ df_PP['CredibleSet'] ] # 예전에는 credible set 단위로 넣었음.
            df_PP = df_PP.iloc[[0], :] # 지금은 정말 top 딱 하나만.

            print(df_PP)

            l_top_signals = df_PP['SNP'].tolist()

        
        print(f"=====[ ROUND 1 ]: {l_top_signals}")


        ###### `out_prefix_SWCA`

        if isinstance(_out_prefix, str):
            out_prefix_SWCA = _out_prefix
        else:
            out_prefix_SWCA = self.out_prefix_SWCA


        ## input ma파일로 이제 r2_pred로 thresholding한게 들어감.
        self.l_conditions, self.d_conditions = iterate_BayesianFineMapping(
            self.df_ma_r2pred, l_top_signals,
            self.fpath_ref_bfile, self.df_ref_ld,
            out_prefix_SWCA,
            _ncp=self.ncp,
            _N_max_iter=self.N_max_iter, _f_polymoprhic_marker=False,
            _plink=self.plink
        )
            # `_f_polymoprhic_marker`얘는 당분간 쓰지 말것.
        
        return self.l_conditions, self.d_conditions
    

    def perform_SWCA_GCTA(self, _out_dir=None, _l_top_signals=None):

        ##### (3) iterate GCTA COJO "--cojo-cond"

        df_PP = pd.read_csv(self.fpath_PP, sep='\t', header=0)
            # 어차피 GCTA도 `df_PP`가 필요함. ROUND_1 top signal은 PP로 찾았을 테니까.

        df_PP = df_PP[ df_PP['SNP'].isin(self.df_ma_r2pred['SNP']) ] # (2025.05.28.) r2pred로 filter하고 남은 애들만.

        # df_PP = df_PP[ df_PP['CredibleSet'] ]
        df_PP = df_PP.iloc[[0], :]

        print(df_PP)
        print(f"=====[ ROUND 1 ]: {df_PP['SNP'].tolist()}")


        if isinstance(_out_dir, str):
            out_prefix_SWCA = join(_out_dir, basename(self.out_prefix_SWCA)) # output directory만 다르게 주고 싶은 경우.
        else:
            out_prefix_SWCA = self.out_prefix_SWCA


        self.l_conditions, self.d_conditions = iterate_GCTA_COJO(
            self.fpath_ma_r2pred, df_PP['SNP'].tolist(),
            self.fpath_ref_bfile, self.df_ref_ld, None,
            out_prefix_SWCA,
            _f_use_finemapping=self.f_use_finemapping, _f_single_factor_markers=False,
            _gcta=self.gcta, _plink=self.plink
        )
            # MAF는 GCTA_COJO에서 single_factor_marker strategy쓸때만 필요함. 그래서 당분간은 안 넣는걸로.

        
        return self.l_conditions, self.d_conditions
    

    def add_r_r2_to_SWCA_output_dictionary(self):

        """
        일부러 output dictionary를 transform하는 형태로 구현했음.
        """

        self.d_conditions_clumped, self.d_conditions_clumped_r2 = SWCA_calc_r_r2.__MAIN__(
            self.d_conditions, dirname(self.out_prefix_SWCA), self.fpath_ref_bfile,
            _plink = self.plink
        )
        
        return 0


    def export_output(self, _out_SWCA_dict=None):

        if isinstance(_out_SWCA_dict, str):
            self.out_SWCA_dict = _out_SWCA_dict
        else:
            self.out_SWCA_dict = join(self.out_dir, basename(self.fpath_ma_r2pred) + ".SWCA.dict") 
                # (ex) No_sim.98.r2pred0.6.ma.SWCA.dict

        ## output dictionary만 export하면 됨.
        with open(self.out_SWCA_dict, 'w') as f_SWCA_dict:
            print(self.out_SWCA_dict)
            json.dump(self.d_conditions_clumped_r2, f_SWCA_dict, indent=4)

        return 0



    ########## Main orchestrator

    def run(self): # MAIN funciton / orchestrator
        
        # logger.info("Running SWCA logic...")
        
        self.perform_Z_imputation()
        self.prepare_COJO_input()

        if self.module_CMVN == "Bayesian": # 여기에 오타있었음 ("Bayesain"). GCTA로 돌아갔음. (2025.08.06.)
            self.perform_SWCA_Bayesian()
        else:
            self.perform_SWCA_GCTA()

        self.add_r_r2_to_SWCA_output_dictionary()

        ## export
        self.export_output()

        return 0


    def __MAIN__(self): # deprecated

        """
        - 근데, 아무리 봐도 일단 갈겨 쓰면서 만들고, 그 다음에 refactoring하는게 맞는것 같다.
            - 차라리 더 functional하게 step-by-step으로 짜. 독립적인 component부분들만 잘 인지하고 짜도 ok임.
        - 저런거를 생각하면서 짜면, 금방 만들것도 엄청 오래 걸릴듯 (제대로 짤 것도 틀리게 짜고).
        - 지금 이렇게 우선 만들고 refactoring하는게 맞는 순서라는 거지.


        - 한편, 지금 SWCA를 수행하기 위한 가장 outer-level (private methods들) vs. 이를 제외한 함수들 이렇게 나누는 상황임.
            - 전자만큼은 class 내에 있는게 맞음.
            - 분류야 분류!
        """



        ##### (2) make the COJO input

        self.df_ma = transform_imputed_Z_to_ma(self.df_Z_imputed, self.df_ref_MAF, self.N)
        self.df_ma_r2pred = transform_imputed_Z_to_ma(self.df_Z_imputed_r2pred, self.df_ref_MAF, self.N)

        OUT_ma = self.out_prefix + ".ma"
        OUT_ma_r2_pred = self.out_prefix + f".r2pred{self.r2_pred}.ma"

        self.df_ma.to_csv(OUT_ma, sep='\t', header=True, index=False, na_rep="NA")
        self.df_ma_r2pred.to_csv(OUT_ma_r2_pred, sep='\t', header=True, index=False, na_rep="NA")



        return 0



if __name__ == '__main__':


    pass