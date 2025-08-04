import os, re
from os.path import basename, dirname, join, exists
import scipy as sp
import pandas as pd

import src.SWCA_SummaryImp as mod_SummaryImp
from src.SWCA_COJO_GCTA import iterate_GCTA_COJO
from src.SWCA_FineMapping import iterate_BayesianFineMapping



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



# def transform_observed_SNPs_to_ma(_df_observed_Z, _df_ref_MAF, _N):
#     ### Required columns of the ma : "SNP	A1	A2	freq	b	se	p	N"
#     # display(_df_obs)

#     ##### (1) MAF

#     if "MAF" in _df_observed_Z:
#         df_obs_2 = _df_observed_Z.loc[:, ['SNP_LD', 'A1_LD', 'A2_LD', 'Z_fixed', 'MAF']]
#     else:
#         df_obs_2 = _df_observed_Z.merge(
#             _df_ref_MAF.drop(['CHR', 'NCHROBS'], axis=1, errors='ignore'),
#             left_on=['SNP_LD', 'A1_LD', 'A2_LD'], right_on=['SNP', 'A1', 'A2']
#         )

#         ## `_df_ref_MAF`에 있는 불필요한 columns들 제거.
#         df_obs_2 = df_obs_2.drop(['SNP', 'A1', 'A2'], axis=1, errors='ignore')
#         # display(df_obs_2)

#     ##### (2) N
#     if "N" not in _df_observed_Z:
#         df_obs_2 = pd.concat(
#             [
#                 df_obs_2,
#                 pd.Series([_N] * df_obs_2.shape[0], index=df_obs_2.index, name='N')
#             ],
#             axis=1
#         )

#     ##### (3) SE
#     sr_SE = pd.Series([1.0] * _df_observed_Z.shape[0], name='se', index=df_obs_2.index)

#     ##### (4) P
#     sr_P = df_obs_2['Z_fixed'].abs().map(lambda x: 2 * sp.stats.norm.cdf(-x)).rename("p")

#     ##### RETURN
#     df_obs_3 = pd.concat(
#         [
#             df_obs_2,
#             sr_SE,
#             sr_P
#         ],
#         axis=1
#     ) \
#                    .loc[:, ["SNP_LD", "A1_LD", "A2_LD", "MAF", "Z_fixed", "se", "p", "N"]] \
#         .rename({
#         "SNP_LD": "SNP", "A1_LD": "A1", "A2_LD": "A2", "MAF": "freq", "Z_fixed": "b"
#     }, axis=1)

#     return df_obs_3



def __MAIN__(_fpath_sumstats3, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _fpath_PP, _out_prefix, _N,
             _module="Bayesian", _f_include_SNPs=False, _f_use_finemapping=True, _f_single_factor_markers=False,
             _r2_pred=0.6, _ncp=5.2, _maf_imputed=0.05, _N_max_iter=5, _gcta="/home/wschoi/bin/gcta64",
             _plink="/home/wschoi/miniconda3/bin/plink"):


    ##### (0) load data

    df_LDmatrix = pd.read_csv(_fpath_ref_ld, sep='\t', header=0) \
                        if isinstance(_fpath_ref_ld, str) else _fpath_ref_ld
    df_LDmatrix.index = df_LDmatrix.columns
        # next top 찾을 때 fine-mapping 활용하게 되면 LDmatrix 한 번 load하고 재사용하게 해야 함.

    df_ref_MAF = pd.read_csv(_fpath_ref_MAF, sep='\s+', header=0).drop(['NCHROBS'], axis=1) \
                    if isinstance(_fpath_ref_MAF, str) else _fpath_ref_MAF

    df_sumstats3 = pd.read_csv(_fpath_sumstats3, sep='\t', header=0) \
                    if isinstance(_fpath_sumstats3, str) else _fpath_sumstats3


    
    ##### (1) Summary Imputation

    df_Z_imputed = mod_SummaryImp.__MAIN__(df_sumstats3, df_LDmatrix, df_ref_MAF)
    df_Z_imputed_r2pred = df_Z_imputed[ df_Z_imputed['r2_pred'] >= _r2_pred ]

    df_Z_imputed.to_csv(_out_prefix + ".Z_imputed", sep='\t', header=True, index=False, na_rep="NA")
    # print(df_Z_imputed)
    # print(df_Z_imputed_r2pred)


    ##### (2) make the COJO input

    df_ma = transform_imputed_Z_to_ma(df_Z_imputed, df_ref_MAF, _N)
    df_ma_r2_pred = transform_imputed_Z_to_ma(df_Z_imputed_r2pred, df_ref_MAF, _N)

    OUT_ma = _out_prefix + ".ma"
    OUT_ma_r2_pred = _out_prefix + f".r2pred{_r2_pred}.ma"

    df_ma.to_csv(OUT_ma, sep='\t', header=True, index=False, na_rep="NA")
    df_ma_r2_pred.to_csv(OUT_ma_r2_pred, sep='\t', header=True, index=False, na_rep="NA")



    ##### (3) new out_prefix for COJO
    basename_temp = basename(_out_prefix) + ".ROUND_"
    dirname_temp = dirname(_out_prefix)

    out_dir_COJO = join(dirname_temp, basename_temp)
    out_prefix_COJO = join(out_dir_COJO, basename_temp)

    os.makedirs(out_dir_COJO, exist_ok=True)



    ##### (4) Main - SWCA based on module types.

    ## (2025.05.22.) 사실상 _module == "Bayesian" 때만 돌아감.

    if _module == "Bayesian":

        # df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0).sort_values("PP", ascending=False)
        df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0)

        df_PP = df_PP[ df_PP['SNP'].isin(df_ma_r2_pred['SNP']) ] # (2025.05.28.) r2pred로 filter하고 남은 애들만.
        # df_PP = df_PP[ df_PP['CredibleSet'] ]
        df_PP = df_PP.iloc[[0], :]

        print(df_PP)

        print(f"Initial_top_signal (One): {df_PP['SNP'].tolist()}")

        ## input ma파일로 이제 r2_pred로 thresholding한게 들어감.
        l_conditions, d_conditions = iterate_BayesianFineMapping(
            df_ma_r2_pred, df_PP['SNP'].tolist(),
            _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF,
            out_prefix_COJO,
            _ncp=_ncp,
            _N_max_iter=_N_max_iter, _f_polymoprhic_marker=_f_single_factor_markers,
            _plink=_plink
        )

        return l_conditions, d_conditions, OUT_ma_r2_pred

    elif _module == "pC":
        pass
    elif _module == "GCTA-COJO": # (deprecated; 2025.05.22.)

        ##### (3) iterate GCTA COJO "--cojo-cond"

        df_ma = make_COJO_input(df_Z_imputed, df_ref_MAF, _N, _maf_imputed=_maf_imputed,
                                _df_ss_SNP=(df_sumstats3 if _f_include_SNPs else None))

        OUT_ma = _out_prefix + ".maf{}.ma".format(str(_maf_imputed).replace(".", "_"))
        df_ma.to_csv(OUT_ma, sep='\t', header=True, index=False, na_rep="NA")

        df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0) \
                    .sort_values("PP", ascending=False) \
                    .iloc[:5, :]

        # display(df_PP)

        initial_top_signal = df_PP['SNP'].iat[0]
        print("initial_top_signal: {}".format(initial_top_signal))



        l_secondary_signals = iterate_GCTA_COJO(
            OUT_ma, initial_top_signal,
            _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF,
            out_prefix_COJO,
            _f_use_finemapping=_f_use_finemapping, _f_single_factor_markers=_f_single_factor_markers,
            _gcta=_gcta, _plink=_plink
        )


        return l_secondary_signals, OUT_ma

    else:
        raise ValueError("Wrong module type for SWCA!")




class SWCA:

    def __init__(self, 
                 _df_sumstats3: pd.DataFrame, 
                 _df_ref_ld: pd.DataFrame, _df_ref_MAF: pd.DataFrame, _fpath_ref_bfile: str, 
                 _fpath_PP: str, 
                 _out_prefix:str, _N: int,
                 _module_CMVN="Bayesian", 
                 _f_include_SNPs=False, _f_use_finemapping=True,
                 _r2_pred=0.6, _ncp=5.2, _N_max_iter=5, 
                 _gcta="/home/wschoi/bin/gcta64", _plink="/home/wschoi/miniconda3/bin/plink"):
        
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

        self.l_conditions = []
        self.d_conditions = {}


        ##### External software
        self.gcta = _gcta
        self.plink = _plink


    @classmethod # Factory method 1
    def from_paths(cls, _fpath_sumstats3, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _fpath_PP, _out_prefix, _N,
                   **kwargs):

        df_sumstats3 = pd.read_csv(_fpath_sumstats3, sep='\t', header=0)

        df_ref_ld = pd.read_csv(_fpath_ref_ld, sep='\t', header=0)
        df_ref_ld.index = df_ref_ld.columns

        df_ref_MAF = pd.read_csv(_fpath_ref_MAF, sep='\s+', header=0).drop(['NCHROBS'], axis=1)

        return cls(
            df_sumstats3, df_ref_ld, df_ref_MAF, _fpath_ref_bfile, _fpath_PP, _out_prefix, _N, **kwargs
        )


    @classmethod # Factory method 2
    def from_config(cls, ): 

        pass


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
        self.df_ma_r2_pred = transform_imputed_Z_to_ma(self.df_Z_imputed_r2pred, self.df_ref_MAF, self.N)

        self.fpath_ma = self.out_prefix + ".ma"
        self.fpath_ma_r2pred = self.out_prefix + f".r2pred{self.r2_pred}.ma"

        self.df_ma.to_csv(self.fpath_ma, sep='\t', header=True, index=False, na_rep="NA")
        self.df_ma_r2_pred.to_csv(self.fpath_ma_r2pred, sep='\t', header=True, index=False, na_rep="NA")

        return 0
    

    def perform_SWCA_Bayesian(self, _out_dir=None, _l_top_signals=None):

        ##### (3) iterate with Bayesian fine-mapping

        l_top_signals = []

        if isinstance(_l_top_signals, list):
            l_top_signals = _l_top_signals
        else:

            df_PP = pd.read_csv(self.fpath_PP, sep='\t', header=0) # 예전에 `.sort_values("PP", ascending=False)` 이렇게했는데 이제 안함.

            df_PP = df_PP[ df_PP['SNP'].isin(self.df_ma_r2_pred['SNP']) ] # (2025.05.28.) r2pred로 filter하고 남은 애들만.

            # df_PP = df_PP[ df_PP['CredibleSet'] ] # 예전에는 credible set 단위로 넣었음.
            df_PP = df_PP.iloc[[0], :] # 지금은 정말 top 딱 하나만.

            print(df_PP)

            l_top_signals = df_PP['SNP'].tolist()

        
        print(f"=====[ ROUND 1 ]: {l_top_signals}")


        if isinstance(_out_dir, str):
            out_prefix_SWCA = join(_out_dir, basename(self.out_prefix_SWCA)) # output directory만 다르게 주고 싶은 경우.
        else:
            out_prefix_SWCA = self.out_prefix_SWCA


        ## input ma파일로 이제 r2_pred로 thresholding한게 들어감.
        self.l_conditions, self.d_conditions = iterate_BayesianFineMapping(
            self.df_ma_r2_pred, l_top_signals,
            self.fpath_ref_bfile, self.df_ref_ld,
            out_prefix_SWCA,
            _ncp=self.ncp,
            _N_max_iter=self.N_max_iter, _f_polymoprhic_marker=False,
            _plink=self.plink
        )
            # `_f_polymoprhic_marker`얘는 당분간 쓰지 말것.
        
        return 0
    

    def perform_SWCA_GCTA(self, _out_dir=None, _l_top_signals=None):

        ##### (3) iterate GCTA COJO "--cojo-cond"

        df_PP = pd.read_csv(self.fpath_PP, sep='\t', header=0)
            # 어차피 GCTA도 `df_PP`가 필요함. ROUND_1 top signal은 PP로 찾았을 테니까.

        df_PP = df_PP[ df_PP['SNP'].isin(self.df_ma_r2_pred['SNP']) ] # (2025.05.28.) r2pred로 filter하고 남은 애들만.

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

        
        return 0



    def run(self): # MAIN funciton / orchestrator

        self.perform_Z_imputation()
        self.prepare_COJO_input()

        if self.module_CMVN == "Bayeisan":
            self.perform_SWCA_Bayesian()
        else:
            self.perform_SWCA_GCTA()

        return 0


    def __MAIN__(self):

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
        self.df_ma_r2_pred = transform_imputed_Z_to_ma(self.df_Z_imputed_r2pred, self.df_ref_MAF, self.N)

        OUT_ma = self.out_prefix + ".ma"
        OUT_ma_r2_pred = self.out_prefix + f".r2pred{self.r2_pred}.ma"

        self.df_ma.to_csv(OUT_ma, sep='\t', header=True, index=False, na_rep="NA")
        self.df_ma_r2_pred.to_csv(OUT_ma_r2_pred, sep='\t', header=True, index=False, na_rep="NA")



        return 0



if __name__ == '__main__':

    ##### `__MAIN__` test.

    r_temp = __MAIN__(
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.M613.sumstats3",
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.LD.whole.M3703.ld",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.PP",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250418_TEST_2.HLA", 58284, _f_include_SNPs=False)
    print("Final secondary signals:")
    print(r_temp)



    pass



"""
(Usage example)

r = __MAIN__(
    d_RA_yuki_clumped['sumstats'], d_RA_yuki_clumped['ld'], d_RA_yuki_clumped['MAF'],
    "20250218_ImpG_HLA_v3/RA_yuki.EUR.CovModel.whole.whole.PP",
    "20250218_ImpG_HLA_v3/TEST.RA_yuki", 
    58284
)

"""