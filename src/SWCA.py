import os, re
from os.path import basename, dirname, join
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



def transform_observed_SNPs_to_ma(_df_observed_Z, _df_ref_MAF, _N):
    ### Required columns of the ma : "SNP	A1	A2	freq	b	se	p	N"
    # display(_df_obs)

    ##### (1) MAF

    if "MAF" in _df_observed_Z:
        df_obs_2 = _df_observed_Z.loc[:, ['SNP_LD', 'A1_LD', 'A2_LD', 'Z_fixed', 'MAF']]
    else:
        df_obs_2 = _df_observed_Z.merge(
            _df_ref_MAF.drop(['CHR', 'NCHROBS'], axis=1, errors='ignore'),
            left_on=['SNP_LD', 'A1_LD', 'A2_LD'], right_on=['SNP', 'A1', 'A2']
        )

        ## `_df_ref_MAF`에 있는 불필요한 columns들 제거.
        df_obs_2 = df_obs_2.drop(['SNP', 'A1', 'A2'], axis=1, errors='ignore')
        # display(df_obs_2)

    ##### (2) N
    if "N" not in _df_observed_Z:
        df_obs_2 = pd.concat(
            [
                df_obs_2,
                pd.Series([_N] * df_obs_2.shape[0], index=df_obs_2.index, name='N')
            ],
            axis=1
        )

    ##### (3) SE
    sr_SE = pd.Series([1.0] * _df_observed_Z.shape[0], name='se', index=df_obs_2.index)

    ##### (4) P
    sr_P = df_obs_2['Z_fixed'].abs().map(lambda x: 2 * sp.stats.norm.cdf(-x)).rename("p")

    ##### RETURN
    df_obs_3 = pd.concat(
        [
            df_obs_2,
            sr_SE,
            sr_P
        ],
        axis=1
    ) \
                   .loc[:, ["SNP_LD", "A1_LD", "A2_LD", "MAF", "Z_fixed", "se", "p", "N"]] \
        .rename({
        "SNP_LD": "SNP", "A1_LD": "A1", "A2_LD": "A2", "MAF": "freq", "Z_fixed": "b"
    }, axis=1)

    return df_obs_3



def __MAIN__(_fpath_ss, _fpath_ref_ld, _fpath_ref_bfile, _fpath_ref_MAF, _fpath_PP, _out_prefix, _N,
             _module="Bayesian",
             _f_include_SNPs=False, _f_use_finemapping=True, _f_single_factor_markers=False,
             _r2_pred=0.85,
             _maf_imputed=0.05, _N_max_iter=5,
             _gcta="/home/wschoi/bin/gcta64", _plink="/home/wschoi/miniconda3/bin/plink"):


    ##### (0) load data

    df_LDmatrix = pd.read_csv(_fpath_ref_ld, sep='\t', header=0) \
                        if isinstance(_fpath_ref_ld, str) else _fpath_ref_ld
    df_LDmatrix.index = df_LDmatrix.columns
        # next top 찾을 때 fine-mapping 활용하게 되면 LDmatrix 한 번 load하고 재사용하게 해야 함.

    df_ref_MAF = pd.read_csv(_fpath_ref_MAF, sep='\s+', header=0).drop(['NCHROBS'], axis=1) \
                    if isinstance(_fpath_ref_MAF, str) else _fpath_ref_MAF

    df_observed_Z = pd.read_csv(_fpath_ss, sep='\t', header=0) \
                    if isinstance(_fpath_ss, str) else _fpath_ss


    
    ##### (1) Summary Imputation

    df_Z_imputed = mod_SummaryImp.__MAIN__(df_observed_Z, df_LDmatrix, df_ref_MAF)
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

        df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0) \
                    .sort_values("PP", ascending=False)
        df_PP = df_PP[ df_PP['CredibleSet'] ]
        print(f"Initial_top_signal: {df_PP['SNP'].tolist()}")

        ## input ma파일로 이제 r2_pred로 thresholding한게 들어감.
        l_conditions, d_conditions = iterate_BayesianFineMapping(
            df_ma_r2_pred, df_PP['SNP'].tolist(),
            _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF,
            out_prefix_COJO,
            _N_max_iter=_N_max_iter, _f_polymoprhic_marker=_f_single_factor_markers,
            _plink=_plink
        )

        return l_conditions, d_conditions, OUT_ma_r2_pred

    elif _module == "pC":
        pass
    elif _module == "GCTA-COJO": # (deprecated; 2025.05.22.)

        ##### (3) iterate GCTA COJO "--cojo-cond"

        df_ma = make_COJO_input(df_Z_imputed, df_ref_MAF, _N, _maf_imputed=_maf_imputed,
                                _df_ss_SNP=(df_observed_Z if _f_include_SNPs else None))

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





if __name__ == '__main__':

    ##### `__MAIN__` test.

    r_temp = __MAIN__(
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.M613.sumstats3",
        "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.LD.whole.M3703.ld",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.whole.PP",
        "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250418_TEST_2.HLA",
        58284,
        _f_include_SNPs=False
    )
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