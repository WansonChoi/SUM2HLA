import os, sys, re
import numpy as np
import pandas as pd

from src.SWCA_CMVN import calc_Z_CMVN
import src.SWCA_PostCalc as SWCA_PostCalc



def transform_CMVN_result_to_ma(_df_CMVN_result):
    ### 간단함. column 재구성 && rename 이거 둘이 끝임.

    """
    Chr	SNP	bp	refA	freq	b	se	p	n	freq_geno	bC	bC_se	pC    
    """

    # display(_df_CMVN_result)

    df_RETURN = pd.concat(
        [
            _df_CMVN_result[['SNP', 'A1', 'A2', 'freq']],
            _df_CMVN_result['cZ'].rename("bC"),
            _df_CMVN_result['se'].rename("bC_se"),
            _df_CMVN_result['cP'].rename("pC"),
            _df_CMVN_result['N'].rename("n"),
        ],
        axis=1
    )

    return df_RETURN



def get_polymorphic_locus_markers(_df, _SNP) -> list:

    def get_polymorphic_locus_marker_prefix(_SNP):
        m_AA_polymorphic = re.match(r'(AA_\S+_-?\d+_\d+_)\S+', _SNP)
        m_intraSNP_polymorphic = re.match(r'(SNP_\S+_\d+_)\S+', _SNP)  # SNP_DQB1_32740666_GA

        if bool(m_AA_polymorphic):
            return m_AA_polymorphic.group(1)
        elif bool(m_intraSNP_polymorphic):
            return m_intraSNP_polymorphic.group(1)
        else:
            return _SNP

    f_isin = _df['SNP'].str.startswith(get_polymorphic_locus_marker_prefix(_SNP))
    df_RETURN = _df[f_isin]

    return df_RETURN['SNP'].tolist()

"""
print(get_polymorphic_locus_marker_prefix("AA_DQB1_57_32740666_AD"))
display(get_polymorphic_locus_markers(df_T1D_answer_HLA_SNP_M2447, "AA_DQB1_57_32740666_AD"))

print(get_polymorphic_locus_marker_prefix("SNP_DQB1_32740666_GA"))
display(get_polymorphic_locus_markers(df_T1D_answer_HLA_SNP_M2447, "SNP_DQB1_32740666_GA"))
"""



def iterate_BayesianFineMapping(_df_ma_init, _cond_init,
                                _fpath_ref, _df_ref_LD, _df_ref_FRQ,
                                _out_prefix,
                                _N_max_iter=100, _f_polymoprhic_marker=False,
                                _plink="/home/wschoi/miniconda3/bin/plink"):

    os.makedirs(os.path.dirname(_out_prefix), exist_ok=True)

    ##### R0

    l_condition = _cond_init.copy() if isinstance(_cond_init, list) else [_cond_init]  # R0's top

    if _f_polymoprhic_marker:
        l_condition = \
            [_ for item in map(lambda x: get_polymorphic_locus_markers(_df_ma_init, x), l_condition) for _ in item]

    d_OUT_conditions = {0: l_condition.copy()}

    for i in range(1, _N_max_iter + 1):

        print("\n=====[ ROUND {} ]".format(i))
        print(f"Conditions: {l_condition}")

        ##### (1) CMVN으로 conditional Z/BETA 계산
        df_CMVN_temp = calc_Z_CMVN(_df_ma_init, _df_ref_LD, l_condition)
        print(df_CMVN_temp.sort_values("cP").head(10))

        ## cma format으로 변환
        df_CMVN_temp_2 = transform_CMVN_result_to_ma(df_CMVN_temp)

        OUT_CMVN = _out_prefix + f".ROUND_{i}.cma.cojo"
        df_CMVN_temp_2.to_csv(OUT_CMVN, sep='\t', header=True, index=False, na_rep="NA")

        ##### (2) Bayesian fine-mapping으로 next top 찾기
        df_PP_cma, OUT_PP_cma = SWCA_PostCalc.__MAIN__(
            OUT_CMVN,
            _fpath_ref,
            _df_ref_LD,
            os.path.dirname(_out_prefix),
            _plink
        )

        df_PP_cma_Credible = df_PP_cma[df_PP_cma['CredibleSet']]
        print(df_PP_cma_Credible)

        l_condition_next = df_PP_cma_Credible['SNP'].tolist()

        if _f_polymoprhic_marker:
            l_condition_next = \
                [_ for item in map(lambda x: get_polymorphic_locus_markers(_df_ma_init, x), l_condition_next) for _ in item]
            l_condition_next = list(np.unique(l_condition_next))
            """
            (ex)
             	rank 	index 	SNP 	LL+Lprior 	LL+Lprior_diff 	LL+Lprior_diff_acc 	logPP 	PP 	CredibleSet
            0 	0 	62 	AA_B_97_31432180_RT 	30.852898 	0.000000 	0.000000 	-0.367145 	0.692709 	True
            1 	1 	9 	AA_A_62_30018696_QR 	29.544792 	1.308105 	1.308105 	-1.675250 	0.187261 	True
            2 	2 	63 	AA_B_97_31432180_SRT 	28.877922 	0.666870 	1.974976 	-2.342120 	0.096124 	True            

            - AA_B_97_31432180 이런 예외도 있네
            """

        l_condition.extend(l_condition_next)
        d_OUT_conditions[i] = l_condition_next

        ##### End condition
        if df_CMVN_temp['cP'].sort_values().iat[0] > 5e-8:
            print("No more signals!")
            break

    return l_condition, d_OUT_conditions





if __name__ == '__main__':

    # transform_CMVN_result_to_ma(r_answer_HLA_SNP_fromTop_PolyMarker_Bayesian_R1)

    # r, r_dict = iterate_SWCA_with_BayesianFineMapping(
    #     df_T1D_answer_HLA_SNP_M2447, "AA_DQB1_57_32740666_D",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA", df_T1DGC_LD, df_T1DGC_MAF,
    #     "/data02/wschoi/_hCAVIAR_v2/20250501_SWCA_GTvsSS_v6/20250505_TEST/TEST",
    #     _N_max_iter=2,
    #     _f_polymoprhic_marker=True
    # )


    pass