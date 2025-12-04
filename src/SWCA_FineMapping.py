import os, sys, re
from os.path import basename, dirname, join
import numpy as np
import pandas as pd

from src.SWCA_CMVN import calc_Z_CMVN
import src.SWCA_PostCalc as SWCA_PostCalc



def transform_CMVN_result_to_ma(_df_CMVN_result, _f_exclude_MAF=True):
    ### 간단함. column 재구성 && rename 이거 둘이 끝임.

    """
    Chr	SNP	bp	refA	freq	b	se	p	n	freq_geno	bC	bC_se	pC    
    """

    # display(_df_CMVN_result)

    df_RETURN = pd.concat(
        [
            _df_CMVN_result[
                (['SNP', 'A1', 'A2'] if _f_exclude_MAF else ['SNP', 'A1', 'A2', 'freq'])
            ],
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
        # m_HLA = re.match(r'HLA_\S+_\d+', _SNP)

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
                                _fpath_ref, _df_ref_LD,
                                _out_prefix,
                                _ncp=5.2,
                                _N_max_iter=5, _f_polymoprhic_marker=False,
                                _plink="/home/wschoi/miniconda3/bin/plink"):
    
    ##### (0) Preliminaries

    os.makedirs(dirname(_out_prefix), exist_ok=True)

    ### initial ma file
    df_ma_init = pd.read_csv(_df_ma_init, sep='\t', header=0) if isinstance(_df_ma_init, str) else _df_ma_init

    ### initial conditions
    l_condition = _cond_init.copy() if isinstance(_cond_init, list) else [_cond_init]  # R0's top

    if _f_polymoprhic_marker:
        """
        - 여기 지금 좀 문제 있음. (2025.05.15.)
        - HLA_DRB1_01 -> 'HLA_DRB1_01'만 가져오지 않고 'HLA_DRB1_0101'도 가져옴 => duplicated markers to condition.
        - 걍 일단 당분간은 쓰지 마셈. 당장 필요한 것도 아니고.
        """
        l_condition = \
            [_ for item in map(lambda x: get_polymorphic_locus_markers(df_ma_init, x), l_condition) for _ in item]

    ### LD matrix
    if isinstance(_df_ref_LD, str):
        df_ref_LD = pd.read_csv(_df_ref_LD, sep='\t', header=0)
        df_ref_LD.index = df_ref_LD.columns
    else:
        df_ref_LD = _df_ref_LD


    ### output variable
    d_OUT_conditions = {1: l_condition.copy()}



    ##### (1) Main iteration

    for i in range(2, _N_max_iter + 1):

        print("\n=====[ ROUND {} ]".format(i))
        print(f"Conditions: {l_condition}")

        ##### (1) CMVN으로 conditional Z/BETA 계산
        df_CMVN_temp = calc_Z_CMVN(df_ma_init, df_ref_LD, l_condition)
        print(df_CMVN_temp.sort_values("cP").head(10))

        ## cma format으로 변환
        df_CMVN_temp_2 = transform_CMVN_result_to_ma(df_CMVN_temp)

        OUT_CMVN = _out_prefix + f".ROUND_{i}.SWCA"
        df_CMVN_temp_2.sort_values("pC").to_csv(OUT_CMVN, sep='\t', header=True, index=False, na_rep="NA")


        ##### End condition (2025.06.04.) 엎으면 PP계산할 이유가 없음.
        if df_CMVN_temp['cP'].sort_values().iat[0] > 5e-8:
            print("No more signals!")
            break



        ##### (2) Bayesian fine-mapping으로 next top 찾기
        df_PP_cma, OUT_PP_cma = SWCA_PostCalc.__MAIN__(
            OUT_CMVN,
            _fpath_ref,
            df_ref_LD,
            dirname(_out_prefix),
            _plink,
            _ncp=_ncp
        )

        # df_PP_cma_Credible = df_PP_cma[df_PP_cma['CredibleSet']]
        df_PP_cma_Credible = df_PP_cma.iloc[[0], :]
        print(df_PP_cma_Credible)

        l_condition_next = df_PP_cma_Credible['SNP'].tolist()

        if _f_polymoprhic_marker:
            l_condition_next = \
                [_ for item in map(lambda x: get_polymorphic_locus_markers(df_ma_init, x), l_condition_next) for _ in item]
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

        # ##### End condition
        # if df_CMVN_temp['cP'].sort_values().iat[0] > 5e-8:
        #     print("No more signals!")
        #     break

    return l_condition, {f"ROUND_{k}": v for k, v in d_OUT_conditions.items()}





if __name__ == '__main__':

    str_HLA_DRB1 = """
    HLA_DRB1_01
    HLA_DRB1_0101
    HLA_DRB1_0102
    HLA_DRB1_03
    HLA_DRB1_0301
    HLA_DRB1_04
    HLA_DRB1_0401
    HLA_DRB1_0402
    HLA_DRB1_0403
    HLA_DRB1_0404
    HLA_DRB1_0405
    HLA_DRB1_07
    HLA_DRB1_0701
    HLA_DRB1_08
    HLA_DRB1_0801
    HLA_DRB1_09
    HLA_DRB1_0901
    HLA_DRB1_11
    HLA_DRB1_1101
    HLA_DRB1_1104
    HLA_DRB1_12
    HLA_DRB1_13
    HLA_DRB1_1301
    HLA_DRB1_1302
    HLA_DRB1_14
    HLA_DRB1_15
    HLA_DRB1_1501
    HLA_DRB1_1502
    HLA_DRB1_16
    HLA_DRB1_1601
    """

    l_HLA_DRB1 = str_HLA_DRB1.split() # T1DGC에 있는 모든 HLA-DRB1 allele markers
    print(l_HLA_DRB1)


    ### Test / 20250515 / RA_soumya_2022 / maf 5% / HLA-DRB1 alleles conditioned (가장 마지막으로 expected한 상태로 나온 결과.)

    df_ma_RA_soumya_2022 = pd.read_csv(
        "/data02/wschoi/_hCAVIAR_v2/20250508_Soumya_summary/20250513_RA_soumya_recent.maf0_05.ma",
        sep='\t', header=0
    )

    l_HLA_DRB1_maf0_05 = [_ for _ in l_HLA_DRB1 if (_ in df_ma_RA_soumya_2022['SNP'].tolist())]
    print(l_HLA_DRB1_maf0_05)



    r, r_dict = iterate_BayesianFineMapping(
        df_ma_RA_soumya_2022,
        l_HLA_DRB1_maf0_05,
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld",
        None,
        "/data02/wschoi/_hCAVIAR_v2/20250508_Soumya_summary/20250515_IntegrationTest.Soumya_2022.maf0_05/20250515_IntegrationTest.Soumya_2022.maf0_05",
        _N_max_iter=5,
        _f_polymoprhic_marker=False
    )



    ### Test / 20250515 / RA_yuki_2013 / maf 5% / HLA-DRB1 alleles conditioned (가장 마지막으로 expected한 상태로 나온 결과.)

    # df_ma_RA_yuki_2013 = pd.read_csv(
    #     "/data02/wschoi/_hCAVIAR_v2/20250407_rerun_GCatal/T1DGC+RA.EUR.GCST002318.hg19.chr6.29-34mb.maf0_05.ma",
    #     sep='\t', header=0
    # )
    #
    # l_HLA_DRB1_maf0_05 = [_ for _ in l_HLA_DRB1 if (_ in df_ma_RA_yuki_2013['SNP'].tolist())]
    # print(l_HLA_DRB1_maf0_05)
    #
    # r, r_dict = iterate_BayesianFineMapping(
    #     df_ma_RA_yuki_2013,
    #     l_HLA_DRB1_maf0_05,
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld",
    #     None,
    #     "/data02/wschoi/_hCAVIAR_v2/20250508_Soumya_summary/20250515_IntegrationTest.Yuki_2013.maf0_05.ROUND_/20250515_IntegrationTest.Yuki_2013.maf0_05",
    #     _N_max_iter=5,
    #     _f_polymoprhic_marker=False
    # )


    pass