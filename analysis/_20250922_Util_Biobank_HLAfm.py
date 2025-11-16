import os, sys, re
import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2



"""
(2025.06.19.)
- 정답 item가지고 와서 정답 table만드는 유틸 script.

(2025.09.22.)
- (뭐하다가 벌써 3개월이 지났냐...)
- 


"""


def find_item(_index, _phe_name_ss, _df_PP, _ANSWER_SNP2HLA, _MarkerType, _HLAgene, _pos, _Allele_Residue):

    # _df_PP_2 = pd.concat(
    # [
    #         (_df_PP['rank'] / _df_PP.shape[0]).rename("rank_p"),
    #         _df_PP[['rank', 'SNP', 'PP', 'CredibleSet']]
    #     ],
    #     axis=1
    # )
    _df_PP_2 = _df_PP[['rank', 'rank_p', 'SNP', 'PP', 'CredibleSet']]
    # display(_df_PP_2.head())

    ##### (1) 'SNP' column으로 정답 row 찾기. (cf. credible set을 먼저 남길 필요 없을 듯.)

    f_find_answer = None

    ## 둘 다 AA를 위한 flag

    if _MarkerType == "AA":

        ### 일단 찾고
        f_find_answer = _df_PP_2['SNP'].str.startswith(_ANSWER_SNP2HLA)

        ### residue가 2개 이상인 애들
        if np.count_nonzero(f_find_answer) > 1:  # True개수 count

            """
            (ex)
            AA_DRB1_26_ => AA_DRB1_26_32660070_{F,Y,L} 3개 걸림.
            """

            df_Found = _df_PP_2[f_find_answer]

            p_residue = re.compile(r"^AA_\S+_-?\d+_\d+_(\S+)$")
            sr_residue = df_Found['SNP'].str.extract(p_residue, expand=False)

            f_find_answer_2_residue = sr_residue == _Allele_Residue

            if np.count_nonzero(f_find_answer_2_residue) == 1:
                f_find_answer = f_find_answer & f_find_answer_2_residue  # broadcasting되는거 같음.

            else:
                pass

            # display(sr_residue)
            # display(f_find_answer_2_residue)
            # display(f_find_answer)

        elif np.count_nonzero(f_find_answer) == 1:
            pass

        else:
            print(f"=====[{_index}]: {_phe_name_ss} / The answer marker can't be found!")
            return -1



    elif _MarkerType == "HLA":  # 얘는 간단함.

        f_find_answer = (_df_PP_2['SNP'] == _ANSWER_SNP2HLA)  # (ex. HLA_B_08 => HLA_B_0801도 찾아올 수 있음.)

        if np.count_nonzero(f_find_answer) > 1:
            print(f"=====[{_index}]: {_phe_name_ss} / Wrong answer HLA alleles!")
            return -1
        elif np.count_nonzero(f_find_answer) == 1:  # No problem
            pass
        else:
            print(f"=====[{_index}]: {_phe_name_ss} / The answer marker can't be found!")
            return -1

    else:
        print(f"=====[{_index}]: {_phe_name_ss} / Wrong marker type!")
        return -1

    if np.count_nonzero(f_find_answer) > 1:  # 여전히 2개 이상의 rows가 찾아짐.
        print(f"=====[{_index}]: {_phe_name_ss} / Found more than 1 rows!")
        print(_df_PP_2[f_find_answer])
        return -1

    df_Found = _df_PP_2[f_find_answer]

    return df_Found

"""
- (Usage example)
    = find_item(0, "Malignant_Lymphoma", df_temp, *['AA_DRB1_26_', 'AA', 'DRB1', '26', 'F'])
"""





def get_performance_eval_table(_df_ssfn, _df_HLAfm, _col_ToUse, _PPtype_ToUse, _ethnicity="EUR"):

    if _PPtype_ToUse not in [".AA+HLA.PP", ".AA.PP", ".HLA.PP", ".HLAtype.PP", ".SNP.PP", ".intraSNP.PP", ".whole.PP"]:
        raise ValueError("Error! Pick one from the below!:\n{}".format(
            [".AA+HLA.PP", ".AA.PP", ".HLA.PP", ".HLAtype.PP", ".SNP.PP", ".intraSNP.PP", ".whole.PP"]
        ))

    if _col_ToUse not in _df_ssfn:
        raise ValueError(f"No '{_col_ToUse}' in the given SSFN file!")

    ##### (0) load data

    ### (0-1) 편의상 round 1만 남기기. 일단 당장은 top을 잘 맞췆는지 부터 봐야 함.
    df_HLAfm_R1 = _df_HLAfm[_df_HLAfm['Round'] == 1]

    ### (0-2) `df_ToScore` 만들기. (사실상 ToIter임
    if _ethnicity == "EUR":


        l_ToExtract = ['TraitType', 'Category', 'Phenotype', 'phe_name_ss', 'No. samples', 'No. cases', 'No. controls',
                       'ToSWCA', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'SUM2HLA_OUT']

        if _col_ToUse not in l_ToExtract: l_ToExtract += [_col_ToUse]

        df_ssfn_2 = _df_ssfn[l_ToExtract]

    elif _ethnicity == "EAS":
        """
        ['TraitType', 'phe_name_dir', 'phe_name_ss(phenocode)', 'phenostring', 'isBinary', 'BBJ_dir', 'num_cases', 'num_controls', 'num_samples', 'category', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'ToSWCA', 'SUM2HLA_OUT']        
        """
        l_ToExtract = ['TraitType', 'category', 'phe_name_ss(phenocode)', 'phenostring', 'num_cases', 'num_controls', 'num_samples',
                       'ToSWCA', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'SUM2HLA_OUT']

        if _col_ToUse not in l_ToExtract: l_ToExtract += [_col_ToUse]


        df_ssfn_2 = _df_ssfn[l_ToExtract] \
            .rename(
            {"category": "Category", 'phe_name_ss(phenocode)': "phe_name_ss", 'phenostring': "Phenotype", 'num_cases': 'No. cases',
             'num_controls': 'No. controls', 'num_samples': 'No. samples'}, axis=1
        )
    else:
        print("Wrong _ethnicity!")
        return -1


    df_ToScore = df_ssfn_2.merge(df_HLAfm_R1.drop(['Category'], axis=1).rename({"Trait": "phe_name_ss"}, axis=1))
        # ssfn으로 left_join함.

    print(df_HLAfm_R1.head(3))
    print(df_HLAfm_R1.shape)

    print(df_ToScore.head(3))
    print(df_ToScore.shape)
    print(df_ToScore.columns.tolist())

    ##### (1) Main iteration (find the answer marker in the PP file)

    d_out_SCORE = {}

    for _index, _sr_row in df_ToScore.iterrows():

        _phe_name_ss = _sr_row['phe_name_ss']
        _fpath_PP = (_sr_row[_col_ToUse] + _PPtype_ToUse)

        _ANSWER_SNP2HLA = _sr_row['ANSWER_SNP2HLA']
        _MarkerType = _sr_row['MarkerType']
        _HLAgene = _sr_row['HLAgene']
        _pos = _sr_row['pos']
        _Allele_Residue = _sr_row['Allele/Residue']

        # print(_ANSWER_SNP2HLA)
        # print(_MarkerType)

        print(f"=====[{_index}]: {_phe_name_ss} / {_ANSWER_SNP2HLA}")

        try:
            _df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0) # PP로 sorting되어있다 가정.
            # print(_df_PP.head(3))
        except FileNotFoundError:
            d_out_SCORE[_phe_name_ss] = None
            continue
        else:
            df_Found = find_item(_index, _phe_name_ss, _df_PP, _ANSWER_SNP2HLA, _MarkerType, _HLAgene, _pos,
                                 _Allele_Residue)
            # display(df_Found)

            ## (2025.09.22.) top candidate HLA variant with PP
            df_SNP_top_PP = _df_PP[['SNP']].iloc[[0], :].rename({"SNP": "SNP_top_PP"}, axis=1)
            df_SNP_top_PP.index = df_Found.index # 둘 다 row 개수 1개짜리라 가정.
            df_Found = pd.concat([df_Found, df_SNP_top_PP], axis=1) # _df_PP['SNP']가 주어진대로 top canddiate일거라 가정.

            # print(df_Found)

            d_out_SCORE[_phe_name_ss] = df_Found.copy()

        # if _index >= 10: break

    # l_ToExtract = ['phe_name_ss', 'SNP', 'PP', 'CredibleSet', 'rank', 'rank_p']
    l_ToExtract = ['phe_name_ss', 'SNP', 'PP', 'CredibleSet', 'rank', 'rank_p', 'SNP_top_PP']

    df_RETURN = pd.concat(d_out_SCORE, names=['phe_name_ss', 'index']) \
                    .reset_index("phe_name_ss", drop=False) \
                    .reset_index("index", drop=True) \
                    .loc[:, l_ToExtract]
    print(df_RETURN.columns.tolist())

    l_ToExtract = ['TraitType', 'Category', 'phe_name_ss', 'Phenotype', 'No. cases', 'No. controls', 'No. samples', 'HLA_Allele'] + \
                    ['Other', 'Effect', 'ImpRsq', 'N', 'Beta', 'SE', 'P', 'any_MHC_association'] + \
                    ["MarkerType", "HLAgene", "pos", "Allele/Residue"]
    df_RETURN = df_ToScore[l_ToExtract].rename({"HLA_Allele": "SNP_answer"}, axis=1) \
        .merge(df_RETURN, on='phe_name_ss')
    # display(df_temp)

    return df_RETURN



def get_r_r2_two_HLAvariants(_df_ref_LD, _sr_SNP_1, _sr_SNP_2):

    df_temp = pd.concat([_sr_SNP_1, _sr_SNP_2], axis=1)

    sr_r = pd.Series(
        [_df_ref_LD.loc[_snp1, _snp2] for _index, _snp1, _snp2 in df_temp.itertuples()],
        name='r',
        index=_sr_SNP_1.index
    )

    sr_r2 = sr_r.map(lambda x: x**2).rename("r2")


    df_RETURN = pd.concat([sr_r, sr_r2], axis=1)

    return df_RETURN



def do_manual_correction(_df_UKBB_Table):

    """
    Manaully 수정해줘야하는 애들 모음.

    """

    df_UKBB_Table_2 = _df_UKBB_Table.set_index("Phenotype", drop=False).copy()

    # display(df_UKBB_Table_2)

    ### (1) PsV
    df_UKBB_Table_2.loc['Psoriasis vulgaris', "rank"] = 1
    df_UKBB_Table_2.loc['Psoriasis vulgaris', 'rank_p'] = 0.0
    df_UKBB_Table_2.loc['Psoriasis vulgaris', 'SNP_top_PP'] = 'HLA_C_0602'
    df_UKBB_Table_2.loc['Psoriasis vulgaris', 'r'] = 1.0
    df_UKBB_Table_2.loc['Psoriasis vulgaris', 'r2'] = 1.0**2

    ### (2) RA
    df_UKBB_Table_2.loc['Rheumatoid arthritis', "rank"] = 1
    df_UKBB_Table_2.loc['Rheumatoid arthritis', 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['Rheumatoid arthritis', 'SNP_top_PP'] = 'AA_DRB1_11_32660115_VL'
    # df_UKBB_Table_2.loc['Rheumatoid arthritis', 'r'] = 1.0
    # df_UKBB_Table_2.loc['Rheumatoid arthritis', 'r2'] = 1.0**2
    df_UKBB_Table_2.loc['Rheumatoid arthritis', 'PP'] = 0.9861576204907249 # VL의 PP
    df_UKBB_Table_2.loc['Rheumatoid arthritis', 'CredibleSet'] = True

    ### (3) IgA
    df_UKBB_Table_2.loc['IgA nephritis', "rank"] = 1
    df_UKBB_Table_2.loc['IgA nephritis', 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['IgA nephritis', 'SNP_top_PP'] = ''
    # df_UKBB_Table_2.loc['IgA nephritis', 'r'] = 1.0
    # df_UKBB_Table_2.loc['IgA nephritis', 'r2'] = 1.0**2
    df_UKBB_Table_2.loc['IgA nephritis', 'CredibleSet'] = True


    ## 여기서 부터는 rank를 override하면 안됨.

    ### (4) Type 1 diabetes	
    # df_UKBB_Table_2.loc['Type 1 diabetes', "rank"] = 1
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'SNP_top_PP'] = ''
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r'] = 1.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r2'] = 1.0**2
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'CredibleSet'] = True
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'PP'] = 1.0 # 실제 "HLA-DQB1 a.a. position 57 D"의 PP


    ##### r / r2값.

    ### (6) Nephrotic syndrome
    # df_UKBB_Table_2.loc['Nephrotic syndrome', "rank"] = 1
    # df_UKBB_Table_2.loc['Nephrotic syndrome', 'rank_p'] = 0.0
    df_UKBB_Table_2.loc['Nephrotic syndrome', 'SNP_top_PP'] = 'HLA_DRB1_03' # ['HLA_DRB1_03', 'AA_DRB1_77_32659917', 'AA_DRB1_74_32659926_R'] 이거 셋이 공동 1등.
    # df_UKBB_Table_2.loc['', 'r'] = 1.0
    # df_UKBB_Table_2.loc['', 'r2'] = 1.0**2
    # df_UKBB_Table_2.loc['Nephrotic syndrome', 'CredibleSet'] = True
    # df_UKBB_Table_2.loc['Nephrotic syndrome', 'PP'] = 0.17409731656873034 # 실제 "HLA_DRB1_03"의 PP

    ### (7) Grave's disease
    # df_UKBB_Table_2.loc["Grave's disease", "rank"] = 1
    # df_UKBB_Table_2.loc["Grave's disease", 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'SNP_top_PP'] = ''
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r'] = 1.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r2'] = 1.0**2
    # df_UKBB_Table_2.loc["Grave's disease", 'CredibleSet'] = True
    # df_UKBB_Table_2.loc["Grave's disease", 'PP'] = 0.989493056799337

    ### (8) Asthma
    # df_UKBB_Table_2.loc['Asthma', "rank"] = 1
    # df_UKBB_Table_2.loc['Asthma', 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'SNP_top_PP'] = ''
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r'] = 1.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r2'] = 1.0**2
    # df_UKBB_Table_2.loc['Asthma', 'CredibleSet'] = True
    # df_UKBB_Table_2.loc['Asthma', 'PP'] = 0.0913029605423122

    ### (8) Pediatric asthma
    # df_UKBB_Table_2.loc['Pediatric asthma', "rank"] = 1
    # df_UKBB_Table_2.loc['Pediatric asthma', 'rank_p'] = 0.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'SNP_top_PP'] = ''
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r'] = 1.0
    # df_UKBB_Table_2.loc['Type 1 diabetes', 'r2'] = 1.0**2
    # df_UKBB_Table_2.loc['Pediatric asthma', 'CredibleSet'] = True
    # df_UKBB_Table_2.loc['Pediatric asthma', 'PP'] = 0.10427171300470647



    df_UKBB_Table_2 = df_UKBB_Table_2.reset_index(drop=True)
    # print(df_UKBB_Table_2)


    """
    논문에다 쓸 내용 미리 정리.

    - T1D는 well-established variant를 top으로 찍었다.
    - RA는 같은 amino acid position이고, 
        - 선행 연구에서 이 position의 SPG와 VL은 different effect direciton (risk/protection)을 가지면서 거의 같은 통계적 유의 수준으로 confer함이 알려져있었기 때문에 
        - replicate한 것으로 처리햇다.
    - IgA도 같은 amino acd position의 거의 같은 resiude의 조합, SWV vs. SV, 을 나타내는 두 varinats들이고, 이 두 variants들 간의 LD r값이 0.9이상이었기 때문에 replicate한것으로 분류했다.
    - Nephrotic syndrome, GD, Asthma, and Ped_asthma의 경우 모두 the reported top과 LD r값이 0.9이상이었기 때문에 replicate한 것으로 분류했다.
    
    
    """

    return df_UKBB_Table_2





def extract_headers_and_rename(_df):

    l_ToExtract = [
        'Category', 'Phenotype', 'SNP_answer', 'P', 'N',
        'rank', 'rank_p', 'PP', 'CredibleSet', 'SNP_top_PP', 'r', 'r2', 'No. cases', 'No. controls', 'No. samples',
    ]
        # ['N', 'No. cases', 'No. controls', 'No. samples'] 는 supple에만 들어가고 main table에서는 뺄거임.

    df_RETURN = _df.loc[:, l_ToExtract]


    d_ToRename = {
        'Category': "ICD 10",
        'Phenotype': "Disease Trait",
        'SNP_answer': "Top Variant (by P-value)",
        'P': "P-value",
        'rank': "SUM2HLA Rank (by PP)",
        'rank_p': "Percent Rank (by PP)",
        'CredibleSet': "Credible Set (99%)",
        "SNP_top_PP": "Top Variant (by PP)",

        "N": "N_total (of P-value)",
        'No. cases': "N_case (of PP)", 
        'No. controls': 'N_control (of PP)', 
        'No. samples': 'N_total (of PP)'
        
    }

    df_RETURN = df_RETURN.rename(d_ToRename, axis=1)


    return df_RETURN



def trim_columns(_df, _M):

    _df = _df.copy()

    """
    다음 columns들 좀 수정
    - "ICD 10": code만 남기기
    - "P-value": scientific notation
    - "Percent Rank (by PP)": precesion => 소수점 4째자리까지만. (deprecated)
    - "PP": 소수점 둘째자리까지만.
    - "Top Variant (by PP)": Saori format으로.
    - 'r' and 'r2': 둘 다 소수점 둘째 자리까지해서 합칠것.

    - "SUM2HLA Rank (by PP)"
    
    """

    ### (0) sample size는 main table에서는 안보여줄거임.
    _df = _df.drop(["N_total (of P-value)", "N_case (of PP)", 'N_control (of PP)', 'N_total (of PP)'], axis=1)


    ### (1) "P-value"
    _df['P-value'] = _df['P-value'].map(lambda x: f"{x:.1E}")  # scientific notation, 1자리 소수점


    ### (2) 'Percent Rank (by PP)' (deprecated)
    # _df['Percent Rank (by PP)'] = _df['Percent Rank (by PP)'].map(lambda x: 100 * x).map(lambda x: f"{x:.1f}")
    # _df = _df.rename({'Percent Rank (by PP)': f"Percent Rank (M={_M}, %)"}, axis=1)


    ### (3) SUM2HLA Rank (by PP) 에 '*' 박기
    df_temp = _df[["SUM2HLA Rank (by PP)", "r"]]
    sr_rank = pd.Series(
        [f"{_rank} (*)" if (_rank > 1 and _r >= 0.9) else f"{_rank}" for _index, _rank, _r in df_temp.itertuples()],
        index=_df.index, name="temp"
    )

    _df["SUM2HLA Rank (by PP)"] = sr_rank
    _df.drop(['Percent Rank (by PP)'], axis=1, inplace=True)
    _df = _df.rename({"SUM2HLA Rank (by PP)": f"SUM2HLA Rank (by PP, M={_M})"}, axis=1)


    ### (4) "PP"
    _df['PP'] = _df['PP'].map(lambda x: f"{x:.2f}")


    ### (5) "r" and "r2"
    def format_r_r2(r, r2, rank=None):
        def custom_round(x):
            # 1.00 근방은 강제로 1.00 처리
            if abs(x - 1.0) < 0.007:
                return "1.00"
            else:
                return f"{x:.2f}"
            
        return f"{custom_round(r)} ({custom_round(r2)})"
        # return f"{custom_round(r)} ({custom_round(r2)})" + ( " (*)" if r >= 0.9 and rank > 1 else "" )

    sr_r_r2 = pd.Series(
        [format_r_r2(r, r2) for _, r, r2 in _df[['r', 'r2']].itertuples()],
        name="r (r2)",
        index=_df.index
    )

    _df = pd.concat(
        [
            _df.drop(['r', 'r2'], axis=1),
            sr_r_r2
        ],
        axis=1
    )


    ### (5) Top Variant (by PP)
    def to_saori_format(_x):

        if _x.startswith("HLA"):

            l_temp = _x.split("_")

            _allele = l_temp[2]

            if bool(re.match(r'\d{4}', _allele)):
                _allele = _allele[:2] + ":" + _allele[2:]

            return f"HLA-{l_temp[1]}*{_allele}"
        
        elif _x.startswith("AA"):

            ## Hard-exception (UKBB만. BBJ는 할거 없음.)
            if _x == "AA_DRB1_77_32659917":
                return "HLA-DRB1 a.a. position 77 N"
            if _x == "AA_B_63_31432527":
                return "HLA-B a.a. position 63 E"


            l_temp = _x.split("_")

            _string = f"HLA-{l_temp[1]} a.a. position {l_temp[2]} {l_temp[4]}"

            # _HLA = f"HLA-{l_temp[1]}"
            # _pos = f"a.a. position" l_temp[2]
            # _residues = l_temp[4]

            return _string

        return "NA"

    _df['Top Variant (by PP)'] = _df['Top Variant (by PP)'].map(lambda x: to_saori_format(x))



    ### (6) ICD-10 
    _df['ICD 10'] = _df['ICD 10'].map(lambda x: x.split()[1])




    return _df


"""

- 우리는 임의로 diseases들을 3개의 classes들로 나눴다: (1) Autoimmune disease ("AutoImm"), (2) Immune-related disease ("Imm"), and (3) Others.
- a.a.은 amino acid를 축약한것이다.
- redidue characters들이 2개 이상 있는 건 OR관계를 나타낸다.
    - 예를 들어 "HLA-DQB1 a.a. position 57 AD"의 경우, 이 57번 position에서 alanine 혹은 aspartate인 variant를 나타낸다.
- residue 'x'는 deletion을 나타낸다.
- r and r2는 two top variants, each identified by P-value and PP, 들 간의 LD이다.

- percentile header관련
    - The "Percent Rank" column은 The "Rank (by PP)" column의 전체 M개의 HLA variant markers들 중 percentile 순위를 나타낸다 (1위를 0.0%로 하여)

"""


##### 이하 deprecated. (2025.09.22.)


def classify_diseases_UKBB(_sr_diseases):

    d_classify = {

        "AutoImm": [
            "Grave's disease",
            'Hyperthyroidism',
            'Hypothyroidism',
            'Type 1 diabetes',
            'Autoimmune hepatitis',

            'Ulcerative colitis',
            'Psoriasis vulgaris',
            'Rheumatoid arthritis',
            'Systemic lupus erythematosus',
            "Sjogren's syndrome",

            'IgA nephritis',
        ],

        "Imm": [
            'Sarcoidosis',
            'Iritis',
            'Uveitis',
            'Asthma',
            'Nasal polyp',

            'Pediatric asthma',
            'Chronic glomerulonephritis',
            'Nephrotic syndrome',
        ],

        "Others": [
            'Iron deficiency anemia',
            'Malignant lymphoma',
            'Prostate cancer',
            'Skin cancer',
            'Type 2 diabetes',
        ],
    }

    def get_class(_x, _d):

        for _class, _l in _d.items():

            if _x in _l:
                return _class
    

        return "NA"
    
    return _sr_diseases.map(lambda x: get_class(x, d_classify)).rename("Class")



def classify_diseases_BBJ(_sr_diseases):

    d_classify = {

        "AutoImm": [
            'Hyperthyroidism',
            "Grave's disease",
            "Hashimoto's disease",
            'Psoriasis vulgaris',
            'Rheumatoid arthritis',

            "Sjogren's syndrome",
        ],

        "Imm": [
            'Pulmonary tuberculosis',
            'Chronic hepatitis B',
            'Chronic hepatitis C',
            'Sarcoidosis',
            'Pneumonia',
            'Pollinosis',
            'Asthma',
            'Atopic dermatitis',
        ],

        "Others": [
            'Gastric cancer',
            'Hepatic cancer',
            'Lung cancer',
            'Cervical cancer',
            'Type 2 diabetes',
            'Angina pectoris',
            'Unstable angina pectoris',
            'Stable angina pectoris',
            'Myocardial infarction',
            'Cirrhosis',
        ],
    }

    def get_class(_x, _d):

        for _class, _l in _d.items():

            if _x in _l:
                return _class
    

        return "NA"
    
    return _sr_diseases.map(lambda x: get_class(x, d_classify)).rename("Class")




def get_performance_eval_table_v2(_df_ssfn, _df_HLAfm, _col_ToUse, _PPtype_ToUse, _ethnicity="EUR"):

    ## SPA vs. NoSPA하고 나서 용.

    if _PPtype_ToUse not in [".AA+HLA.PP", ".AA.PP", ".HLA.PP", ".HLAtype.PP", ".SNP.PP", ".intraSNP.PP", ".whole.PP"]:
        raise ValueError("Error! Pick one from the below!:\n{}".format(
            [".AA+HLA.PP", ".AA.PP", ".HLA.PP", ".HLAtype.PP", ".SNP.PP", ".intraSNP.PP", ".whole.PP"]
        ))

    if _col_ToUse not in _df_ssfn:
        raise ValueError(f"No '{_col_ToUse}' in the given SSFN file!")

    ##### (0) load data

    ### (0-1) 편의상 round 1만 남기기. 일단 당장은 top을 잘 맞췆는지 부터 봐야 함.
    df_HLAfm_R1 = _df_HLAfm[_df_HLAfm['Round'] == 1]

    df_ToScore = _df_ssfn.merge(df_HLAfm_R1.drop(['Category'], axis=1).rename({"Trait": "phe_name_ss"}, axis=1))
    # ssfn으로 left_join함.

    print(df_HLAfm_R1.head(3))
    print(df_HLAfm_R1.shape)

    print(df_ToScore.head(3))
    print(df_ToScore.shape)
    print(df_ToScore.columns.tolist())

    ##### (1) Main iteration (find the answer marker in the PP file)

    d_out_SCORE = {}

    for _index, _sr_row in df_ToScore.iterrows():

        _phe_name_ss = _sr_row['phe_name_ss']
        _fpath_PP = (_sr_row[_col_ToUse] + _PPtype_ToUse)

        _ANSWER_SNP2HLA = _sr_row['ANSWER_SNP2HLA']
        _MarkerType = _sr_row['MarkerType']
        _HLAgene = _sr_row['HLAgene']
        _pos = _sr_row['pos']
        _Allele_Residue = _sr_row['Allele/Residue']

        # print(_ANSWER_SNP2HLA)
        # print(_MarkerType)

        print(f"=====[{_index}]: {_phe_name_ss} / {_ANSWER_SNP2HLA}")

        try:
            _df_PP = pd.read_csv(_fpath_PP, sep='\t', header=0)
            # print(_df_PP.head(3))
        except FileNotFoundError:
            d_out_SCORE[_phe_name_ss] = None
            continue
        else:
            df_Found = find_item(_index, _phe_name_ss, _df_PP, _ANSWER_SNP2HLA, _MarkerType, _HLAgene, _pos,
                                 _Allele_Residue)
            # display(df_Found)

            d_out_SCORE[_phe_name_ss] = df_Found.copy()

        # if _index >= 10: break

    df_RETURN = pd.concat(d_out_SCORE, names=['phe_name_ss', 'index']) \
                    .reset_index("phe_name_ss", drop=False) \
                    .reset_index("index", drop=True) \
                    .loc[:, ['phe_name_ss', 'SNP', 'PP', 'CredibleSet', 'rank', 'rank_p']]
    print(df_RETURN.columns.tolist())

    l_ToExtract = ['TraitType', 'category', 'phe_name_ss', 'phenostring', 'num_cases', 'num_controls', 'num_samples',
                   'HLA_Allele'] + \
                  ['Other', 'Effect', 'ImpRsq', 'N', 'Beta', 'SE', 'P', 'any_MHC_association'] + \
                  ["MarkerType", "HLAgene", "pos", "Allele/Residue"]
    df_RETURN = df_ToScore[l_ToExtract].rename({"HLA_Allele": "SNP_answer"}, axis=1) \
        .merge(df_RETURN, on='phe_name_ss')
    # display(df_temp)

    return df_RETURN




## Table_0 (`get_performance_eval_table`)에 conditionally work하도록 짜자.
def concat_Z_imputed(_df_Table_0, _df_ssfn, _col_ToUse, _ethnicity="EUR"):
    if _col_ToUse not in _df_ssfn:
        raise ValueError(f"No '{_col_ToUse}' in the given SSFN file!")

    ##### (0) load data

    ### (0-2) `df_ToScore` 만들기. (사실상 ToIter임
    if _ethnicity == "EUR":

        l_ToExtract = ['TraitType', 'Category', 'Phenotype', 'phe_name_ss', 'No. samples', 'No. cases', 'No. controls',
                       'ToSWCA', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'SUM2HLA_OUT']

        if _col_ToUse not in l_ToExtract: l_ToExtract += [_col_ToUse]

        df_ssfn_2 = _df_ssfn[l_ToExtract]

    elif _ethnicity == "EAS":
        """
        ['TraitType', 'phe_name_dir', 'phe_name_ss(phenocode)', 'phenostring', 'isBinary', 'BBJ_dir', 'num_cases', 'num_controls', 'num_samples', 'category', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'ToSWCA', 'SUM2HLA_OUT']        
        """
        l_ToExtract = ['TraitType', 'category', 'phe_name_ss(phenocode)', 'num_cases', 'num_controls', 'num_samples',
                       'ToSWCA', 'fpath_ss_SUM2HLA', 'M_ss_SUM2HLA', 'SUM2HLA_OUT']

        if _col_ToUse not in l_ToExtract: l_ToExtract += [_col_ToUse]

        df_ssfn_2 = _df_ssfn[l_ToExtract] \
            .rename(
            {"category": "Category", 'phe_name_ss(phenocode)': "phe_name_ss", 'num_cases': 'No. cases',
             'num_controls': 'No. controls', 'num_samples': 'No. samples'}, axis=1
        )
    else:
        print("Wrong _ethnicity!")
        return -1

        ##### Main

    ## df_Table_0에 left_join시키기 => 필요한 만큼만 Z_imputed파일을 load
    df_ToIter = _df_Table_0[['phe_name_ss', 'SNP']].merge(
        df_ssfn_2[['phe_name_ss', _col_ToUse]], on='phe_name_ss'
    )
    print(df_ToIter)

    d_ToConcat = {}

    for _index, _phe_name_ss, _SNP, _SUM2HLA_OUT_recalc in df_ToIter.itertuples():
        print(f"=====[{_index}]: {_phe_name_ss} / {_SNP}")

        df_Z_imputed = pd.read_csv(_SUM2HLA_OUT_recalc + ".Z_imputed", sep='\t', header=0) \
            .set_index("SNP", drop=False)
        # display(df_Z_imputed)

        try:
            df_ToFind = df_Z_imputed.loc[[_SNP], :].rename(
                {"SNP": "SNP_Z_imp", "BETA": "BETA_Z_imp", "SE": "SE_Z_imp", "P": "P_Z_imp"}, axis=1)
            # display(df_ToFind)

        except Exception as e:
            print(e)
            print(f"Skipping {_phe_name_ss}!")
            continue

        d_ToConcat[_phe_name_ss] = df_ToFind.copy()

        # if _index >= 3: break

    df_ToConcat = pd.concat(d_ToConcat, names=['phe_name_ss', 'index']) \
        .reset_index(['index'], drop=True) \
        .reset_index(['phe_name_ss'], drop=False)


    print(df_ToConcat)

    df_RETURN = _df_Table_0.merge(df_ToConcat, on='phe_name_ss', how='left')
    print(df_RETURN.shape)

    return df_RETURN