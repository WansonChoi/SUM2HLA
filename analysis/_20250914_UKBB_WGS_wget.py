import os, sys, re
import pandas as pd
import numpy as np

import math

"""
- wget bash command 만들기 위한 steps들
- 이런 잔작업 + 잔작업의 구현과정은 ipynb에서 안할거임.

- 아래처럼 declarative하게 필요한 steps들은 함수로 만들고, ipython에서 running을 replicate하는식으로 만들면서 run할거임.
"""


def print_Hello():

    print("Hello")

    print("Hello")
    print("Hello")

    return 0


def prepr_GWAScatalog_export_df(_df_GWAScatalog): # deprecated

    """
    - 당장 'discoverySampleAncestry' column의 값이 어떤 정규표현식으로 주어질지 모름.
    - 저번에 UKBB_WGS할때야, 한 group에서 Open한거라 대충 짜서 처리 가능했음. 
    - 이걸 일반화 하려니 시간 아까움. 그냥 GWAScatalog에서 exported table은 주어진대로 써.
        - 그리고, 필요한 애들 다운 받고 나서 그 다음 sample sizes들을 정리하자거.
    
    """

    ## discoverySampleAncestry => N_total, Ethnicity
    p_N_total = re.compile(r'(\S+)\s+(\S+)')
    df_temp1 = _df_GWAScatalog['discoverySampleAncestry'].str.extract(p_N_total).rename({1: "N_total", 2: ""})

    ## initialSampleDescription => N_case, N_control

    return 0


def make_outdir(_df, _fpath_out_dir=None):

    ## 예상 out_dir이름: "UKBB_WGS_{efoTraits}_{Ethnicity}_{accessionId}"
    l_ToExtract = ["efoTraits", "Ethnicity", "accessionId"]

    sr_efoTraits2 = _df['efoTraits'] \
                        .str.replace("(", "") \
                        .str.replace(")", "") \
                        .str.replace(" ", "_").rename("efoTraits2")

    def map_Ethnicity(_x):

        """
        'African unspecified', 'Other', 'East Asian', 'European', 'South Asian'        
        """

        if _x == 'European': return "EUR"
        if _x == 'East Asian': return "EAS"
        if _x == 'South Asian': return "SAS"
        if _x == 'African unspecified': return "AFR"

        return _x


    sr_Ethnicity = _df['Ethnicity'].map(lambda x: map_Ethnicity(x)).rename("Ethnicity2")

    df_temp = pd.concat([sr_efoTraits2, sr_Ethnicity, _df['accessionId']], axis=1)

    sr_OUT_DIR = pd.Series(
        [f"UKBB_WGS_{_efoTrait}_{_Ethnicity}_{_accID}" for _index, _efoTrait, _Ethnicity, _accID in df_temp.itertuples()],
        name = "out_dir",
        index=_df.index
    )

    return sr_OUT_DIR



def make_wget_CMD(_df, _out):

    df_temp = _df[['accessionId', 'reportedTrait', 'efoTraits', 'discoverySampleAncestry']]
    sr_ftp = _df['summaryStatistics'].str.replace("http", "ftp")
    sr_out_dir = make_outdir(_df)

    df_temp = pd.concat([df_temp, sr_ftp, sr_out_dir], axis=1)

    CMD = \
        "wget -c -r -nH --cut-dirs=6 --no-parent --reject index.html* \\\n" + \
            '\t--exclude-directories={exclude_dir} \\\n' + \
            "\t{ftp} \\\n" + \
            "\t-P {out_dir}"

    ## 시간 없어서 "harmonized/"는 제외.
    ## 이거 제끼는거 시행착오가 존나 많았음;; 결론은 서버기준 절대경로로 하나씩 언급해야 함. '*'같은 정규표현식은 인지를 못함.

    with open(_out, 'w') as f_out:

        for _index, _accessionId, _reportedTrait, _efoTraits, _discoverySampleAncestry, _ftp, _out_dir in df_temp.itertuples():

            str_signpost = f"\n\n##### {_accessionId} / {_reportedTrait} / {_discoverySampleAncestry}"

            f_out.write(str_signpost + "\n")
            f_out.write(CMD.format(ftp=_ftp, out_dir=_out_dir, exclude_dir=_ftp.replace("ftp://ftp.ebi.ac.uk", "") + "/harmonised/"))


    return _out
