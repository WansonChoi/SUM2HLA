import os, sys, re
import pandas as pd
import numpy as np

import math

"""
- UKBB_WGS에 이어서 MVP도 좀 마저 받아볼게 있음.

"""



def make_outdir(_df, _fpath_out_dir=None):

    """
    - UKBB_WGS때처럼 output_dir을 fancy하게 못 만들겠음. 
        - 이거하다가 시간이 다 갈듯.
        - UKBB_WGS는 같은 group이 Open한거여서 운 좋게 이렇게 만들 수 있었던듯.

    - acc_id랑 efoTraits만 붙이자.

    """

    l_ToExtract = ['efoTraits', 'accessionId']

    sr_efoTraits = _df['efoTraits'].map(lambda x: x.replace(" ", "_"))

    sr_outdir = pd.Series(
        [f"MVP_{_efoTraits}_{_accID}" for _efoTraits, _accID in zip(sr_efoTraits, _df['accessionId'])],
        name='out_dir',
        index=_df.index
    )

    return sr_outdir



def make_wget_CMD(_df, _out):

    df_temp = _df[['accessionId', 'reportedTrait', 'efoTraits', 'discoverySampleAncestry']]
    sr_ftp = _df['summaryStatistics'].str.replace("http", "ftp")
    sr_out_dir = make_outdir(_df)

    df_temp = pd.concat([df_temp, sr_ftp, sr_out_dir], axis=1)

    CMD = \
        "wget -r -nH --cut-dirs=6 --no-parent --reject index.html* \\\n" + \
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

    df_RETURN = pd.concat([_df, sr_out_dir], axis=1)

    return _out, df_RETURN