"""
- a marker와 this marker에 clumping된 markers들 간의 r and r2값을 계산하는 modules들
- 현재 SWCA가 output dictionary를 return하면, 이 dictionary를 transform하는 형태로 작업할거임.

"""

import os, sys, re
from os.path import dirname, basename, join, exists
import json
import numpy as np
import pandas as pd

import subprocess


##### (Derepcated in 2026.01.05.) listdir()했을 때 ROUND_1 -> 2 -> 3 -> 10 ... 이런식으로 sorting되어 주어질거라는 보장이 없음 => 추후 함수에서 order mismatch발생.
# def get_SWCA_Each_Round_Clumped_Files(_out_dir_SWCA):
    
#     p_clumped_file = re.compile(r'\.ROUND_(\d+)\..*\.clumped$')
    
#     l_files = os.listdir(_out_dir_SWCA)
#     l_files_clumped = [_ for _ in l_files if bool(p_clumped_file.search(_))]
#     l_files_round = [p_clumped_file.search(_).group(1) for _ in l_files_clumped]

#     # display(l_files_clumped)
#     # display(l_files_round)
    
#     # d_RETURN = {int(_ROUND): _fpath for _ROUND, _fpath in zip(l_files_round, map(lambda x: join(_out_dir_SWCA, x), l_files_clumped))}    
#     d_RETURN = {f"ROUND_{_ROUND}": _fpath for _ROUND, _fpath in zip(l_files_round, map(lambda x: join(_out_dir_SWCA, x), l_files_clumped))}

#     return d_RETURN


def get_SWCA_Each_Round_Clumped_Files(_out_dir_SWCA):
    
    p_clumped_file = re.compile(r'\.ROUND_(\d+)\..*\.clumped$')
    
    l_files = os.listdir(_out_dir_SWCA)
    
    # 1. Clumped file filtering
    l_files_clumped = [_ for _ in l_files if bool(p_clumped_file.search(_))]
    
    # 2. [수정됨] Round 번호(int) 기준으로 정렬 (Sort by Round Number)
    # 문자열 정렬이 아닌 숫자 크기 순 정렬을 보장합니다.
    l_files_clumped.sort(key=lambda x: int(p_clumped_file.search(x).group(1)))
    
    # 3. Round Key 추출
    l_files_round = [p_clumped_file.search(_).group(1) for _ in l_files_clumped]

    # 4. Dictionary 생성
    d_RETURN = {f"ROUND_{_ROUND}": join(_out_dir_SWCA, _x) for _ROUND, _x in zip(l_files_round, l_files_clumped)}

    return d_RETURN



def get_clumped_markers_of_a_marker(_marker, _df_clumped, _f_sort=False): # deprecated

    ##### _index로 쓸 marker 찾기
    f_is_index_SNP = _marker in _df_clumped['SNP'].values

    if f_is_index_SNP:

        ##### 별말 없으면 top item만 챙기자.
        _df_clumped_2 = _df_clumped.set_index("SNP")
        
        _clumped_markers = _df_clumped_2.loc[_marker, "SP2"]
        
        l_clumped_markers = [re.sub(r'\(\d+\)', '', _) for _ in _clumped_markers.split(',')]

        # display(_top_marker)
        # display(_clumped_markers)
        # display(l_clumped_markers)

        d_RETURN = {_marker: list(sorted(l_clumped_markers)) if _f_sort else l_clumped_markers}

    else:

        sr_in_SP = _df_clumped['SP'].map(lambda x: re.search(_marker, x))
        f_in_SP = sr_in_SP.map(lambda x: bool(x))



    
    return d_RETURN



def get_clumped_markers_of_a_marker_v2(_marker, _df_clumped, _f_sort=False):

    sr_SNP = _df_clumped['SNP']
    sr_SP2 = _df_clumped['SP2'].map(lambda x: [re.sub(r'\(\d+\)', '', _) for _ in x.split(',')])
    # print(sr_SP2)


    f_in_SNP = sr_SNP.map(lambda x: x == _marker) # signal이 index SNP으로 있는 경우
    f_in_SP = sr_SP2.map(lambda x: _marker in x)  # signal이 'SP2' column에 있는 경우

    if f_in_SNP.any():

        d_RETURN = {sr_SNP.loc[f_in_SNP].iat[0]: sr_SP2.loc[f_in_SNP].iat[0]}

    elif f_in_SP.any():

        index_SNP = sr_SNP.loc[f_in_SP].iat[0]
        l_temp = [_ if _ != _marker else index_SNP for _ in sr_SP2.loc[f_in_SP].iat[0]] # `_marker`만 `index_SNP`으로 교체
        
        d_RETURN = {_marker: l_temp}
    else:
        d_RETURN = {}
        # d_RETURN = {"Wrong": ["Wrong"]}
        # raise Exception("Wrong!")
    


    return d_RETURN



def calc_r_r2(_top_SNP, _fpath_ref_bfile, _out_prefix, _plink="/home/wschoi/miniconda3/bin/plink"):    
        
    ##### (1) r 계산.
    
    out_prefix_r = _out_prefix + ".r"
    
    cmd = [
        _plink,
        "--r",
        "--keep-allele-order", # critical for the sign of r.
        "--bfile", _fpath_ref_bfile,
        "--ld-snp", _top_SNP,
        # "--ld-window", "99999", "--ld-window-kb", "250", "--ld-window-r2", "0.5", # clumping에서와 똑같은 default parameter로 통일.
        "--ld-window", "99999", "--ld-window-kb", "250", "--ld-window-r2", "0.2", # 어떤 이유에서인지 r2 threshold를 좀 더 널널하게 줘야 함.
        "--out", out_prefix_r,
        "--allow-no-sex", 
    ]
    # print(' '.join(cmd))

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    except subprocess.CalledProcessError as e:
        print(json.dumps(cmd, indent='\t'))
        raise e    
    
    
    ##### (1) r2 계산.
    
    out_prefix_r2 = _out_prefix + ".r2"

    cmd = [
        _plink,
        "--r2",
        "--keep-allele-order", # 그냥 r계산할때와 똑같이 넣자.
        "--bfile", _fpath_ref_bfile,
        "--ld-snp", _top_SNP,
        # "--ld-window", "99999", "--ld-window-kb", "250", "--ld-window-r2", "0.5", # clumping에서와 똑같은 default parameter로 통일.
        "--ld-window", "99999", "--ld-window-kb", "250", "--ld-window-r2", "0.2", # 어떤 이유에서인지 r2 threshold를 좀 더 널널하게 줘야 함. 심지어 r2값 조차도.
        "--out", out_prefix_r2,
        "--allow-no-sex", 
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    except subprocess.CalledProcessError as e:
        print(json.dumps(cmd, indent='\t'))
        raise e    
    
    

    return out_prefix_r + ".ld", out_prefix_r2 + ".ld"



def __MAIN__(_fpath_SWCA_out_dict, _out_dir_clumped, _fpath_ref_bfile, _f_old=False, _plink="/home/wschoi/miniconda3/bin/plink"):
    
    ##### (0) load data
    
    ### (0-1) SWCA output dictionary
    if isinstance(_fpath_SWCA_out_dict, str):
        with open(_fpath_SWCA_out_dict, 'r') as f_dict:
            d_SWCA_out = json.load(f_dict)
    elif isinstance(_fpath_SWCA_out_dict, dict):
        d_SWCA_out = _fpath_SWCA_out_dict.copy()
    else:
        raise Exception("Wrong dictionary!")

    ### (0-2) Clumped file dictionary
    d_clumped_files = get_SWCA_Each_Round_Clumped_Files(_out_dir_clumped)
        
        
    if _f_old:
        ### (2025.06.04.) 당장은 이렇게 예외 전처리
        d_SWCA_out = {k: v for i, (k, v) in enumerate(d_SWCA_out.items()) if i > 0} # Round 1은 버리기
        d_clumped_files = {(k+1): v for k, v in d_clumped_files.items()} # key값만 +1
            # 나중에 main pipeline에서 여기 알아서 해줘야 함.
        

    # print(json.dumps(d_SWCA_out, indent=4))
    # print(json.dumps(d_clumped_files, indent=4))

    """
    - zip은 쳐야하는게 맞는 것 같음. (length 찐빠나는거 예외처리)
    - 여기, 차원이 너무 많아서 functional하게 좀 짜봐.
    """

    # Round 1 제외한 Dictionary 생성
    d_SWCA_out_2 = {k: v for i, (k, v) in enumerate(d_SWCA_out.items()) if i > 0}


    ##### [수정됨] step1: zip 제거 및 Key 기반 접근 (Robust Iteration) (2026.01.05.)
    d_RETURN_step1 = {}
    
    # zip(items, items) 대신 d_SWCA_out_2를 기준으로 순회
    for i, (_ROUND_N, _l_markers) in enumerate(d_SWCA_out_2.items()):

        # Key 매칭 확인 (Safety Check)
        if _ROUND_N not in d_clumped_files:
            print(f"[WARNING] Clumped file for {_ROUND_N} not found. Skipping...")
            continue

        # Key를 이용해 정확한 파일 경로 가져오기
        _fpath_clumped = d_clumped_files[_ROUND_N]

        # print(f"=====[{i}] {_ROUND_N} / {_l_markers} / {_fpath_clumped}")
                
        df_temp_clumped = pd.read_csv(_fpath_clumped, sep=r'\s+', header=0)
        # print(df_temp_clumped.head(5))

        d_temp = {}
        for j, _marker in enumerate(_l_markers):
            d_clumped_markers = get_clumped_markers_of_a_marker_v2(_marker, df_temp_clumped)
            d_temp.update(d_clumped_markers)

        d_RETURN_step1[_ROUND_N] = d_temp.copy()

    # print(json.dumps(d_RETURN_step1, indent=4))


    ##### step2 / logic이 너무 복잡해져서 한번 끊고.

    d_RETURN_step2 = {}

    for i, (_ROUND_N, _d_clumped) in enumerate(d_RETURN_step1.items()):

        # print(f"=====[{i}] {_ROUND_N} / {_d_clumped.keys()}")

        d_temp = {}

        for j, (_marker, _l_clumped) in enumerate(_d_clumped.items()):

            out_r, out_r2 = calc_r_r2(_marker, _fpath_ref_bfile, d_clumped_files[_ROUND_N], _plink=_plink)

            df_r = pd.read_csv(out_r, sep=r'\s+', header=0, usecols=['SNP_A', 'SNP_B', 'R'])
            df_r2 = pd.read_csv(out_r2, sep=r'\s+', header=0, usecols=['SNP_A', 'SNP_B', 'R2'])

            d_r = {(_SNP_A, _SNP_B): _R for _index, _SNP_A, _SNP_B, _R in df_r.itertuples()}
            d_r2 = {(_SNP_A, _SNP_B): _R2 for _index, _SNP_A, _SNP_B, _R2 in df_r2.itertuples()}


            l_r = [d_r[(_marker, _)] if (_marker, _) in d_r else np.nan for _ in _l_clumped]
            l_r2 = [d_r2[(_marker, _)] if (_marker, _) in d_r2 else np.nan for _ in _l_clumped]

            # l_RETURN = [f"{_} (r:{_r},r2:{_r2})" for _, _r, _r2 in zip(_l_clumped, l_r, l_r2)] # as string
            # l_RETURN = [(_, _r, _r2) for _, _r, _r2 in zip(_l_clumped, l_r, l_r2)] # as tuple
            # l_RETURN = [(_, {"r": _r, "r2": _r2}) for _, _r, _r2 in zip(_l_clumped, l_r, l_r2)] # as dictionary
            l_RETURN = {_: {"r": _r, "r2": _r2} for _, _r, _r2 in zip(_l_clumped, l_r, l_r2)}

            d_temp[_marker] = l_RETURN


        d_RETURN_step2[_ROUND_N] = d_temp.copy()


    ##### step 3

    d_RETURN_step1_2 = {"ROUND_1": d_SWCA_out["ROUND_1"]}
    d_RETURN_step2_2 = {"ROUND_1": d_SWCA_out["ROUND_1"]}

    d_RETURN_step1_2.update(d_RETURN_step1)
    d_RETURN_step2_2.update(d_RETURN_step2)

    """
    - ROUND_1은 clumping result를 참조할 수 없음. 최초에 SNPs들만 clumping되어 만들어내는 결과이기 때문에.
    - ROUND_1도 clumping r2값을 참조해서 채점할까 했는데, 애초에 불가능함.
        - 이걸 알면 채점함수를 좀 더 단순하게 짜도 됨.
    
    """


    return d_RETURN_step1_2, d_RETURN_step2_2