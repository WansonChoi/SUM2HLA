import os, sys, re
import numpy as np
import pandas as pd
import json
import subprocess

from datetime import datetime



def is_HLA_locus(_sr_markers) -> pd.Series:
    
    """
    - 아래 `subset_bim_SNP`에서 bim파일 단위로 작업하는 것보다, 
    - 좀 더 작게 Series-level에서 하는게 더 활용도가 좋은 것 같음.
    """
    
    f_AA = _sr_markers.str.startswith("AA")
    f_SNP_intra = _sr_markers.map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x))
    f_HLA = _sr_markers.str.startswith("HLA")
    f_INS = _sr_markers.str.startswith("INS_")

    f_HLAs = f_AA | f_SNP_intra | f_HLA | f_INS
    
    return f_HLAs



def subset_bim_SNP(_df_bim):

    # extract only SNPs
    f_AA = _df_bim['SNP'].str.startswith("AA")
    f_SNP_intra = _df_bim['SNP'].map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x))
    f_HLA = _df_bim['SNP'].str.startswith("HLA")
    f_INS = _df_bim['SNP'].str.startswith("INS_")

    f_HLAs = f_AA | f_SNP_intra | f_HLA | f_INS
    
    return _df_bim[~f_HLAs]



def make_psd(matrix):
    # 대칭화
    matrix = (matrix + matrix.T) / 2

    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 음수 고유값을 작은 양수로 대체
    eigenvalues[eigenvalues < 1e-10] = 1e-10

    # PSD 행렬 재구성
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T



def make_SUM2HLA_LDmatrix(_df_ld_raw, _df_bim_SNP_HLA):

    """
    - (2025.12.03.)
        - 얘도 아래 stream으로 처리하는거 준비하면서 다시 test했음. 완벽하게 replicaiton됨.
    
    """

    df_ld_NoNA = _df_ld_raw.fillna(0.0)

    for i in range(df_ld_NoNA.shape[0]):
        df_ld_NoNA.loc[i, i] = 1.0 # diagonal은 1.0으로.


    df_ld_NoNA_PSD = make_psd(df_ld_NoNA.values)
    df_ld_NoNA_PSD = pd.DataFrame(df_ld_NoNA_PSD, columns=_df_bim_SNP_HLA.iloc[:, 1], index=_df_bim_SNP_HLA.iloc[:, 1])

    return df_ld_NoNA_PSD



def wrapper_make_SUM2HLA_LDmatrix(_fpath_ld_raw, _fpath_bim_SNP_HLA, _out):

    print("Loading raw LD matrix for SUM2HLA.")
    df_LD_raw = pd.read_csv(_fpath_ld_raw, sep='\t', header=None)

    print("Loading bim file of the T1DGC reference panel.")
    df_bim = pd.read_csv(_fpath_bim_SNP_HLA, sep='\t', header=None, names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])


    print("Preprocessing the LD matrix.")
    df_LD_NoNA_PSD = make_SUM2HLA_LDmatrix(df_LD_raw, df_bim)


    df_LD_NoNA_PSD.to_csv(_out, sep='\t', header=True, index=False, na_rep="NA")

    return _out, df_LD_NoNA_PSD



def make_SUM2HLA_LDmatrix_stream(_fpath_LD_matrix_raw, _fpath_bim_SNP_HLA, _out, _nrows=None):

    """
    - (2025.12.03.)
        - "_ClusterPhes_v4/20241204_prepr_LDmatrix.ipynb 내에 `prepr_LD_matrix()` 함수를 살짝만 수정한거.
        - 완벽하게 replication되는거 이 날 재확인함.
        - 얘는 일단 일단 deprecate. 위에꺼가 좀 더 간단해서 얘를 우선 쓰기로 함.
    
    """
    
    df_bim = pd.read_csv(
        _fpath_bim_SNP_HLA, sep='\t', header=None, names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
    )
    print(df_bim)
    
    
    l_header = df_bim['SNP'].tolist()    
    print(l_header[:5])
    print(len(l_header))
    

    with open(_fpath_LD_matrix_raw, 'r') as f_LD_matrix, open(_out, 'w') as f_out:
        
        ### set header
        f_out.write('\t'.join(l_header) + "\n")
        
        
        for i, line in enumerate(f_LD_matrix):
            
            l_temp = list(map(lambda x: float(x), line.split()))
            
            if i % 1000 == 0:
                print("===[{}]: {}".format(i, l_temp[:10]))
            
            
            ### 값 수정
            iter_isna = np.isnan(l_temp)
            iter_fillna_zero = (0.0 if _f_isna else _ld_value for _f_isna, _ld_value in zip(iter_isna, l_temp))
            iter_fill_diagonal = (1.0 if i == j else _ for j, _ in enumerate(iter_fillna_zero))
                    
            f_out.write('\t'.join(list(map(lambda x: str(x), iter_fill_diagonal))) + "\n")

            if isinstance(_nrows, int) and i >= _nrows:
                break
            
            
    return _out



def is_psd(matrix):
    """
    - 당장 쓰지 않는 함수임.
    - cholesky decomposition으로 PSD를 더 빨리 check할 수 있는 idea. (by chat-gpt)
    """

    """ 빠른 PSD 체크: Cholesky 분해 이용 """

    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False



def get_N_of_ss(_ss):

    """
    return sample size ('N' column in the header) of the GWAS summary
    """

    df_ss_temp = pd.read_csv(_ss, sep=r'\s+', header=0, nrows=5)

    N_sample_size = df_ss_temp['N'].iat[0]

    """
    (cf) 맨 처음 argument check할 때 'N'이 column이 없으면 죽도록 만들었음. 안심하고 저렇게 쓰면 됨.
    """


    return N_sample_size



def mask_codingAA_and_HLA(_x):
    
    if _x.startswith("AA"):
        return bool(re.match(r"AA_(\w+)_(\d+)_", _x))
    
    if _x.startswith("HLA"):
        return True
    
    return False




##### 이하 deprecated.


def split_summary(_df_ss, _col_BP='BP') -> dict:
    
    d_LDtect_HLAregion_hg19_2 = {
        "whole": (29_000_000, 34_000_000),
        "HLAregion2": (29737971, 30798168),
        "HLAregion3": (30798168, 31571218),
        "HLAregion4": (31571218, 32682664),
        "HLAregion5": (32682664, 33236497),
    }
    
    d_out = {}
    
    for i, (_hla_region, (_bp_s, _bp_e)) in enumerate(d_LDtect_HLAregion_hg19_2.items()):
        
        f_BP = (_df_ss[_col_BP] >= _bp_s) & (_df_ss[_col_BP] < _bp_e)
        
        d_out[_hla_region] = _df_ss[f_BP]
    
    
    
    return d_out



def calc_concordance(_fPath_WTCCC_signals, _fPath_PP, _type_PP="whole", 
                     _l_thresh_top_rank=[0.01, 0.05, 0.10, 0.15, 0.20],
                    _col_SNP_PP='SNP', _col_SNP_WTCCC='SNP', _col_P_WTCCC='P'):
        
    print("\nCaculating Concordance. ({})".format(_type_PP))
    print("\nThe signal proportion within each HLA sub-region.")
    
    ##### (1) df_PP for each HLA region
    d_df_PP = {}
    
    with open(_fPath_PP, 'r') as f_PP:
        d_df_PP = json.load(f_PP)
        
    d_df_PP = {k: pd.read_csv(v[_type_PP], sep='\t', header=0) for k, v in d_df_PP.items()}
    
    # display(d_df_PP)
    
    
    
    ##### (2) df_WTCCC_signals for each HLA region (answer)
    df_WTCCC_signals = pd.read_csv(_fPath_WTCCC_signals, sep=r'\s+', header=0)
    df_WTCCC_signals = df_WTCCC_signals[ df_WTCCC_signals[_col_P_WTCCC] < 5e-8 ]
        
    d_df_WTCCC_signals = split_summary(df_WTCCC_signals)    
    # display(d_df_WTCCC_signals)
    
    
    if d_df_PP.keys() != d_df_WTCCC_signals.keys():
        print("Wrong keys")
        return -1
    
    
    d_df_count_and_prop = {}
    
    for i, _hla_region in enumerate(d_df_PP.keys()):
        
        print("\n===[{}]: {}".format(i, _hla_region))
        

        df_temp_WTCCC = d_df_WTCCC_signals[_hla_region]
        df_temp_PP = d_df_PP[_hla_region]
        
        
        d_temp_thresh = {}
        
        for j, _thresh_top_rank in enumerate(_l_thresh_top_rank):
            
            print("=====[{}]: threshold: {}".format(j, _thresh_top_rank))

            idx_top_rank = round( df_temp_PP.shape[0] * _thresh_top_rank ) # sorted LL 상위 5%까지의 row index
            df_temp_PP_threshold = df_temp_PP.iloc[:idx_top_rank, :]

            # display(df_temp_PP)
            # display(df_temp_WTCCC)



            set_markers_PP = set(df_temp_PP_threshold[_col_SNP_PP])
            set_markers_WTCCC = set(df_temp_WTCCC[_col_SNP_WTCCC])
            set_marekrs_intersect = set_markers_PP.intersection(set_markers_WTCCC)

            print("The top {}% LL markers in PP: {}".format(_thresh_top_rank*100, len(set_markers_PP)))
            print("The WTCCC signals: {}".format(len(set_markers_WTCCC)))
            print("The intersection: {}".format(len(set_marekrs_intersect)))        
            
            d_temp_thresh[_thresh_top_rank] = {
                "count_PP": len(set_markers_PP),
                "count_WTCCC": len(set_markers_WTCCC),
                "count_intersect": len(set_marekrs_intersect),
                "concordance": len(set_marekrs_intersect) / len(set_markers_PP)
            }
            
            
        d_df_count_and_prop[_hla_region] = pd.DataFrame.from_dict(d_temp_thresh, orient="index")
    
    
    df_RETURN = pd.concat(
        {_hla_region: _df['concordance'].rename(_hla_region) for _hla_region, _df in d_df_count_and_prop.items()},
        axis=1
    )
    df_RETURN.index = df_RETURN.index.astype('category')
    
    return d_df_count_and_prop, df_RETURN

