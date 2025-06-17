"""
- hCAVIAR를 돌리기 위한 핵심 전처리 함수들.
    - Practically, 매우 중요함. 
    - 이게 제대로 안되면 mean-/cov-model 둘 다 오작동함.
    - (ex)
        - A GWAS summary - LDmatrix 간 BP를 바탕으로 SNP set을 match시키기.
        - LDtect region별로 자르기.


- (backlink)
    - 20250109_prototype_v2.ipynb
    
    
- 큰 흐름:
    - (0) raw summary의 'Z' column 준비 && effect allele column 'A1'으로 rename.
        - 얘는 알아서 해서 주어진다 가정
        - 20250202_BBJ_autoimm.ipynb
    - (1) match: a GWAS summary vs. the curated LD panel (-> '*.sumstats2')
        - the summary는 모두 hg19로 준비.
        - `match_eff_direc_GWASsummary()`
            - 이 함수를 call하는게 항상 main 시작임.
        - the curated LD panels들은 다음의 directory에 있음.
            - "LD_from_HLA_reference_panel/"
    - (2) PLINK로 clumping하기.
        - (1)의 output을 input으로 PLINK clumping 수행.
    - (3) `prepare_inputs()` (-> '*.sumstats3')
        - subset: the clumped SNPset으로.
        - subset: LDtect range로.
        - the summary and curated LD panel 둘 다 LD tect로 자름.
            - {'sumstats': "...", 'ld': "..."} 이렇게 준비되면 이제 hCAVAIR LL 계산을 하면 됨.

"""

import os, sys, re
import numpy as np
import scipy as sp
import pandas as pd
import math
import json

import subprocess
from shutil import which

from datetime import datetime

from src.Util import is_HLA_locus, run_PLINK_clump




########## [1] 임의의 GWAS summary를 내가 effect direction 맞춰놓은 LD panel에 match시키는 함수

## raw summary => the matched summary
def match_eff_direc_GWASsummary(_df_GWASsummary, _df_bim_LD_curated):
    
    
    """
    - `_df_GWASsummary`는 다음 3개의 columns들이 required됨.
        - (1) 'Z', (2) 'A1' (effect), and (3) 'A2' (non-effect)
    - 이 프레임 형태로 알아서 전처리하고 이 함수에 집어넣을 것.
    
    """
    
    ### (1) Remove ambiguous SNPs
    sr_allele_set = _df_GWASsummary[['A1', 'A2']].apply(lambda x: set(x), axis=1).rename("Allele_set")
    f_ambig = sr_allele_set.map(lambda x: x == {"A", "T"} or x == {"G", "C"}).rename("f_ambig")
    
    df_GWASsummary_2 = _df_GWASsummary[~f_ambig]
    
    # print(_df_GWASsummary.shape)
    # print(df_GWASsummary_2)
    
    ### (2) (BP, set(['A1', 'A2']))
    def complement(_base):
        if _base == "A": return "T"
        if _base == "C": return "G"
        if _base == "G": return "C"
        if _base == "T": return "A"
        return "-1"
    
    sr_A1_A2 = df_GWASsummary_2[['A1', 'A2']].apply(lambda x: set(x), axis=1).rename("Allele_set")
    sr_A1_A2_flip = df_GWASsummary_2[['A1', 'A2']] \
                        .map(lambda x: complement(x)) \
                        .apply(lambda x: set(x), axis=1) \
                        .rename("Allele_set_flip")
     # applymap에서 map으로 (2025.03.09.)
    
    sr_BP_A1_A2 = pd.concat([df_GWASsummary_2['BP'], sr_A1_A2], axis=1) \
        .apply(lambda x: tuple(x), axis=1)
        
    sr_BP_A1_A2_flip = pd.concat([df_GWASsummary_2['BP'], sr_A1_A2_flip], axis=1) \
        .apply(lambda x: tuple(x), axis=1).rename("BP_A1_A2_flip")
    
    ## LD 
    sr_BP_A1_A2_LD = pd.concat(
        [
            _df_bim_LD_curated['BP'], 
            _df_bim_LD_curated[['A1', 'A2']].apply(lambda x: set(x), axis=1)
        ], 
        axis=1
    ) \
        .apply(lambda x: tuple(x), axis=1).rename("LD_BP_A1_A2")
    
    # display(sr_BP_A1_A2)
    # display(sr_BP_A1_A2_flip)
    # display(sr_BP_A1_A2_LD)
    
    
    f_BP_A1_A2 = sr_BP_A1_A2.isin(sr_BP_A1_A2_LD)
    f_BP_A1_A2_flip = sr_BP_A1_A2_flip.isin(sr_BP_A1_A2_LD) # flip한게 LD에 있음.
    f_isin = f_BP_A1_A2 | f_BP_A1_A2_flip
    
    f_isin_LD = sr_BP_A1_A2_LD.isin(sr_BP_A1_A2) | sr_BP_A1_A2_LD.isin(sr_BP_A1_A2_flip)
    
    df_GWASsummary_3 = pd.concat([df_GWASsummary_2, f_BP_A1_A2_flip.rename("ToFlip")], axis=1).loc[f_isin, :]
    df_bim_LD_curated_2 = _df_bim_LD_curated[f_isin_LD]
    
    
    # print("\nThe raw summary (without ambig. SNPs) to inner_join:")
    # print(df_GWASsummary_3)
    # print("\nLD (only SNPs) to inner_join:")
    # print(df_bim_LD_curated_2)
    
    if df_GWASsummary_3.shape[0] != df_bim_LD_curated_2.shape[0]:

        ## 더이상 duplicated되는 애들은 없다 가정
        ## Duplicated BPs들이 있으면 들어오는 block임.
        
        # print(df_bim_LD_curated_2['BP'].duplicated().any())
        # display(df_bim_LD_curated_2[ df_bim_LD_curated_2['BP'] == 30313268 ])
        
        print("BP duplicated: ", df_GWASsummary_2['BP'].duplicated().any())

        raise ValueError("There is BP-duplicated marker in the GWAS summary!")


    
    ### (3) merge 준비
    
    df_temp = pd.DataFrame(
        [[complement(_A1), complement(_A2)] if _flag else [_A1, _A2] \
             for _index, _A1, _A2, _flag in df_GWASsummary_3[['A1', 'A2', 'ToFlip']].itertuples()],
        index=df_GWASsummary_3.index,
        columns=['A1_GWAS_flip', 'A2_GWAS_flip']
    )
    
    df_GWASsummary_3 = pd.concat([df_GWASsummary_3, df_temp], axis=1)
    # display(df_GWASsummary_3)
    
    
    df_mg0 = df_GWASsummary_3.merge(
        _df_bim_LD_curated.drop(['GD'], axis=1),
        on=['CHR', 'BP'],
        suffixes=('_GWAS', '_LD')
    )
    # display(df_mg0)
    
    
    ### (4) fix the 'Z'
    f_ToFix = df_mg0['A1_GWAS_flip'] != df_mg0['A1_LD']
    
    sr_Z_fixed = (df_mg0['Z'] * f_ToFix.map(lambda x: -1.0 if x else 1.0)).rename("Z_fixed")
    
    df_RETURN = pd.concat([df_mg0, sr_Z_fixed], axis=1)
    
    return df_RETURN


"""
- (example)
### RA_Yuki (EUR) + T1DGC
df_eff_direc_panel_RA_EUR = match_eff_direc_GWASsummary(df_Yuki_RA_EUR, df_bim_T1DGC)
display(df_eff_direc_panel_RA_EUR)

df_eff_direc_panel_RA_EUR.to_csv(
    "20250109_prototype_v2/RA_Yuki_EUR+T1DGC.hg19.M4988.sumstats2",
    sep='\t', header=True, index=False, na_rep="NA"
)
"""





########## [2] The matched summary를 subset하기.


### [2-1] 우선 clumped set으로 subset해야함. (그다음 [2-2]의 HLA region으로 )
def subset_matched_ss(_fPath_ss_matched, _fPath_clumped) -> pd.DataFrame:
    
    df_ss_matched_RA = pd.read_csv(_fPath_ss_matched, sep='\t', header=0)
    df_clumpped_RA = pd.read_csv(_fPath_clumped, sep=r'\s+', header=0)

    # display(df_ss_matched_RA)
    # display(df_clumpped_RA)

    f_isin = df_ss_matched_RA['BP'].isin(df_clumpped_RA['BP'])
    # display(df_ss_matched_RA[f_isin])
    
    return df_ss_matched_RA[f_isin]



### [2-2] 그 다음 LDtect HLA sub-region으로 나눠야 함.

d_LDtect_HLAregion_hg19 = {
    "HLAregion1": (28917608, 29737971),
    "HLAregion2": (29737971, 30798168),
    "HLAregion3": (30798168, 31571218),
    "HLAregion4": (31571218, 32682664),
    "HLAregion5": (32682664, 33236497),
    "HLAregion6": (33236497, 35455756)
}

d_LDtect_HLAregion_hg19_2 = {
    # "whole": (29_000_000, 34_000_000),
    "whole": (28_000_000, 35_000_000), # 뭐던동 reference data의 BP range를 cover할 수 있도록.
    # "HLAregion2": (29737971, 30798168),
    # "HLAregion3": (30798168, 31571218),
    # "HLAregion4": (31571218, 32682664),
    # "HLAregion5": (32682664, 33236497), # 당분간은 'whole'만 (2025.03.07.)
}

def split_summary_byLDtect(_df_matched, _d_LDtect=d_LDtect_HLAregion_hg19_2):

    """
    - 얘는 더 이상 쓸모가 없음. (LD를 region별로 나누는 것보다도 쓸모가 없음.)
    - 뭐던동 `d_LDtect_HLAregion_hg19_2`의 'whole'은 reference data의 BP range를 cover할 수 있어야 함.
    - (1) reference marker set만 생각하면 됨, GWAS summary의 marker set은 결국 reference data와 share하는 만큼만 남으니까.
    - (2) HAN은 28mb~이고 GWAS summary도 28mb~인데, 여기서 29mb~로 잘라버리면 28~29mb의 SNPs들이 소실됨.
    - (3) 그 와중에 여기를 제외하면 모두 주어진대로 28mb~로 작업함. => down-stream에서 차원 mismatch발생.


    """
    
    d_RETURN = {}
    
    for i, (_label, (_bp_s, _bp_e)) in enumerate(_d_LDtect.items()):
        
        # print("\n===[{}]: {} / {} - {}".format(i, _label, _bp_s, _bp_e))
        
        f_BP = (_df_matched['BP'] >= _bp_s) & (_df_matched['BP'] < _bp_e)
        
        df_temp = _df_matched[f_BP]
        # display(df_temp)
        # print(df_temp.shape)
                
        d_RETURN[_label] = df_temp.copy()
    
    return d_RETURN





########## [2] LD matrix load관련.

## 내가 the cureated LD panels들은 미리 HLA region별로 잘라놨기 때문에,
## 저 이미 잘려있는 LD panel에서 clumped SNP set만 subset하면 됨.
def subset_LDpanel_SNPs(_d_fPath_LD, _fpath_ss, _col_SNP_ss="SNP_LD"):
    
    d_out_HLAregion = {}
    
    
    ##### (0) load

    ### (0-1) ss (or clumping result) <= 결국 SNP set column이 필요함.
    df_ss = pd.read_csv(_fpath_ss, sep=r'\s+', header=0)
    # display(df_ss)
    
    
    for i, (_hla_region, _fPath_LD) in enumerate(_d_fPath_LD.items()):
        
        # print("\n===[{}]: {} ".format(i, _hla_region))
        
        ### (0-1) LD
        df_LD = pd.read_csv(_fPath_LD, sep='\t', header=0)
        df_LD.index = df_LD.columns

        sr_SNP_LD = df_LD.columns.to_series().reset_index(drop=True).rename("SNP_inLD")

        # display(df_LD)
        # display(sr_SNP_LD)
        
        
        ##### (1) subset just SNPs.

        def subset_just_SNP(_sr_SNP_LD, _sr_SNP_ss):

            ### (1) _sr_SNP_LD를 SNP + HLA 로 disjoint하게 나눠야 함.

            f_AA = _sr_SNP_LD.str.startswith("AA")
            f_SNP_intra = _sr_SNP_LD.map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x))
            f_HLA = _sr_SNP_LD.str.startswith("HLA")
            f_INS = _sr_SNP_LD.str.startswith("INS_")

            f_HLAs = f_AA | f_SNP_intra | f_HLA | f_INS


            sr_SNP_markers = _sr_SNP_LD.loc[~f_HLAs] # SNP
            sr_HLA_markers = _sr_SNP_LD.loc[f_HLAs] # HLA 



            ### (2) sr_SNP_markers
            f_isin = sr_SNP_markers.isin(_sr_SNP_ss)
            sr_SNP_markers_2 = sr_SNP_markers.loc[f_isin]
            # display(sr_SNP_markers_2)

            """
            => sr_SNP_markers_2 = sr_SNP_markers.loc[f_isin]

            - sr_SNP_markers_2 = _sr_SNP_ss 그냥 이렇게 해도 문제 없음, 앞서 match_eff_direc_GWASsummary() 함수를 call하기 때문에.
            - 혹시 몰라서 그냥 이렇게 함.
            - 문제없다면 저렇게 하면 그냥 _sr_SNP_ss 이 나오게 됨.

            """


            sr_RETURN = pd.concat([sr_SNP_markers_2, sr_HLA_markers]).sort_index()

            return sr_RETURN
    

        sr_subset = subset_just_SNP(sr_SNP_LD, df_ss[_col_SNP_ss])

        d_out_HLAregion[_hla_region] = df_LD.loc[sr_subset, sr_subset].copy()
        
        # if i >= 0: break
        
            
    return d_out_HLAregion



def prepare_inputs(_fPath_ss_matched:str, _fPath_clumped:str, _d_fPath_LDcurated:dict,
                   _col_SNP_ss="SNP_LD", _out_prefix_ss=None, _out_prefix_LD=None):
    
    ##### (1) subset the matched ss
    # print("\n##### (1) subset the matched ss")

    if _fPath_clumped == None: # 그냥 the matched ss만 load
        df_ss_subset = pd.read_csv(_fPath_ss_matched, sep='\t', header=0)
        
        ## 바로 다음 `subset_LDpanel_SNPs`에서 plug-in 편하게 하려고
        _fPath_clumped = _fPath_ss_matched
        _col_SNP_ss = "SNP_LD"
        
    else: # 
        df_ss_subset = subset_matched_ss(_fPath_ss_matched, _fPath_clumped)

    
    d_ss_HLAregions = split_summary_byLDtect(df_ss_subset)
    
    
    ##### (2) subset the LD curated.
    # print("\n##### (2) subset the LD curated.")
    
    d_LD_curated = subset_LDpanel_SNPs(_d_fPath_LDcurated, _fPath_clumped, _col_SNP_ss)
    
    
    if isinstance(_out_prefix_ss, str) and isinstance(_out_prefix_LD, str):
        
        d_RETURN = {}

        ##### (3) export
        # print("\n##### (3) export.")
        
        for i, _hla_region in enumerate(d_ss_HLAregions.keys()):
            
            # print("===[{}]: {}".format(i, _hla_region))
            
            df_ss_temp = d_ss_HLAregions[_hla_region]
            df_LD_temp = d_LD_curated[_hla_region]
            
            out_temp_ss = _out_prefix_ss + ".{}.M{}.sumstats3".format(_hla_region, df_ss_temp.shape[0])
            out_temp_LD = _out_prefix_LD + ".{}.M{}.ld".format(_hla_region, df_LD_temp.shape[0])
            
            # out_temp_LD = os.path.join(
            #     os.path.dirname(_out_prefix),
            #     re.sub(r'.ld$', '.M{}.ld'.format(df_LD_temp.shape[0]), os.path.basename(_d_fPath_LDcurated[_hla_region]))
            # )
            
            
            # display(df_ss_temp)
            # display(df_LD_temp)

            # print("\t{}".format(out_temp_ss))
            # print("\t{}".format(out_temp_LD))
            
            
            ### export
            df_ss_temp.to_csv(out_temp_ss, sep='\t', header=True, index=False, na_rep="NA")
            df_LD_temp.to_csv(out_temp_LD, sep='\t', header=True, index=False, na_rep="NA")
            
            d_RETURN[_hla_region] = {
                "sumstats": out_temp_ss,
                "ld": out_temp_LD
            }
            
            
            # if i >= 0: break
            
            
        return d_RETURN
    
    else:
    
        return d_ss_HLAregions, d_LD_curated

    
    
    
########## main wrapper ##########
def __MAIN__(_fpath_ss, _d_fpath_LD:dict, _fpath_LD_SNP_bim, _fpath_LD_SNP_HLA, 
             _out_prefix_ss, _out_prefix_LD,
             _f_do_clump = True,
             _plink = "~/miniconda3/bin/plink"):

    df_ss = pd.read_csv(_fpath_ss, sep='\t', header=0).rename({"STAT": "Z"}, axis=1)

    ## Error when there is no any signal.
    arr_P = 2 * sp.stats.norm.cdf( -df_ss['Z'].abs() )
    N_signals = np.count_nonzero(arr_P < 5e-8)
    if N_signals == 0:
        raise ValueError(f"There is no any signal in the GWAS summary! ({_fpath_ss})")

    
    df_ref_bim_SNP_HLA = pd.read_csv(
        _fpath_LD_SNP_HLA + ".bim", sep='\t', header=None,
        names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
    )

    f_is_HLA_locus = is_HLA_locus(df_ref_bim_SNP_HLA['SNP'])
    df_ref_bim_SNP = df_ref_bim_SNP_HLA[~f_is_HLA_locus]

    # df_ref_bim_SNP = pd.read_csv(
    #     _fpath_LD_SNP_bim, sep='\t', header=None,
    #     names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
    # )
    


    ##### (1) match: a GWAS summary vs. the curated LD panel (-> '*.sumstats2')
    
    df_matched = match_eff_direc_GWASsummary(df_ss, df_ref_bim_SNP)
    out_matched = _out_prefix_ss + ".sumstats2"
    # display(df_matched)

    ## export
    df_matched.to_csv(out_matched, sep='\t', header=True, index=False, na_rep="NA")

    

    ##### (2) PLINK로 clumping하기.
    if _f_do_clump:
        
        df_matched[['SNP_LD', 'P']].rename({"SNP_LD": "SNP"}, axis=1) \
            .to_csv(_out_prefix_ss + ".ToClump", sep='\t', header=True, index=False, na_rep="NA")
    
        out_ToClump = _out_prefix_ss + ".ToClump"
        out_clumped = run_PLINK_clump(out_ToClump, _fpath_LD_SNP_HLA, _out_prefix_ss, _plink)
        # print(out_clumped)

    else:
        print("No clumping!")
        out_ToClump = None
        out_clumped = None

    
    
    ##### (3) `prepare_inputs()` (-> '*.sumstats3')
    
    d_RETURN = prepare_inputs(
        out_matched, out_clumped, _d_fpath_LD,
        _out_prefix_ss=_out_prefix_ss, 
        _out_prefix_LD=_out_prefix_LD,
        _col_SNP_ss='SNP'
    )
    
    out_json = _out_prefix_ss+".hCAVIAR_input.json"
    
    with open(out_json, 'w') as f_json:
        json.dump(d_RETURN, f_json, indent='\t')


    return out_matched, out_ToClump, out_clumped, out_json