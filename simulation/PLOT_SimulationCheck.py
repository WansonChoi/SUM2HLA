import os, sys, re
import pandas as pd
import src.Util as Util

def concat_expected_means(_d_mean):

    """
    - 다음과 같이 생겼다 가정.

    _l_ncp_2nd = [6.0, 8.0, 10.0, 20.0, 30.0, 40.0]
    _out_dir = "/data02/wschoi/_hCAVIAR_v2/20250721_Simulation_input_v4/"

    d_BINS_0_0_0_1_HLA_C_0702_SSFN = {
        _ncp_2nd: join(_out_dir, f"BINS_0.0–0.1.AA_DRB1_11_32660115_SPG.ncp1_+50.0.HLA_C_0702.ncp2_{_ncp_2nd:+}.SSFN") for _ncp_2nd in _l_ncp_2nd
    }
    
    """
    
    def load_item(_fpath, _ncp_2nd):
        return pd.read_csv(_fpath, sep='\t', header=None, names=['SNP', _ncp_2nd], index_col=['SNP'])
    
    return pd.concat([load_item(_fpath, _ncp_2nd) for _ncp_2nd, _fpath in _d_mean.items()], axis=1)
    



def get_HLA_markertype_and_gene(_marker_label):
    
    p_HLA = re.compile(r'^HLA_(\w+)_\d+')
    p_AA = re.compile(r'AA_(\w+)_-?\d+_\d+')
    p_SNP = re.compile(r'SNP_(\w+)_\d+')
    
    m_HLA = p_HLA.match(_marker_label)
    m_AA = p_AA.match(_marker_label)
    m_SNP = p_SNP.match(_marker_label)
    
    if bool(m_HLA):
        return "HLA", m_HLA.group(1)
    elif bool(m_AA):
        return "AA", m_AA.group(1)
    elif bool(m_SNP):
        return "intraSNP", m_SNP.group(1)
    else:
        return "SNP", "intergenic"
        
    """
    - 다음의 instances들로 check함. 문제 없는듯.

    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("HLA_A_30"))
    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("HLA_A_3004"))
    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("HLA_DRB1_0437"))

    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("AA_B_298_31430910_x"))
    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("AA_B_296_31430916"))

    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("SNP_B_31430904"))
    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("SNP_B_31432179_A"))

    print(PLOT_SimulationCheck.get_HLA_markertype_and_gene("rs7766843"))
    
    """


def prepare_index(_df):
    
    df_index = _df.index.to_frame(index=False)
    # print(df_index)
    
    ### (1) SNP vs. HLA markers    
    sr_is_HLA_locus = Util.is_HLA_locus(df_index.iloc[:, 0]).rename("is_HLA")
    # print(sr_is_HLA_locus)
    
    
    ### (2) HLA marker type and gene (걍 같이 처리)
    
    sr_type_and_gene = df_index.iloc[:, 0].map(lambda x: get_HLA_markertype_and_gene(x))
    
    df_type_and_gene = pd.DataFrame(
        [list(_) for _ in sr_type_and_gene],
        columns=['MarkerType', 'HLAgene'],
        index = df_index.index
    )
    # print(df_type_and_gene)
    
    df_index = pd.concat([df_index, sr_is_HLA_locus, df_type_and_gene], axis=1)

    return df_index
    


##### Plotting