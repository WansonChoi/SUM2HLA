# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import join, dirname, basename, exists
import numpy as np
import pandas as pd
import math
import importlib


import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib import gridspec

import anndata as ad
print(ad.__version__)

import scanpy as sc
sc.settings.verbosity=0


"""
- Server C에서 환경새로 받은 버전
    - scanpy=1.11.4
    - anndata=0.12.2


- 예전에 Server D
    - scanpy=1.9.3
    - anndata=0.9.1

"""




def gen_Anndata(
        _df_zscore = pd.read_csv("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_CM/UKBB.Z_table.tr.only_HLA.txt", sep='\t',header=0, index_col=0), 
        _l_N_neighbors = [5,],
        _metric='euclidean', _seed=0):
    
    ### (1) Init Anndata
    adata_init = ad.AnnData(_df_zscore)
    
    
    ### (2) run PCA
    sc.tl.pca(adata_init, n_comps=_df_zscore.shape[0], svd_solver='full')
    # N_PCs는 max PC로 고정. (고정했다! ㅋㅋㅋ 고정! ㅋㅋㅋㅋ (고정이래 ㅋㅋㅋ))
    
    print(adata_init)
    
    
    ### (3) kNN
    _N_PCs = _df_zscore.shape[0]

    d_KNN_result = {}

    for i, _N_neighbors in enumerate(_l_N_neighbors):

        print("=====[{}]: N_neighbors: {}".format(i, _N_neighbors))


        ##### (2) KNN
        df_temp_KNN = sc.pp.neighbors(adata_init, n_pcs=_N_PCs, n_neighbors=_N_neighbors, 
                                      metric=_metric, random_state=_seed, copy=True)

        ##### (3) Louvain
        # sc.tl.louvain(df_temp_KNN, key_added="louvain_res0_25", resolution=0.25, random_state=_seed)
        # sc.tl.louvain(df_temp_KNN, key_added="louvain_res0_5", resolution=0.5, random_state=_seed)
        # sc.tl.louvain(df_temp_KNN, key_added="louvain_res1", resolution=1.0, random_state=_seed)
        
        ## louvain은 안해도 된데! ㅋㅋㅋㅋ

        ##### (4) Leiden
        sc.tl.leiden(df_temp_KNN, key_added="leiden_res0_25", resolution=0.25, random_state=_seed)
        sc.tl.leiden(df_temp_KNN, key_added="leiden_res0_5", resolution=0.5, random_state=_seed)
        sc.tl.leiden(df_temp_KNN, key_added="leiden_res1", resolution=1.0, random_state=_seed)

        d_KNN_result[_N_neighbors] = df_temp_KNN.copy()
        print(d_KNN_result[_N_neighbors])
    
        
    
    return d_KNN_result



def reset_category_names(_adata):

    _adata = _adata.copy()

    df_leiden_results = _adata.obs


    def reset_category_0(_sr_leiden):

        ToRename = max(map(lambda x: int(x), _sr_leiden.cat.categories.tolist())) + 1

        return _sr_leiden.cat.rename_categories({"0": f"{ToRename}"})

    df_leiden_results = pd.concat([reset_category_0(_sr) for _col_name, _sr in df_leiden_results.items()], axis=1)

    _adata.obs = df_leiden_results

    return _adata



def run_UMAP_with_params_subplots(_adata_UKBB, _l_spread, _l_min_dist, _res="leiden_res1", _figsize=None):

    # 우리는 지금 res0_5써야 함.
    
    _adata_UKBB = _adata_UKBB.copy() # Just in case.
    
    if _figsize == None:
        _figsize = (5.5*len(_l_min_dist), 5.5*len(_l_spread))
    
    fig, axes = plt.subplots(len(_l_spread), len(_l_min_dist), constrained_layout=True, figsize=_figsize)
    # axes는 2차원인게 fix됨.
    

    for i, _spread in enumerate(_l_spread):

        print("=====[{}]: _spread: {}".format(i, _spread))    
    
        for j, _min_dist in enumerate(_l_min_dist):

            print("===[{}]: _min_dist: {}".format(j, _min_dist))
        

            # print("=====[{}, {}]: min_dist: {} / spread: {}".format(i, j, _min_dist, _spread))
            
            with rc_context({'figure.figsize': (5.5, 5.5), 'figure.dpi': 300}):

                sc.tl.umap(_adata_UKBB, min_dist=_min_dist, spread=_spread)

                ### 지금은 res하나로 fix
                sc.pl.umap(_adata_UKBB, color=_res, title="spread: {} / min_dist: {}".format(_spread, _min_dist),
                           legend_loc="on data", add_outline=True, legend_fontoutline=4,
                          ax=axes[i, j], show=False)
                    
    return fig




def run_tSNE_with_params_subplots(_adata_UKBB, _perplexity, _l_early_exaggeration, _l_learning_rate, _res="leiden_res1", _figsize=None):
    
    _adata_UKBB = _adata_UKBB.copy() # Just in case.
    
    if _figsize == None:
        _figsize = (5.5*len(_l_learning_rate), 5.5*len(_l_early_exaggeration))
    
    fig, axes = plt.subplots(len(_l_early_exaggeration), len(_l_learning_rate), constrained_layout=True, figsize=_figsize)
    # axes는 2차원인게 fix됨.
    

    for i, _early_exaggeration in enumerate(_l_early_exaggeration):

        print("=====[{}]: _early_exaggeration: {}".format(i, _early_exaggeration))    
    
        for j, _learning_rate in enumerate(_l_learning_rate):

            print("===[{}]: _learning_rate: {}".format(j, _learning_rate))
        

            # print("=====[{}, {}]: min_dist: {} / spread: {}".format(i, j, _min_dist, _spread))
            
            with rc_context({'figure.figsize': (5.5, 5.5), 'figure.dpi': 300}):

                sc.tl.tsne(_adata_UKBB, perplexity=_perplexity, early_exaggeration=_early_exaggeration, learning_rate=_learning_rate)

                ### 지금은 res하나로 fix
                sc.pl.tsne(_adata_UKBB, color=_res, title=f"early_exaggeration: {_early_exaggeration} / learning_rate: {_learning_rate}",
                           legend_loc="on data", add_outline=True, legend_fontoutline=4,
                          ax=axes[i, j], show=False)
                    
    return fig