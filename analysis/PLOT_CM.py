# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import join, dirname, basename, exists
import numpy as np
import pandas as pd
import math
import importlib

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import rc_context
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import to_rgba_array

from matplotlib import colors as mcolors


import anndata as ad
print(ad.__version__)

import scanpy as sc
sc.settings.verbosity=0




class plot_CM():

    """
    - plotter class for the community detection analysis
    
    
    """

    def __init__(self, _adata_UMAP, _adata_tSNE, _df_corr, _figsize=(8, 4+6), _dpi=100):

        self.adata_UMAP = _adata_UMAP.copy()
        self.adata_tSNE = _adata_tSNE.copy()
        self.df_corr = _df_corr.copy()

        self.res = "leiden_res1"

        self.df_CM = _adata_UMAP.obs.copy()

        self.d_colors_clusters = {
            '3': '#1f77b4',
            '1': '#ff7f0e',
            '2': '#2ca02c'
        }
        self.l_colors_clusters = [v for k, v in self.d_colors_clusters.items()]

        self.sr_traits = None
        self.sr_CM_clusters = None
        
        
        flag = (self.df_CM == _adata_tSNE.obs).all().all()
        if not flag:
            raise ValueError("Different Leiden result!")
        
        ## Figure parameters
        self.figsize = _figsize
        self.dpi = _dpi



    def set_colors_UMAP_tSNE(self):

        self.adata_UMAP.uns['leiden_res1_colors'] = self.l_colors_clusters
        self.adata_tSNE.uns['leiden_res1_colors'] = self.l_colors_clusters


        # self.adata_UMAP.obs


        return 0



    def prepr_corr(self):

        ### sorting먼저.
        d_trait_of_group = {_cluster_N: _df.index.tolist() for _cluster_N, _df in self.df_CM.groupby(self.res)}
        print(d_trait_of_group)

        l_ToExtract = d_trait_of_group['2'] + d_trait_of_group['3'] + d_trait_of_group['1']
        l_cluster = ['2'] * len(d_trait_of_group['2']) + ['3'] * len(d_trait_of_group['3']) + ['1'] * len(d_trait_of_group['1'])

        self.df_corr.index = self.df_corr.columns

        self.df_corr = self.df_corr.loc[l_ToExtract, l_ToExtract]


        ### 여기서부터 renaming시작

        d_ToRename = {
            'Malignant_Lymphoma': 'Malignant lymphoma',
            'PrC': 'Prostate cancer',
            'SkC': 'Skin cancer',
            'Iron_Deficiency_Anemia': 'Iron deficiency anemia',
            'Sarcoidosis': 'Sarcoidosis',
            'GD': "Grave's disease",
            'Hyperthyroidism': 'Hyperthyroidism',
            'Hypothyroidism': 'Hypothyroidism',
            'T1D': 'Type 1 diabetes',
            'Iritis': 'Iritis',
            'Uveitis': 'Uveitis',
            'Asthma': 'Asthma',
            'Nasal_polyp': 'Nasal polyp',
            'Ped_asthma': 'Pediatric asthma',
            'AIH': 'Autoimmune hepatitis',
            'UC': 'Ulcerative colitis',
            'Gastic_polyp': 'Gastric polyp',
            'PsV': 'Psoriasis vulgaris',
            'T2D': 'Type 2 diabetes',
            'RA': 'Rheumatoid arthritis',
            'SLE': 'Systemic lupus erythematosus',
            'Sjogren_Syndrome': "Sjogren's syndrome",
            'Chronic_Glomerulonephritis': 'Chronic glomerulonephritis',
            'IgA_nephritis': 'IgA nephritis',
            'Nephrotic_Syndrome': 'Nephrotic syndrome',
            'PAD': 'Peripheral arterial disease',
            'Hashimoto_Disease': "Hashimoto's Disease",
            'Hearing_Loss': 'Hearing Loss',
            'Chronic_Sinusitis': 'Chronic Sinusitis',
            'JRA': 'Juvenile rheumatoid arthritis',
            'SAP': 'Stable angina pectoris'
        }

        sr_traits_sorted = self.df_corr.index.to_series().map(lambda x: d_ToRename[x] if x in d_ToRename else x)
        sr_traits_sorted = pd.Series(
            [f"{_trait} ({_clusterN})" for _trait, _clusterN in zip(sr_traits_sorted, l_cluster)],
            index = sr_traits_sorted.index,
            name = 'Diseases'
        )
        # print(sr_traits_sorted)

        # items = sr_traits_sorted.map(lambda x: d_ToRename[x] if x in d_ToRename else x)
        # palette = self.d_colors_clusters

        # sr_CM_clusters_sorted = self.adata_UMAP.obs[self.res].loc[sr_traits_sorted]

        # col_colors = sr_CM_clusters_sorted.map(palette).to_numpy()
        
        # df_index = pd.MultiIndex.from_arrays([sr_traits_sorted.map(lambda x: d_ToRename[x] if x in d_ToRename else x).tolist(), sr_CM_clusters_sorted.tolist()], names=['Trait', 'Cluster'])

        self.df_corr.index = sr_traits_sorted
        self.df_corr.columns = sr_traits_sorted

        return self.df_corr



    def run(self):

        self.set_colors_UMAP_tSNE()

        width_in=self.figsize[0]
        bottom_shape="6x8"
        dpi=self.dpi       

        # 1) 원하는 축 비율 설정
        top_box_aspect = 1.0              # 정사각(4x4)
        bottom_box_aspect = 6/8 if bottom_shape=="6x8" else 1.0  # 6x8 또는 8x8

        # 2) fig 높이를 폭에서 유도 (여백은 대략 10% 가산)
        #   - 윗줄 두 축은 각자 폭이 width_in/2 → 높이는 (width_in/2)*1.0 = 0.5*width_in
        #   - 아랫줄 히트맵은 폭이 width_in → 높이는 bottom_box_aspect*width_in
        h_top    = 0.5 * width_in
        h_bottom = bottom_box_aspect * width_in
        h_margin = 0.1 * width_in
        height_in = h_top + h_bottom + h_margin

        fig = plt.figure(figsize=(width_in, height_in), dpi=dpi, constrained_layout=False)

        # 바깥 Grid: 2행(위=산포도 2개, 아래=히트맵+컬러바)
        outer = gridspec.GridSpec(
            nrows=2, ncols=1, figure=fig,
            height_ratios=[h_top, h_bottom],
            hspace=0.15  # 층 사이 간격
        )

        # 윗줄: 1×2
        gs_top = outer[0].subgridspec(1, 2, wspace=0.24, hspace=0.12)
        ax_umap = fig.add_subplot(gs_top[0, 0])
        ax_tsne = fig.add_subplot(gs_top[0, 1])

        _dotsize = 120_000 / 60

        ax_umap.text(-0.08, 1.05, 'a', transform=ax_umap.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        sc.pl.umap(self.adata_UMAP, color=self.res, title="",
                    legend_loc="on data", add_outline=True, legend_fontoutline=4, outline_width=(0.2, 0.05),
                    size = _dotsize,  clip_on=True,
                    ax=ax_umap, show=False)
    
        ax_umap.spines['top'].set_visible(False)
        ax_umap.spines['right'].set_visible(False)



        ax_tsne.text(-0.08, 1.05, 'b', transform=ax_tsne.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        sc.pl.tsne(self.adata_tSNE, color=self.res, title="",
                    legend_loc="on data", add_outline=True, legend_fontoutline=4, outline_width=(0.2, 0.05),
                    size = _dotsize,  clip_on=True,
                    ax=ax_tsne, show=False)

        ax_tsne.spines['top'].set_visible(False)
        ax_tsne.spines['right'].set_visible(False)




        ## (변경전)

        # gs_bot = outer[1]
        # ax_heat = fig.add_subplot(gs_bot)

        # _width_ratios=[0.08, 0.92]
        _width_ratios=[0.25, 0.75]
        gs_bot = outer[1].subgridspec(1, 2, width_ratios=_width_ratios, wspace=0.0)



        ax_pad  = fig.add_subplot(gs_bot[0, 0])
        ax_pad.axis("off")

        ax_pad.text(-0.08, 1.0, 'c', transform=ax_pad.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right') # 얘는 그냥 label을 내리는걸로 (1.05 -> )


        ax_heat = fig.add_subplot(gs_bot[0, 1])

        df_corr_matrix = self.prepr_corr()
        sns.heatmap(df_corr_matrix, ax=ax_heat, cbar=True, cbar_kws={"shrink":0.9, "pad":0.02}, cmap='coolwarm')


        ax_heat.set_xticklabels(
            ax_heat.get_xticklabels(),
            rotation=55,        # 45도나 90도로 변경 가능
            ha="right",         # 회전했을 때 정렬
            # fontsize=8          # 필요시 크기도 조절
        )



        # 원하는 비(정사각/직사각) 고정
        ax_umap.set_box_aspect(top_box_aspect)
        ax_tsne.set_box_aspect(top_box_aspect)
        ax_heat.set_box_aspect(top_box_aspect)

        return fig, (ax_umap, ax_tsne, ax_heat)





class plot_CM_v2(plot_CM):

    """
    - Seaborn의 cluster heatmap을 활용해서 heatmap하나만 출력.
        - clustermap함수가 axes를 받아서 뭘 할 수가 없음.
    
    
    """

    def prepr_corr_v2(self, _col_trait = "Disease", _col_cluster = "Cluster No."):

        ##### plotting하려는 resolution 1만 가져오기 && cluster number 수정하기.
        df_trait_clusters = \
                self.adata_UMAP.obs[self.res] \
                    .rename_axis(_col_trait, axis=0).reset_index(drop=False) \
                    .rename({self.res: _col_cluster}, axis=1) \
                    .copy()
        

        ## cluster No 수정 && order 수정.

        sr_cluster = df_trait_clusters[_col_cluster].cat.rename_categories({"2": "1", "1": "2"})
        sr_cluster = sr_cluster.cat.reorder_categories(["1", "2", "3"], ordered=True)

        df_trait_clusters[_col_cluster] = sr_cluster
                                        
        df_trait_clusters = df_trait_clusters.sort_values(_col_cluster)
        # print(df_trait_clusters)

        ## cluster color 설정
        d_palette = {"1": "red", "2": "blue", "3": "green"}


        ##### df_corr준비

        self.df_corr.index = self.df_corr.columns

        ## sorting한번 하고 시작하자.
        self.df_corr = self.df_corr.loc[df_trait_clusters[_col_trait], df_trait_clusters[_col_trait]]

        
        ##### renaming하기

        ## 앞서 sorting했기 때문에 여기서 renaming하고 고대로 갔다 박을거임.

        d_ToRename = {
            'Malignant_Lymphoma': 'Malignant lymphoma',
            'PrC': 'Prostate cancer',
            'SkC': 'Skin cancer',
            'Iron_Deficiency_Anemia': 'Iron deficiency anemia',
            'Sarcoidosis': 'Sarcoidosis',
            'GD': "Grave's disease",
            'Hyperthyroidism': 'Hyperthyroidism',
            'Hypothyroidism': 'Hypothyroidism',
            'T1D': 'Type 1 diabetes',
            'Iritis': 'Iritis',
            'Uveitis': 'Uveitis',
            'Asthma': 'Asthma',
            'Nasal_polyp': 'Nasal polyp',
            'Ped_asthma': 'Pediatric asthma',
            'AIH': 'Autoimmune hepatitis',
            'UC': 'Ulcerative colitis',
            'Gastic_polyp': 'Gastric polyp',
            'PsV': 'Psoriasis vulgaris',
            'T2D': 'Type 2 diabetes',
            'RA': 'Rheumatoid arthritis',
            'SLE': 'Systemic lupus erythematosus',
            'Sjogren_Syndrome': "Sjogren's syndrome",
            'Chronic_Glomerulonephritis': 'Chronic glomerulonephritis',
            'IgA_nephritis': 'IgA nephritis',
            'Nephrotic_Syndrome': 'Nephrotic syndrome',
            'PAD': 'Peripheral arterial disease',
            'Hashimoto_Disease': "Hashimoto's Disease",
            'Hearing_Loss': 'Hearing Loss',
            'Chronic_Sinusitis': 'Chronic Sinusitis',
            'JRA': 'Juvenile rheumatoid arthritis',
            'SAP': 'Stable angina pectoris'
        }

        df_trait_clusters[_col_trait] = df_trait_clusters[_col_trait].map(lambda x: d_ToRename[x] if x in d_ToRename else x)

        print(df_trait_clusters)


        ## 방금 renaming한걸로 df_corr의 multiindex박기

        df_multiindex = pd.MultiIndex.from_frame(df_trait_clusters)

        self.df_corr.columns = df_multiindex
        self.df_corr.index = df_multiindex


        return df_trait_clusters[_col_trait], df_trait_clusters[_col_cluster], d_palette, self.df_corr
    


    def plot_CM_heatmap_v2(self):

        # 그룹별 색 지정
        sr_traits, sr_cluster_No, d_palette, df_corr = self.prepr_corr_v2()

        items = sr_traits
        col_colors = sr_cluster_No.map(lambda x: d_palette[x] if x in d_palette else "black").to_numpy()


        # 방법 1: col_colors를 위에 띠로 붙이기 (clustermap이 가장 쉬움)

        # _cmap = "bwr"
        # _cmap = "PiYG_r"
        _cmap = "RdBu_r"

        g = sns.clustermap(
            df_corr, row_cluster=False, col_cluster=False,
            xticklabels=items, yticklabels=items,
            col_colors=col_colors, row_colors=col_colors,
            cmap=_cmap, center=0, linewidths=0.0
        )

        ax_hm = g.ax_heatmap

        ## x-lables들 회전
        ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ## colorbar 위치 옮기기
        g.cax.set_position([1.0, 0.25, 0.015, 0.50])  # 오른쪽에 세로로

        return g


    def run_v2(self):

        ## 얘는 좀 예외적인게 axes를 안받아서 의미없긴함.

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        self.plot_CM_heatmap_v2()


        plt.show()

        return fig


    def plot_CM_heatmap_v3(self):

        # 그룹별 색 지정
        sr_traits, sr_cluster_No, d_palette, df_corr = self.prepr_corr_v2()

        # d_palette = { # Accent
        #     "1": '#7fc97f',
        #     "2": '#beaed4',
        #     "3": '#fdc086'
        # }

        # d_palette = { # Set2
        #     "1": '#66c2a5',
        #     "2": '#fc8d62',
        #     "3": '#8da0cb'
        # }


        d_palette = { # Set3
            "1": '#8dd3c7',
            "2": '#ffffb3',
            "3": '#bebada'
        }

        items = sr_traits
        col_colors = sr_cluster_No.map(lambda x: d_palette[x] if x in d_palette else "black").to_numpy()


        # 방법 1: col_colors를 위에 띠로 붙이기 (clustermap이 가장 쉬움)

        # _cmap = "bwr"
        # _cmap = "PiYG_r"
        _cmap = "RdBu_r"
        # _cmap = "vanimo"

        ## 하단 colorbar
        g = sns.clustermap(
            df_corr, row_cluster=False, col_cluster=False,
            xticklabels=items, yticklabels=items,
            col_colors=col_colors, row_colors=col_colors,
            cmap=_cmap, center=0, linewidths=0.0,
            cbar_kws={
                "orientation": "horizontal",
                # "label": r"$r_{\mathrm{HLA\_alleles}}$"
                # "label": "r (Between a pair of diseases with XX HLA allele markers)"
                "label": "r\n(between imputed association z-scores of the 424 HLA allele markers)"
            },
            cbar_pos=(0.35, 0.05, 0.4, 0.02),  # (left, bottom, width, height),
            figsize=self.figsize
        )



        ax_hm = g.ax_heatmap

        ## x-lables들 회전
        ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ## 
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        ## cluster label
        # grouped = sr_cluster_No.groupby(sr_cluster_No).indices  # {cluster_no: [pos1, pos2, ...]}
        # ax = g.ax_heatmap

        # for clust_no, indices in grouped.items():
        #     xmin, xmax = min(indices), max(indices)
        #     print(xmin)
        #     print(xmax)
        #     xpos = (xmin + xmax) / 2   # cluster 범위의 중앙
        #     ax.text(
        #         xpos + 0.6,                  # x 좌표 (컬럼 인덱스 단위)
        #         -2.0,                  # y 좌표 (heatmap 위보다 살짝 더 위로), 조절 필요
        #         f"Cluster {clust_no}", # 표시할 텍스트
        #         ha="center", va="bottom",
        #         fontsize=12, fontweight="bold", color="black",
        #         transform=ax.transData, clip_on=False
        #     )


        # 1) 상단 클러스터 라벨을 col_colors 축 안쪽에 그리기
        grouped = sr_cluster_No.groupby(sr_cluster_No).indices  # {cluster_no: [pos...]}
        ax_cc = g.ax_col_colors

        for clust_no, idxs in grouped.items():
            xmin, xmax = min(idxs), max(idxs)
            xpos = (xmin + xmax) / 2
            ax_cc.text(
                xpos + 0.6, 0.5, f"Cluster {clust_no}",
                ha="center", va="center",
                fontsize=11, fontweight=("bold" if clust_no == "1" else "normal"),
                color="black",
                transform=ax_cc.transData, clip_on=True,   # ← 띠 내부에 가둠,
                zorder=3
            )



        return g


    def run_v3(self):

        ## 얘는 좀 예외적인게 axes를 안받아서 의미없긴함.

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)

        g = self.plot_CM_heatmap_v3()


        plt.show()

        return fig, g # export할때는 정작 g를 써야 함.



    # def prepr_corr_v2(self):

    #     # 1) 정렬 (원래 trait id 순서 보존)
    #     d_trait_of_group = {_cluster_N: _df.index.tolist()
    #                         for _cluster_N, _df in self.df_CM.groupby(self.res)}
    #     l_ToExtract = d_trait_of_group['2'] + d_trait_of_group['3'] + d_trait_of_group['1']

    #     # corr 정렬
    #     self.df_corr.index = self.df_corr.columns
    #     self.df_corr = self.df_corr.loc[l_ToExtract, l_ToExtract]

    #     # 2) rename 표기
    #     d_ToRename = {
    #         'Malignant_Lymphoma': 'Malignant lymphoma',
    #         'PrC': 'Prostate cancer',
    #         'SkC': 'Skin cancer',
    #         'Iron_Deficiency_Anemia': 'Iron deficiency anemia',
    #         'Sarcoidosis': 'Sarcoidosis',
    #         'GD': "Grave's disease",
    #         'Hyperthyroidism': 'Hyperthyroidism',
    #         'Hypothyroidism': 'Hypothyroidism',
    #         'T1D': 'Type 1 diabetes',
    #         'Iritis': 'Iritis',
    #         'Uveitis': 'Uveitis',
    #         'Asthma': 'Asthma',
    #         'Nasal_polyp': 'Nasal polyp',
    #         'Ped_asthma': 'Pediatric asthma',
    #         'AIH': 'Autoimmune hepatitis',
    #         'UC': 'Ulcerative colitis',
    #         'Gastic_polyp': 'Gastric polyp',
    #         'PsV': 'Psoriasis vulgaris',
    #         'T2D': 'Type 2 diabetes',
    #         'RA': 'Rheumatoid arthritis',
    #         'SLE': 'Systemic lupus erythematosus',
    #         'Sjogren_Syndrome': "Sjogren's syndrome",
    #         'Chronic_Glomerulonephritis': 'Chronic glomerulonephritis',
    #         'IgA_nephritis': 'IgA nephritis',
    #         'Nephrotic_Syndrome': 'Nephrotic syndrome',
    #         'PAD': 'Peripheral arterial disease',
    #         'Hashimoto_Disease': "Hashimoto's Disease",
    #         'Hearing_Loss': 'Hearing Loss',
    #         'Chronic_Sinusitis': 'Chronic Sinusitis',
    #         'JRA': 'Juvenile rheumatoid arthritis',
    #         'SAP': 'Stable angina pectoris'
    #     }

    #     # 라벨(표기용)
    #     items = pd.Index([d_ToRename.get(x, x) for x in l_ToExtract], name="Trait")

    #     # 3) cluster → color 매핑 (정렬 순서대로)
    #     #   adata.obs.index 가 l_ToExtract 와 일치한다고 가정
    #     sr_clusters = self.adata_UMAP.obs[self.res].loc[l_ToExtract]
    #     palette = self.d_colors_clusters  # 예: {'1':'#...', '2':'#...', '3':'#...'}
    #     row_colors = sr_clusters.map(palette).to_numpy()
    #     col_colors = row_colors  # 대칭 행렬이라 동일

    #     # 4) df_corr의 라벨 교체
    #     self.df_corr.index = items
    #     self.df_corr.columns = items

    #     return self.df_corr, items, row_colors, col_colors



    # def run_v2(self):

    #     self.set_colors_UMAP_tSNE()

    #     width_in=self.figsize[0]
    #     bottom_shape="6x8"
    #     dpi=self.dpi

    #     top_box_aspect = 1.0
    #     bottom_box_aspect = 6/8 if bottom_shape=="6x8" else 1.0

    #     h_top    = 0.5 * width_in
    #     h_bottom = bottom_box_aspect * width_in
    #     h_margin = 0.1 * width_in
    #     height_in = h_top + h_bottom + h_margin

    #     fig = plt.figure(figsize=(width_in, height_in), dpi=dpi, constrained_layout=False)

    #     outer = gridspec.GridSpec(
    #         nrows=2, ncols=1, figure=fig,
    #         height_ratios=[h_top, h_bottom],
    #         hspace=0.15
    #     )

    #     # ── top row (UMAP / tSNE)
    #     gs_top = outer[0].subgridspec(1, 2, wspace=0.24, hspace=0.12)
    #     ax_umap = fig.add_subplot(gs_top[0, 0])
    #     ax_tsne = fig.add_subplot(gs_top[0, 1])

    #     ax_umap.text(-0.08, 1.05, 'a', transform=ax_umap.transAxes,
    #             fontweight='bold', fontsize=16, va='bottom', ha='right')
    #     sc.pl.umap(self.adata_UMAP, color=self.res, title="spread: 0.3 / min_dist: 0.0",
    #             legend_loc="on data", add_outline=True, legend_fontoutline=4, outline_width=(0.2, 0.05),
    #             ax=ax_umap, show=False)

    #     ax_tsne.text(-0.08, 1.05, 'b', transform=ax_tsne.transAxes,
    #             fontweight='bold', fontsize=16, va='bottom', ha='right')
    #     sc.pl.tsne(self.adata_tSNE, color=self.res, title="early_exaggeration: XXX / learning_rate: XXX",
    #             legend_loc="on data", add_outline=True, legend_fontoutline=4, outline_width=(0.2, 0.05),
    #             ax=ax_tsne, show=False)

    #     # ── bottom row: pad + heatmap composite
    #     _width_ratios=[0.25, 0.75]
    #     gs_bot = outer[1].subgridspec(1, 2, width_ratios=_width_ratios, wspace=0.0)

    #     ax_pad  = fig.add_subplot(gs_bot[0, 0])
    #     ax_pad.axis("off")
    #     ax_pad.text(-0.08, 1.0, 'c', transform=ax_pad.transAxes,
    #             fontweight='bold', fontsize=16, va='bottom', ha='right')

    #     # heatmap composite: 2x2 (corner, col_colors, row_colors, heatmap)
    #     gs_hm = gs_bot[0, 1].subgridspec(
    #         2, 2,
    #         width_ratios=[0.04, 0.96],   # 왼쪽 색띠 4%
    #         height_ratios=[0.04, 0.96],  # 위쪽 색띠 4%
    #         wspace=0.01, hspace=0.01
    #     )
    #     ax_corner = fig.add_subplot(gs_hm[0, 0]); ax_corner.axis("off")
    #     ax_col    = fig.add_subplot(gs_hm[0, 1])
    #     ax_row    = fig.add_subplot(gs_hm[1, 0])
    #     ax_heat   = fig.add_subplot(gs_hm[1, 1])

    #     # 데이터 준비
    #     df_corr_matrix, items, row_colors, col_colors = self.prepr_corr()

    #     # heatmap
    #     hm = sns.heatmap(
    #         df_corr_matrix, ax=ax_heat,
    #         cmap="vlag", center=0, linewidths=0.0,
    #         xticklabels=items, yticklabels=items,
    #         cbar=True, cbar_kws={"shrink":0.9, "pad":0.02}
    #     )
    #     # 축/라벨 스타일
    #     ax_heat.set_xticklabels(ax_heat.get_xmajorticklabels(), rotation=90)
    #     ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(), rotation=0)
    #     for spine in ["top", "right"]:
    #         ax_heat.spines[spine].set_visible(False)
    #     ax_heat.set_box_aspect(1.0)  # 정사각 비율

    #     # 색띠 그리기 (RGBA 변환 후 imshow)
    #     from matplotlib.colors import to_rgba_array
    #     col_rgba = to_rgba_array(col_colors)  # (N, 4)
    #     row_rgba = to_rgba_array(row_colors)

    #     ax_col.imshow(col_rgba[np.newaxis, :, :], aspect="auto")
    #     ax_row.imshow(row_rgba[:, np.newaxis, :], aspect="auto")

    #     # 색띠 축 미니멀화
    #     for axc in (ax_col, ax_row):
    #         axc.set_xticks([]); axc.set_yticks([])
    #         for sp in axc.spines.values():
    #             sp.set_visible(False)

    #     # 위/오른쪽 여백 조정은 필요시 fig.subplots_adjust로 미세조정

    #     # 원하는 비(정사각/직사각) 고정
    #     ax_umap.set_box_aspect(top_box_aspect)
    #     ax_tsne.set_box_aspect(top_box_aspect)
    #     # ax_heat는 위에서 1.0으로 고정

    #     return fig, (ax_umap, ax_tsne, ax_heat)










