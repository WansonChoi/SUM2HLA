# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import basename, dirname, join
import numpy as np
import scipy as sp
import pandas as pd

import statsmodels.api as sm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

# arial_paths = [
#     "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf",                     # regular
#     "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Bold.ttf",                # bold
#     "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf",              # italic
#     "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Bold Italic.ttf",         # bold italic
# ]

# # font_manager에 수동 추가
# for path in arial_paths:
#     fm.fontManager.addfont(path)

# # Arial.ttf를 기본 폰트로 설정
# mpl.rcParams['font.family'] = fm.FontProperties(fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf").get_name()

# arial_font = fm.FontProperties(fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf")
# print(arial_font.get_name())


import analysis.PLOT_HLA_manhattan as PLOT_HLA_manhattan



class plot_Figure_WTCCC():

    def __init__(self, 
                 _df_RA_PP_Round1, _df_RA_Pval_Round1, _df_RA_PP_Round2, _df_RA_Pval_Round2, 
                 _df_T1D_PP_Round1, _df_T1D_Pval_Round1, _df_T1D_PP_Round2, _df_T1D_Pval_Round2, 
                 _df_ref_bim_HLA,
                 _figsize=(11, 8), _dpi=300):


        self.plotter_RA_PP_Round1 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_RA_PP_Round1, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_RA_Pval_Round1 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_Pvalue(_df_RA_Pval_Round1, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)

        self.plotter_RA_PP_Round2 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_RA_PP_Round2, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_RA_Pval_Round2 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_Pvalue(_df_RA_Pval_Round2, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)


        self.plotter_T1D_PP_Round1 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_T1D_PP_Round1, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_T1D_Pval_Round1 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_Pvalue(_df_T1D_Pval_Round1, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)

        self.plotter_T1D_PP_Round2 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_T1D_PP_Round2, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_T1D_Pval_Round2 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_Pvalue(_df_T1D_Pval_Round2, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)



        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )



    def run_v2(self): # 얘가 Main임.

        _color_round2 = "#86c66c"

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        subfig_a, subfig_b = fig.subfigures(2, 1, hspace=0.12)  # ← a/b 사이 공간

        # ─────────────────────────────────────────
        # SubFigure a: (라벨 'a') → [Round1(1×2), Round2(1×2)]
        # ─────────────────────────────────────────
        subfig_a.text(-0.02, 1.02, "a", fontsize=16, fontweight="bold",
                    transform=subfig_a.transSubfigure, ha="right", va="bottom")

        # a를 다시 2×1로 쪼개서 "round1"과 "round2" 구역을 만든다
        axes_a = subfig_a.subplots(2, 2)
        print(axes_a)

        # Round 1: 1×2
        axes_a[0, 0].set_title("Rheumatoid arthritis / Round 1", x = 1.12, y=1.15, fontsize=12)  # ← 이 두 패널 묶음의 중앙 상단 제목

        x_top_RA_PP_round1, y_top_RA_PP_round1 = self.plotter_RA_PP_Round1.plot_HLA_manhattan_PP(axes_a[0, 0])
        x_top_RA_Pval_round1, y_top_RA_Pval_round1 = self.plotter_RA_Pval_Round1.plot_HLA_manhattan_Pvalue(axes_a[0, 1])


        # Round 2: 1×2 + 중앙 상단 제목(suptitle)
        axes_a[1, 0].set_title("Round 2", x = 1.12, y=1.1, fontsize=12)  # ← 이 두 패널 묶음의 중앙 상단 제목

        x_top_RA_PP_round2, y_top_RA_PP_round2 = self.plotter_RA_PP_Round2.plot_HLA_manhattan_PP(axes_a[1, 0], _color_top=_color_round2)
        x_top_RA_Pval_round2, y_top_RA_Pval_round2 = self.plotter_RA_Pval_Round2.plot_HLA_manhattan_Pvalue(axes_a[1, 1], _color_top=_color_round2)


        ##### x-label한번만 보이도록
        axes_a[0, 0].set_xlabel("")
        axes_a[0, 1].set_xlabel("")



        ##### RA_2012의 x-labels 위치 좀만 이동. (+4/72만큼 이동해놓은거에서, 중첩해서 다시 +4/72만큼 이동.)
        import matplotlib.transforms as mtransforms

        for _col_idx in [0, 1]:

            _ax = axes_a[0, _col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)


        for _col_idx in [0, 1]:

            _ax = axes_a[1, _col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)


        ##### Round 1 top의 label
        ax = axes_a[0, 0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.38, 1.0,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            "Amino acid pos. 11 and 13\n" + r"in HLA-DRB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0, zorder=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes_a[0, 1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.38, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            "Amino acid pos. 11 and 13\n" + r"in HLA-DRB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0, zorder=0)
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Round 2 top의 label
        ax = axes_a[1, 0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.33, 1.0,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 71 with K\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            "Amino acid pos. 71 with K\n" + r"in HLA-DRB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes_a[1, 1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.32, 1.0,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 71 with K\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            "Amino acid pos. 71 with K\n" + r"in HLA-DRB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Conditioning mark


        _ax = axes_a[1, 0]
        mx = float(np.mean(x_top_RA_PP_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            # "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DR$\beta$1",
            "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DRB1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )


        _ax = axes_a[1, 1]
        mx = float(np.mean(x_top_RA_Pval_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            # "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DR$\beta$1",
            "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DRB1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        # ─────────────────────────────────────────
        # SubFigure b: (라벨 'b') → 2×2
        # ─────────────────────────────────────────
        subfig_b.text(-0.02, 1.02, "b", fontsize=16, fontweight="bold",
                    transform=subfig_b.transSubfigure, ha="right", va="bottom")


        # a를 다시 2×1로 쪼개서 "round1"과 "round2" 구역을 만든다
        axes_b = subfig_b.subplots(2, 2)
        print(axes_b)

        # Round 1: 1×2
        axes_b[0, 0].set_title("Type 1 diabetes / Round 1", x = 1.12, y=1.15, fontsize=12)  # ← 이 두 패널 묶음의 중앙 상단 제목

        x_top_T1D_PP_round1, y_top_T1D_PP_round1 = self.plotter_T1D_PP_Round1.plot_HLA_manhattan_PP(axes_b[0, 0])
        x_top_T1D_Pval_round1, y_top_T1D_Pval_round1 = self.plotter_T1D_Pval_Round1.plot_HLA_manhattan_Pvalue(axes_b[0, 1])


        # Round 2: 1×2 + 중앙 상단 제목(suptitle)
        axes_b[1, 0].set_title("Round 2", x = 1.12, y=1.1, fontsize=12)  # ← 이 두 패널 묶음의 중앙 상단 제목

        self.plotter_T1D_PP_Round2.plot_HLA_manhattan_PP(axes_b[1, 0], _color_top=_color_round2)
        self.plotter_T1D_Pval_Round2.plot_HLA_manhattan_Pvalue(axes_b[1, 1], _color_top=_color_round2)


        ##### x-label한번만 보이도록
        axes_b[0, 0].set_xlabel("")
        axes_b[0, 1].set_xlabel("")



        for _col_idx in [0, 1]:

            _ax = axes_b[0, _col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)


        for _col_idx in [0, 1]:

            _ax = axes_b[1, _col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)




        ##### Round 1 top의 label
        ax = axes_b[0, 0]  # T1D / Round1

        txt = ax.text(
            0.53, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 57 with D\n" + r"in HLA-DQ$\beta$1",  # 원하는 라벨
            "Amino acid pos. 57 with D\n" + r"in HLA-DQB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes_b[0, 1]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.60, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 57\n" + r"in HLA-DQ$\beta$1",  # 원하는 라벨
            "Amino acid pos. 57\n" + r"in HLA-DQB1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Round 2 top의 label
        ax = axes_b[1, 0]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.45, 1.0,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 52 with R\n" + r"in HLA-DQ$\alpha$1" + "\n(r=-1.0)",  # 원하는 라벨
            # "Amino acid pos. 52 with R\n" + r"in HLA-DQ$\alpha$1",
            "Amino acid pos. 52 with R\n" + r"in HLA-DQA1",
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes_b[1, 1]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.43, 1.0,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            # "Amino acid pos. 52 with R\nand 47 with RK\n" + r"in HLA-DQ$\alpha$1",  # 원하는 라벨
            "Amino acid pos. 52 with R\nand 47 with RK\n" + r"in HLA-DQA1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none", pad=0)        
            # bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)





        ##### Conditioning mark

        _ax = axes_b[1, 0]
        mx = float(np.mean(x_top_T1D_PP_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx - dx          # 원하는 만큼 왼쪽
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            # "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQ$\beta$1",
            "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQB1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        _ax = axes_b[1, 1]
        mx = float(np.mean(x_top_T1D_Pval_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        x_text = mx - dx        # 원하는 만큼 왼쪽
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            # "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQ$\beta$1",
            "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQB1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )


        return fig





    def run(self):

        _color_round2 = "#86c66c"

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        subfig_a, subfig_b = fig.subfigures(2, 1, hspace=0.12)  # ← a/b 사이 공간

        # ─────────────────────────────────────────
        # SubFigure a: (라벨 'a') → [Round1(1×2), Round2(1×2)]
        # ─────────────────────────────────────────
        subfig_a.text(-0.02, 1.02, "a", fontsize=16, fontweight="bold",
                    transform=subfig_a.transSubfigure, ha="right", va="bottom")

        # a를 다시 2×1로 쪼개서 "round1"과 "round2" 구역을 만든다
        sf_a_r1, sf_a_r2 = subfig_a.subfigures(2, 1, hspace=0.1)

        # Round 1: 1×2
        sf_a_r1.suptitle("Rheumatoid arthritis / Round 1", y=1.15, fontsize=14)  # ← 이 두 패널 묶음의 중앙 상단 제목
        axs_a_r1 = sf_a_r1.subplots(1, 2)  # [ax11, ax12]

        # (예시 플롯)
        # for ax in axs_a_r1:
        #     x = np.linspace(0, 1, 100); ax.plot(x, np.sin(6*np.pi*x))
        #     ax.set_title("Round 1 panel")

        x_top_RA_PP_round1, y_top_RA_PP_round1 = self.plotter_RA_PP_Round1.plot_HLA_manhattan_PP(axs_a_r1[0])
        x_top_RA_Pval_round1, y_top_RA_Pval_round1 = self.plotter_RA_Pval_Round1.plot_HLA_manhattan_Pvalue(axs_a_r1[1])


        # Round 2: 1×2 + 중앙 상단 제목(suptitle)
        axs_a_r2 = sf_a_r2.subplots(1, 2)  # [ax21, ax22]
        sf_a_r2.suptitle("Round 2", y=1.1, fontsize=14)  # ← 이 두 패널 묶음의 중앙 상단 제목

        x_top_RA_PP_round2, y_top_RA_PP_round2 = self.plotter_RA_PP_Round2.plot_HLA_manhattan_PP(axs_a_r2[0], _color_top=_color_round2)
        x_top_RA_Pval_round2, y_top_RA_Pval_round2 = self.plotter_RA_Pval_Round2.plot_HLA_manhattan_Pvalue(axs_a_r2[1], _color_top=_color_round2)


        ##### x-label한번만 보이도록
        axs_a_r1[0].set_xlabel("")
        axs_a_r1[1].set_xlabel("")

        ##### RA_2012의 x-labels 위치 좀만 이동. (+4/72만큼 이동해놓은거에서, 중첩해서 다시 +4/72만큼 이동.)
        import matplotlib.transforms as mtransforms

        for _col_idx in [0, 1]:

            _ax = axs_a_r1[_col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

        for _col_idx in [0, 1]:

            _ax = axs_a_r2[_col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)


        ##### Round 1 top의 label
        ax = axs_a_r1[0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.42, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axs_a_r1[1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.42, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Round 2 top의 label
        ax = axs_a_r2[0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.35, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 71 with K\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axs_a_r2[1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.35, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 71 with K\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Conditioning mark


        _ax = axs_a_r2[0]
        mx = float(np.mean(x_top_RA_PP_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DR$\beta$1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )


        # 텍스트와 화살표를 한 번에 annotate
        # _ax.annotate(
        #     "control for\n" + "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",
        #     xy=(mx, my),               # 화살표 머리(끝점): 평균 좌표
        #     xytext=(mx, my + dy),      # 화살표 시작점(텍스트 위치): 위쪽
        #     ha="center", va="bottom",
        #     fontsize=11,
        #     arrowprops=dict(
        #         # 삼각형(정삼각형에 가까운) 헤드
        #         arrowstyle="Simple,head_length=6,head_width=6,tail_width=1.2",
        #         # 선 모양을 직선으로
        #         connectionstyle="arc3",
        #         shrinkA=0, shrinkB=0,
        #         mutation_scale=1.0,   # 필요시 전체 스케일 조절(예: 1.2, 1.5 등)
        #         color='#d62728'
        #     ),
        # )

        _ax = axs_a_r2[1]
        mx = float(np.mean(x_top_RA_Pval_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DR$\beta$1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        # 텍스트와 화살표를 한 번에 annotate
        # _ax.annotate(
        #     "control for\n" + "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",
        #     xy=(mx, my),               # 화살표 머리(끝점): 평균 좌표
        #     xytext=(mx, my + dy),      # 화살표 시작점(텍스트 위치): 위쪽
        #     ha="center", va="bottom",
        #     fontsize=11,
        #     arrowprops=dict(
        #         # 삼각형(정삼각형에 가까운) 헤드
        #         arrowstyle="Simple,head_length=6,head_width=6,tail_width=1.2",
        #         # 선 모양을 직선으로
        #         connectionstyle="arc3",
        #         shrinkA=0, shrinkB=0,
        #         mutation_scale=1.0,   # 필요시 전체 스케일 조절(예: 1.2, 1.5 등)
        #         color='#d62728'
        #     ),
        # )





        # ─────────────────────────────────────────
        # SubFigure b: (라벨 'b') → 2×2
        # ─────────────────────────────────────────
        subfig_b.text(-0.02, 1.02, "b", fontsize=16, fontweight="bold",
                    transform=subfig_b.transSubfigure, ha="right", va="bottom")


        # a를 다시 2×1로 쪼개서 "round1"과 "round2" 구역을 만든다
        sf_b_r1, sf_b_r2 = subfig_b.subfigures(2, 1, hspace=0.1)

        # Round 1: 1×2
        sf_b_r1.suptitle("Type 1 diabetes / Round 1", y=1.15, fontsize=14)  # ← 이 두 패널 묶음의 중앙 상단 제목
        axs_b_r1 = sf_b_r1.subplots(1, 2)  # [ax11, ax12]


        x_top_T1D_PP_round1, y_top_T1D_PP_round1 = self.plotter_T1D_PP_Round1.plot_HLA_manhattan_PP(axs_b_r1[0])
        x_top_T1D_Pval_round1, y_top_T1D_Pval_round1 = self.plotter_T1D_Pval_Round1.plot_HLA_manhattan_Pvalue(axs_b_r1[1])

        # Round 2: 1×2 + 중앙 상단 제목(suptitle)
        sf_b_r2.suptitle("Round 2", y=1.1, fontsize=14)  # ← 이 두 패널 묶음의 중앙 상단 제목
        axs_b_r2 = sf_b_r2.subplots(1, 2)  # [ax21, ax22]

        self.plotter_T1D_PP_Round2.plot_HLA_manhattan_PP(axs_b_r2[0], _color_top=_color_round2)
        self.plotter_T1D_Pval_Round2.plot_HLA_manhattan_Pvalue(axs_b_r2[1], _color_top=_color_round2)


        ##### 이하 후처리

        ## x-label한번만 보이도록
        axs_b_r1[0].set_xlabel("")
        axs_b_r1[1].set_xlabel("")


        for _col_idx in [0, 1]:

            _ax = axs_b_r1[_col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

        for _col_idx in [0, 1]:

            _ax = axs_b_r2[_col_idx]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPB1" and _col_idx == 0:
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1" and _col_idx == 1:
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)




        ##### Round 1 top의 label
        ax = axs_b_r1[0]  # T1D / Round1

        txt = ax.text(
            0.6, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 57\n" + r"in HLA-DQ$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axs_b_r1[1]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.6, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 57\n" + r"in HLA-DQ$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Round 2 top의 label
        ax = axs_b_r2[0]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.45, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 52 with R\n" + r"in HLA-DQ$\alpha$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axs_b_r2[1]  # T1D / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.45, 1.05,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 52 with R\nand 47 with RK\n" + r"in HLA-DQ$\alpha$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)





        ##### Conditioning mark

        _ax = axs_b_r2[0]
        mx = float(np.mean(x_top_T1D_PP_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx - dx          # 원하는 만큼 왼쪽
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQ$\beta$1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        _ax = axs_b_r2[1]
        mx = float(np.mean(x_top_T1D_Pval_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        x_text = mx - dx        # 원하는 만큼 왼쪽
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            "control for\n" + "Amino acid pos. 57 with D\n" + r"in HLA-DQ$\beta$1",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )


        return fig, (axs_a_r1, axs_a_r2, axs_b_r1, axs_b_r2)        


        # # b는 2×2 grid
        # gs_b = subfig_b.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
        # axs_b = np.array([
        #     [subfig_b.add_subplot(gs_b[0, 0]), subfig_b.add_subplot(gs_b[0, 1])],
        #     [subfig_b.add_subplot(gs_b[1, 0]), subfig_b.add_subplot(gs_b[1, 1])]
        # ])
        # # (예시 플롯)
        # for row in axs_b:
        #     for ax in row:
        #         x = np.linspace(0, 1, 100); ax.scatter(x, x + 0.1*np.random.randn(100), s=8)
        #         ax.set_title("T1D panel")

        # 필요하면 개별 축 비율 고정
        # for ax in axs_a_r1 + axs_a_r2 + axs_b.ravel().tolist():
        #     ax.set_box_aspect(1.0)  # 정사각




class LR_WTCCC():

    def __init__(self, _df_gold, _df_imputed):

        """
        - Required columns: Z-score, SNP_label, effect_allele (지금은 맞춰졌다 가정.)
        """

        self.df_gold = _df_gold.copy()
        self.df_imputed = _df_imputed.copy()

        self.df_ToLR = None # 이걸로 LR 진행할거임.
        self.model_R = None


    def prepr(self):

        df_gold = self.df_gold \
                    .rename({'STAT': 'Z'}, axis=1) \
                    .dropna(subset=['Z']) \
                    .loc[:, ['SNP', 'A1', 'Z']]
        
        df_imputed = self.df_imputed \
                    .rename({'b': 'Z'}, axis=1) \
                    .dropna(subset=['Z']) \
                    .loc[:, ['SNP', 'A1', 'Z']]
        

        self.df_ToLR = df_gold.merge(df_imputed, on=['SNP', 'A1'], suffixes = ['_gold', '_imputed'])

        return self.df_ToLR
    

    def perform_LR(self, _f_add_intercept=False):

        import statsmodels.api as sm

        Y = self.df_ToLR['Z_gold']
        X = sm.add_constant(self.df_ToLR['Z_imputed']) if _f_add_intercept else self.df_ToLR['Z_imputed']

        self.model_LR = sm.OLS(
            Y, X
        ).fit()

        print(self.model_LR.summary())

        return self.model_LR


    def plot_LR_model(self, _ax):

        # 정작 linear regression은 regplot내부에서 또 하고, 사실상 R2값과 Beta값 때문에 

        sns.regplot(data=self.df_ToLR, 
                    x='Z_imputed', y='Z_gold', line_kws={'color': 'red'}, ax=_ax)
        _ax.set_title(
            f"R²: {self.model_LR.rsquared:.4f} / Coef: {self.model_LR.params['Z_imputed']:.4f} / t-value: {self.model_LR.tvalues['Z_imputed']:.4f}"
            )
        # plt.title(f"R²: {model.rsquared:.4f}")


        return 0


    def run(self):

        self.prepr()
        self.perform_LR()


        # fig, axes = plt.subplots(1,1, figsize=(6, 6), dpi=150)

        # self.plot_LR_model(axes)


        # plt.show()


        return 0
    

    
