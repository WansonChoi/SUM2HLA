# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import basename, dirname, join
import numpy as np
import scipy as sp
import pandas as pd

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




def load_and_prepare_mlog10(_fpath): 

    """
    d_df_RA_2012_round = {
        1: pd.read_csv("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_with_highprecision_P.tsv", sep='\t', header=0) \
                .drop(["P_hp_str", "P_filled"], axis=1).rename({"minus_log10P_hp": "minus_log10P"}, axis=1),
        2: load_and_prepare_mlog10("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_BFng1076_MOESM17_ESM.v2.col_renamed.Round_2.txt"),
        3: load_and_prepare_mlog10("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_BFng1076_MOESM17_ESM.v2.col_renamed.Round_3.txt")
    }
    # display(d_df_RA_2012_round)
    display(d_df_RA_2012_round[1])
    """


    df_temp = pd.read_csv(_fpath, sep='\t', header=0)
    # display(df_temp)

    sr_mlog10 = pd.Series(
        -np.log10(df_temp['P']), name='minus_log10P', index=df_temp.index
    )

    df_RETURN = pd.concat([df_temp, sr_mlog10], axis=1)

    return df_RETURN



class plot_HLA_manhattan_Pvalue_RA_2012(): 

    ## 앞서만든 `PLOT_HLA_manhattan.plot_HLA_manhattan_Pvalue`이랑 다르게 처리해줘야할게 너무 많아서 그냥 새로 만듬.

    def __init__(self, _df_Pvalue_2012, _col_mlog10P="minus_log10P", _col_HLA="Gene", _figsize=(11, 8), _dpi=300):

        self.df_ToPlot = _df_Pvalue_2012 ## 이미 전처리 되어서 왔다 가정.

        self.col_mlog10P = _col_mlog10P
        self.col_HLA = _col_HLA

        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )

    

    def plot_HLA_manhattan_Pvalue(self, _ax, _color_top='#d62728'):

        df_ToPlot = self.df_ToPlot.sort_values("Index") # 얘는 "Index"라는 column이 있음.


        # 예시 MultiIndex
        items  = [f"item{i}" for i in range(1, df_ToPlot.shape[0]+1)] # (deprecated)
        groups = df_ToPlot[self.col_HLA].tolist()
        idx = pd.MultiIndex.from_arrays([items, groups], names=["item", "group"])

        arr_mlog10Pval = df_ToPlot[self.col_mlog10P]
        # print(arr_mlog10Pval)


        df = pd.DataFrame({"y": list(arr_mlog10Pval)}, index=idx) # P-value instead of PP (list아니면 안됨.)
        print(df)

        item_labels  = df.index.get_level_values("item") # (deprecated)
        group_labels = df.index.get_level_values("group")

        # group 경계와 중심 계산
        boundaries = np.flatnonzero(group_labels[:-1].to_numpy() != group_labels[1:].to_numpy()) + 1
        # [4, 8] 같은 식으로 경계 인덱스가 나옵니다 (0-based, next-start 위치)
        runs = np.split(np.arange(len(df)), boundaries)
        group_centers = [r.mean() for r in runs]
        group_names   = [group_labels[i[0]] for i in runs]  # 각 run의 대표 group명


        boundary_ticks = np.array(boundaries, dtype=float) - 0.5

        # x, y좌표와 라벨
        x = np.arange(len(df))
        y = df["y"].to_numpy()

        ###### the top 1

        tol = 1e-2
        order = np.argsort(y)[::-1]                  # 내림차순 인덱스
        top_idx = order[0]
        second_idx = order[1] if len(order) > 1 else None

        if second_idx is not None and (y[top_idx] - y[second_idx]) < tol:
            # 공동 top: 최대값과 tol 이내인 모든 점을 포함 (동률이 2개 이상일 수도 있으므로)
            cotop_mask = (y >= y[top_idx] - tol)
        else:
            # 단일 top
            cotop_mask = np.zeros_like(y, dtype=bool)
            cotop_mask[top_idx] = True


        # top_idx = int(np.argmax(y))
        # top_mask = np.zeros_like(y, dtype=bool); 
        # top_mask[top_idx] = True

        top_mask = cotop_mask

        _ax.scatter(x[top_mask], y[top_mask],
                    s=48, color=_color_top, marker='D', edgecolors='black', zorder=2, alpha=0.9, 
                    label='top PP')

            ## Anotation은 바깥 (main figure)에서 하는걸로. hard-coding / fine-tuining할게 너무 많음.


        # ###### Credible set
        # CredibleSet_99_mask = df_ToPlot['CredibleSet'].astype(bool).values & (~top_mask)

        # # (b) 상위권: 진한 파랑 다이아몬드 (예시)
        # _ax.scatter(x[CredibleSet_99_mask], y[CredibleSet_99_mask],
        #             s=16, color='#d62728', marker='D', edgecolors='none', zorder=1, alpha=0.9, label='PP ≥ 0.10') # s=28, color='#1f77b4'

        ###### 나머지
        # rest_mask = ~(top_mask | CredibleSet_99_mask)
        rest_mask = ~(top_mask)

        # --- 그리기 ---
        # (a) 나머지: 연한 회색 다이아몬드
        _ax.scatter(x[rest_mask], y[rest_mask],
                    s=16, color='#C9CDD2', marker='D', edgecolors='none', zorder=0, alpha=0.9, label='others')



        # 셀 중앙 정렬 (반칸 여백)
        # _ax.set_xlim(-0.5, len(df)-0.5)

        # ── (1) Major: 그룹 중심만 + 라벨, tick 선은 숨김
        _ax.xaxis.set_major_locator(FixedLocator(group_centers))
        _ax.xaxis.set_major_formatter(FixedFormatter([f"{_}" for _ in group_names]))
        _ax.tick_params(axis="x", which="major", length=0, pad=6)  # 라벨 간격만

        # 라벨 스타일 적용 (rotation + italic font)
        for label in _ax.get_xticklabels(which="major"):
            # print(label)
            label.set_rotation(30)
            label.set_fontproperties(self.arial_italic)
            label.set_fontstyle("italic")  # 보조적으로 italic 명시

            # label.set_ha("right")             # 회전 기준점
            # label.set_rotation_mode("anchor") # 글자 시작점(anchor) 기준으로 회전

            label_text = label.get_text()
            if label_text in ['DQA1', 'DQB1', 'DPA1', 'DPB1']:
                label.set_fontsize(8)  

            import matplotlib.transforms as mtransforms

            if label_text == "DPA1":
                # offset = mtransforms.ScaledTranslation(-1/72, 0, _ax.figure.dpi_scale_trans)
                # label.set_transform(label.get_transform() + offset)
                pass

            elif label_text == "DPB1":
                offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                label.set_transform(label.get_transform() + offset)
            """
            혹시 라벨이 많아서 겹치거나 padding이 안 맞으면, tick_params(axis='x', pad=6)를 조금 조정하거나, label.set_ha("right") / label.set_va("top")을 붙여서 위치를 보정하면 더 깔끔해집니다.            
            
            """     

        # ── (2) Minor: 경계 위치(b-0.5)에 짧은 tick, 라벨 없음
        _ax.xaxis.set_minor_locator(FixedLocator(boundary_ticks))
        _ax.xaxis.set_minor_formatter(NullFormatter())              # 라벨 제거
        # ax.tick_params(axis="x", which="minor",
        #             length=8, width=1.2, top=True, bottom=True) # 위·아래로 튀어나오게
        # 필요하다면 안쪽 방향 화살표 느낌:
        _ax.tick_params(axis="x", which="minor", direction="inout", length=8) # legnth는 manually 조정해줘야 할 수 있음.

        # Y축 라벨 등
        _ax.set_ylabel("-log10(P-value)")

        _ax.set_xlabel("HLA variants")
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

        return x[top_mask], y[top_mask]


    def run(self):

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        top_mask = self.plot_HLA_manhattan_Pvalue(ax)
        print(top_mask)


        plt.tight_layout()
        plt.show()

        return fig, ax




class plot_Figure_RA_2012_vs_2022():

    def __init__(self, 
                 _df_RA_2012_round1, _df_RA_2012_round2, _df_RA_2012_round3, 
                 _df_RA_2022_round1, _df_RA_2022_round2, _df_RA_2022_round3, _df_ref_bim_HLA,
                 _figsize=(8, 12), _dpi=300):


        self.plotter_Pval_RA_2012_round1 = \
            plot_HLA_manhattan_Pvalue_RA_2012(_df_RA_2012_round1, _figsize=_figsize, _dpi=_dpi)
        self.plotter_Pval_RA_2012_round2 = \
            plot_HLA_manhattan_Pvalue_RA_2012(_df_RA_2012_round2, _figsize=_figsize, _dpi=_dpi)
        self.plotter_Pval_RA_2012_round3 = \
            plot_HLA_manhattan_Pvalue_RA_2012(_df_RA_2012_round3, _figsize=_figsize, _dpi=_dpi)


        self.plotter_PP_RA_2022_round1 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_RA_2022_round1, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_PP_RA_2022_round2 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_RA_2022_round2, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)
        self.plotter_PP_RA_2022_round3 = \
            PLOT_HLA_manhattan.plot_HLA_manhattan_PP(_df_RA_2022_round3, _df_ref_bim_HLA, _figsize=_figsize, _dpi=_dpi)


        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )


    @classmethod
    def from_the_usual_please(cls, **kwargs):

        df_RA_2012_round1 = pd.read_csv(
            "/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_with_highprecision_P.tsv", sep='\t', header=0
        ) \
            .drop(["P_hp_str", "P_filled"], axis=1).rename({"minus_log10P_hp": "minus_log10P"}, axis=1)
        df_RA_2012_round2 = load_and_prepare_mlog10("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_BFng1076_MOESM17_ESM.v2.col_renamed.Round_2.txt")
        df_RA_2012_round3 = load_and_prepare_mlog10("/data02/wschoi/_hCAVIAR_v2/Paper_Fig_RA_12_vs22/41588_2012_BFng1076_MOESM17_ESM.v2.col_renamed.Round_3.txt")


        df_RA_2022_round1 = pd.read_csv(
            "/data02/wschoi/_hCAVIAR_v2/20250523_soumya_2022_in_2012/RA.Soumya2022in2012.HLAtype.PP",
            sep='\t', header=0
        )
        df_RA_2022_round2 = pd.read_csv(
            "/data02/wschoi/_hCAVIAR_v2/20250523_soumya_2022_in_2012/RA.Soumya2022in2012.h-tuning.HLA.saveDRB1.r2pred0.0.imaf0.0.ROUND_/RA.Soumya2022in2012.h-tuning.HLA.r2pred0.0.imaf0.0.ROUND_1.cma.cojo.PP",
            sep='\t', header=0
        )
        df_RA_2022_round3 = pd.read_csv(
            "/data02/wschoi/_hCAVIAR_v2/20250523_soumya_2022_in_2012/RA.Soumya2022in2012.h-tuning.HLA.saveDRB1.r2pred0.0.imaf0.0.ROUND_/RA.Soumya2022in2012.h-tuning.HLA.r2pred0.0.imaf0.0.ROUND_2.cma.cojo.PP",
            sep='\t', header=0
        )

        df_ref_bim_HLA = pd.read_csv(
            "/data02/wschoi/_hCAVIAR_v2/Paper_Fig_HLAmanhattan/REF_T1DGC.hg19.HLA.HLAgenes.bim", sep='\t', header=0
        )

        return cls(df_RA_2012_round1, df_RA_2012_round2, df_RA_2012_round3,
                   df_RA_2022_round1, df_RA_2022_round2, df_RA_2022_round3, df_ref_bim_HLA, 
                   **kwargs)







    def run(self):

        fig, axes = plt.subplots(3, 2, figsize=self.figsize, dpi=self.dpi, constrained_layout=True) # , gridspec_kw={'width_ratios': [4.8, 5.2]}
        fig.subplots_adjust(hspace=0.3)

        _color_round2 = "#86c66c"

        x_top_2012_round1, y_top_2012_round1 = self.plotter_Pval_RA_2012_round1.plot_HLA_manhattan_Pvalue(axes[0, 1])
        x_top_2012_round2, y_top_2012_round2 = self.plotter_Pval_RA_2012_round2.plot_HLA_manhattan_Pvalue(axes[1, 1], _color_top=_color_round2)
        x_top_2012_round3, y_top_2012_round3 = self.plotter_Pval_RA_2012_round3.plot_HLA_manhattan_Pvalue(axes[2, 1], _color_top="blue")

        x_top_2022_round1, y_top_2022_round1 = self.plotter_PP_RA_2022_round1.plot_HLA_manhattan_PP(axes[0, 0])
        x_top_2022_round2, y_top_2022_round2 = self.plotter_PP_RA_2022_round2.plot_HLA_manhattan_PP(axes[1, 0], _color_top=_color_round2)
        x_top_2022_round3, y_top_2022_round3 = self.plotter_PP_RA_2022_round3.plot_HLA_manhattan_PP(axes[2, 0], _color_top="blue")


        ##### x-lable한번만 보이도록
        axes[0, 1].set_xlabel("")
        axes[1, 1].set_xlabel("")

        axes[0, 0].set_xlabel("")
        axes[1, 0].set_xlabel("")


        ##### Title
        axes[0, 0].set_title("Rheumatoid arthritis / Round 1", x=1.12, y=1.05, fontsize=14)
        axes[1, 0].set_title("Round 2", x=1.12, y=1.05, fontsize=14)
        axes[2, 0].set_title("Round 3", x=1.12, y=1.05, fontsize=14)
        # ax[0, 1].set_title("RA 2012 (P-value)")



        ##### subfigure label
        axes[0, 0].text(-0.2, 1.10, 'a', transform=axes[0, 0].transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        axes[1, 0].text(-0.2, 1.10, 'b', transform=axes[1, 0].transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        axes[2, 0].text(-0.2, 1.10, 'c', transform=axes[2, 0].transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')


        ##### RA_2012의 x-labels 위치 좀만 이동. (+4/72만큼 이동해놓은거에서, 중첩해서 다시 +4/72만큼 이동.)
        import matplotlib.transforms as mtransforms

        for _row_idx in [0, 1, 2]:

            _ax = axes[_row_idx, 1]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                if label_text == "DPA1":
                    offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)

                elif label_text == "DPB1":
                    offset = mtransforms.ScaledTranslation(+4/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)


        for _row_idx in [0, 1, 2]:

            _ax = axes[_row_idx, 0]
            
            for label in _ax.get_xticklabels(which="major"):

                label_text = label.get_text()

                # if label_text == "DPA1":
                #     offset = mtransforms.ScaledTranslation(+2/72, 0, _ax.figure.dpi_scale_trans)
                #     label.set_transform(label.get_transform() + offset)

                if label_text == "DPB1":
                    offset = mtransforms.ScaledTranslation(+1/72, 0, _ax.figure.dpi_scale_trans)
                    label.set_transform(label.get_transform() + offset)



        ##### Round 1 top의 label
        ax = axes[0, 0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.35, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes[0, 1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.35, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 11 and 13\n" + r"in HLA-DR$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### Round 2 top의 label
        ax = axes[1, 0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.78, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "A G/C nucleotide variant\n" + r"in $\it{HLA\text{-}B}$"+ "\n(r=0.75)",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes[1, 1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.75, 1.02,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "HLA-B*08:01 and\nAmino acid pos. 9 with D\nin HLA-B",
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0),
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        # txt = ax.text(
        #     0.35, 0.7,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
        #     "Amino acid pos. 9 with D\n" + r"in $\it{HLA\text{-}B}$",
        #     transform=ax.transAxes,              # ← 축 분수 좌표!
        #     ha="center", va="top",
        #     fontsize=10, linespacing=1.1,
        #     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0),
        # )
        # txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ##### Round 3 top의 label
        ax = axes[2, 0]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.6, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 170 with I\n" + r"in HLA-DP$\beta$1" + "\n(r=-0.85)",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0)
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)


        ax = axes[2, 1]  # RA / Round1 / PP 패널 (빨간 박스가 있는 축)

        txt = ax.text(
            0.6, 1.00,                          # (x,y) = 축 내 분수 좌표 (좌측 5%, 상단 8% 아래)
            "Amino acid pos. 9 with F\n" + r"in HLA-DP$\beta$1",  # 원하는 라벨
            transform=ax.transAxes,              # ← 축 분수 좌표!
            ha="center", va="top",
            fontsize=10, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, linewidth=0),
        )
        txt.set_in_layout(False)  # 레이아웃에 영향 주지 않도록(축 폭 줄어드는 것 방지)



        ##### control point.


        ## (b) - 2022 (left)

        _ax = axes[1, 0]
        mx = float(np.mean(x_top_2022_round1))
        # my = float(np.mean(y_top_2022_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.05 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            "control for\n" + r"HLA-DR$\beta$1",
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


        mx_last_2022 = mx
        my_last_2022 = my

        # _ax = axes[1, 0]
        # mx = float(np.mean(x_top_2022_round1))
        # my = float(np.mean(y_top_2022_round1))

        # # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        # y0, y1 = _ax.get_ylim()
        # dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        # my = 0.1 * (y1 - y0)

        # # 텍스트와 화살표를 한 번에 annotate
        # _ax.annotate(
        #     "control for\n" + r"HLA-DR$\beta$1",
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



        ## (b) - 2012 (right)


        _ax = axes[1, 1]
        mx = float(np.mean(x_top_2012_round1))
        # my = float(np.mean(y_top_2012_round1))

        # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        y0, y1 = _ax.get_ylim()
        dy = 0.05 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        my = 0.1 * (y1 - y0)

        x0, x1 = _ax.get_xlim()
        dx = 0.08 * (x1 - x0)

        # (1) 텍스트: 왼쪽으로 이동
        x_text = mx
        y_text = my + dy          # 머리 위에 위치
        txt = _ax.text(
            x_text, y_text,
            # "control for\n" + "Amino acid pos. 11 with SPG\n" + r"in HLA-DR$\beta$1",
            "control for\n" + r"HLA-DR$\beta$1",
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

        mx_last_2012 = mx
        my_last_2012 = my



        ## (c) - 2022 (left)
        _ax = axes[2, 0]
        mx = float(np.mean(x_top_2022_round2))
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
            "control for\n" + "HLA-B",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color=_color_round2,
            zorder=6, clip_on=False
        )

        _head = _ax.scatter(
            [mx_last_2022], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        ## (c) - 2012 (right)
        _ax = axes[2, 1]
        mx = float(np.mean(x_top_2012_round2))
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
            "control for\n" + "HLA-B",
            ha="center", va="bottom", fontsize=10, zorder=5,
        )
        # constrained_layout이 축 폭을 줄이지 않게(레이아웃 영향 제거)
        txt.set_in_layout(False)


        _head = _ax.scatter(
            [mx], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color=_color_round2,
            zorder=6, clip_on=False
        )

        _head = _ax.scatter(
            [mx_last_2012], [my],
            marker='v',           # 아래로 향한 삼각형
            s=32,                 # 크기
            color='#d62728',
            zorder=6, clip_on=False
        )



        # _ax = axes[2, 0]
        # mx = float(np.mean(x_top_2022_round2))
        # my = float(np.mean(y_top_2022_round2))

        # # 라벨을 화살표 위쪽에 두기 위해 y방향으로 살짝 올린 시작점
        # y0, y1 = _ax.get_ylim()
        # dy = 0.08 * (y1 - y0)  # 필요하면 0.05~0.12 범위에서 조절
        # my = 0.2 * (y1 - y0)

        # # 텍스트와 화살표를 한 번에 annotate
        # _ax.annotate(
        #     "control for\n" + r"HLA-B",
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
        #         color=_color_round2
        #     ),
        # )

        # ## round 1의 control
        # _ax.annotate(
        #     "\n" + r"    ",
        #     xy=(mx_last_2022, my),               # 화살표 머리(끝점): 평균 좌표
        #     xytext=(mx_last_2022, my + dy),      # 화살표 시작점(텍스트 위치): 위쪽
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







        ##### 시마이.

        # plt.tight_layout()
        plt.show()

        return fig, axes




    def run_v2(self):

        color_r2 = "#86c66c"
        color_r3 = "blue"

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
        outer = fig.add_gridspec(3, 2, hspace=0.12,) #  wspace=0.25

        def make_row(spec, round_title, label):
            # spec은 outer[r, :] 처럼 1행 2열 전체를 가리키는 SubplotSpec
            sf = fig.add_subfigure(spec)
            sf.text(-0.02, 1.02, label, fontsize=16, fontweight="bold",
                    transform=sf.transSubfigure, ha="right", va="bottom")
            sf.suptitle(round_title, y=1.03, fontsize=12)
            axL, axR = sf.subplots(1, 2)  # 여기서 1×2 패널 생성
            return sf, axL, axR

        sfa, axa_l, axa_r = make_row(outer[0, :], "Round 1", "a")

        ##### Round 1
        x_22_r1, y_22_r1 = self.plotter_PP_RA_2022_round1.plot_HLA_manhattan_PP(axa_l)
        x_12_r1, y_12_r1 = self.plotter_Pval_RA_2012_round1.plot_HLA_manhattan_Pvalue(axa_r)


        sfb, axb_l, axb_r = make_row(outer[1, :], "Round 2", "b")
        
        ##### Round 2
        x_22_r2, y_22_r2 = self.plotter_PP_RA_2022_round2.plot_HLA_manhattan_PP(axb_l)
        x_12_r2, y_12_r2 = self.plotter_Pval_RA_2012_round2.plot_HLA_manhattan_Pvalue(axb_r)
        
        
        
        sfc, axc_l, axc_r = make_row(outer[2, :], "Round 3", "c")

        ##### Round 3
        x_22_r3, y_22_r3 = self.plotter_PP_RA_2022_round3.plot_HLA_manhattan_PP(axc_l, _color_top=color_r3)
        x_12_r3, y_12_r3 = self.plotter_Pval_RA_2012_round3.plot_HLA_manhattan_Pvalue(axc_r, _color_top=color_r3)




        # ── 아래 줄에만 x-label 보이기
        for ax in (axa_l, axa_r, axb_l, axb_r):
            ax.set_xlabel("")


        # plt.tight_layout()


        return 0



    def run_v1(self):

        color_r2 = "#86c66c"
        color_r3 = "blue"

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi, constrained_layout=False)
        sfa, sfb, sfc = fig.subfigures(3, 1, hspace=0.15)  # a, b, c

        # ── 공용: 여백 튜닝(필요시)
        # fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.15, hspace=0.12)

        # ── SubFigure a (Round 1)
        sfa.text(-0.02, 1.02, "a", fontsize=16, fontweight="bold",
                transform=sfa.transSubfigure, ha="right", va="bottom")
        sfa.suptitle("RA 2022 vs. 2012 / Round 1", y=1.1, fontsize=12)
        axa_l, axa_r = sfa.subplots(1, 2)

        ##### Round 1
        x_22_r1, y_22_r1 = self.plotter_PP_RA_2022_round1.plot_HLA_manhattan_PP(axa_l)
        x_12_r1, y_12_r1 = self.plotter_Pval_RA_2012_round1.plot_HLA_manhattan_Pvalue(axa_r)



        # ── SubFigure b (Round 2)
        sfb.text(-0.02, 1.02, "b", fontsize=16, fontweight="bold",
                transform=sfb.transSubfigure, ha="right", va="bottom")
        sfb.suptitle("Round 2", y=1.1, fontsize=12)
        axb_l, axb_r = sfb.subplots(1, 2)

        x_22_r2, y_22_r2 = self.plotter_PP_RA_2022_round2.plot_HLA_manhattan_PP(axb_l)
        x_12_r2, y_12_r2 = self.plotter_Pval_RA_2012_round2.plot_HLA_manhattan_Pvalue(axb_r)



        # ── SubFigure c (Round 3)
        sfc.text(-0.02, 1.02, "c", fontsize=16, fontweight="bold",
                transform=sfc.transSubfigure, ha="right", va="bottom")
        sfc.suptitle("Round 3", y=1.1, fontsize=12)
        axc_l, axc_r = sfc.subplots(1, 2)

        x_22_r3, y_22_r3 = self.plotter_PP_RA_2022_round3.plot_HLA_manhattan_PP(axc_l, _color_top=color_r3)
        x_12_r3, y_12_r3 = self.plotter_Pval_RA_2012_round3.plot_HLA_manhattan_Pvalue(axc_r, _color_top=color_r3)



        # ── 컬럼 헤더(첫 줄에만)
        # axa_l.set_title("RA 2022 (PP)")
        # axa_r.set_title("RA 2012 (P-value)")



        # 필요 시: 상단 마커 잘림 방지용 여유
        # def add_headroom(ax, frac=0.08):
        #     y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + frac*(y1 - y0))
        # for ax in (axa_l, axa_r, axb_l, axb_r, axc_l, axc_r):
        #     add_headroom(ax, 0.06)

        # ── (옵션) 특정 tick 라벨 미세 이동 예시
        # import matplotlib.transforms as mtrans
        # for ax in (axa_r, axb_r, axc_r):  # 2012 열
        #     for lab in ax.get_xticklabels(which="major"):
        #         if lab.get_text() == "DPB1":
        #             off = mtrans.ScaledTranslation(+4/72, 0, ax.figure.dpi_scale_trans)
        #             lab.set_transform(lab.get_transform() + off)

        # 저장 시 잘림 방지
        # fig.savefig("RA_3x2.pdf", bbox_inches="tight", pad_inches=0.04)

        plt.constrans

        return fig, dict(
            a=(axa_l, axa_r),
            b=(axb_l, axb_r),
            c=(axc_l, axc_r),
            round_coords=dict(
                r1=(x_22_r1, y_22_r1, x_12_r1, y_12_r1),
                r2=(x_22_r2, y_22_r2, x_12_r2, y_12_r2),
                r3=(x_22_r3, y_22_r3, x_12_r3, y_12_r3),
            )
        )
