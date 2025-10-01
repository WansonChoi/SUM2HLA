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



def extract_HLAgene(_marker):

    """
    아래와 같이 활용해서 검증함.

    sr_HLA_gene = df_T1DGC_bim_HLA['SNP'].map(lambda x: extract_HLAgene(x))
    print(sr_HLA_gene.isna().any())

    l_HLAgenes = ['A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1']
    print(sr_HLA_gene.map(lambda x: x in l_HLAgenes).all())

    display(sr_HLA_gene)

    """

    p_AA = re.compile(r"^AA_(\w+)_-?\d+_\d+(_\w+)?$")
    p_HLA = re.compile(r"^HLA_(\w+)_\w+$")
    p_SNP = re.compile(r"^SNP_(\w+)_\d+(_\w+)?$")

    m_AA = p_AA.match(_marker)
    m_HLA = p_HLA.match(_marker)
    m_SNP = p_SNP.match(_marker)

    if bool(m_AA):
        return m_AA.group(1)
    elif bool(m_HLA):        
        return m_HLA.group(1)
    elif bool(m_SNP):
        return m_SNP.group(1)
    else:
        return np.nan
    


class plot_HLA_manhattan_PP():

    def __init__(self, _df_PP, _df_bim_HLAgenes, _figsize=(11, 8), _dpi=300):

        self.df_PP = _df_PP
        self.df_bim_HLAgenes = _df_bim_HLAgenes

        self.df_ToPlot = None
        self.x_top_mask = None
        self.y_top_mask = None

        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )


    def prepr_PP(self):

        # self.df_PP = self.df_PP.merge(self.df_bim_HLAgenes, on='SNP', how='left')

        # self.df_ToPlot = self.df_bim_HLAgenes.merge(self.df_PP, on='SNP', how='left') \
        #                 .dropna(subset=["PP", "CredibleSet"])
            ## 이게 right로 해놓으면 다시 PP로 sorting해놓은 순서가 되어버림.
            ## 그 와중에 left로 해놓으면 intragenic SNPs들도 다시 남음.
            ## 걍 left로 해서 bim파일 전체 집합으로 match하고, 걍 dropna()하는게 낫겠음.

        self.df_ToPlot = self.df_bim_HLAgenes.merge(self.df_PP, on='SNP', how='left')
        self.df_ToPlot['PP'].fillna(0.0, inplace=True)
        self.df_ToPlot['CredibleSet'].fillna(False, inplace=True)
            ## marker set을 3,090 HLA variants들로 통일하려면, 그냥 left_join시키고 NA있는대로 두셈.



        return self.df_ToPlot
    

    def plot_HLA_manhattan_PP(self, _ax, _color_top='#d62728'):

        df_ToPlot = self.prepr_PP()


        ##### MultiIndex 만들기 (지금은 multi-index는 아님.)

        items  = [f"item{i}" for i in range(1, df_ToPlot.shape[0]+1)] # (deprecated)
        groups = df_ToPlot['HLA'].tolist()
        idx = pd.MultiIndex.from_arrays([items, groups], names=["item", "group"])

        df = pd.DataFrame({"y": df_ToPlot['PP'].tolist()}, index=idx)
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
                    s=48, color=_color_top, marker='D', edgecolors='black', zorder=3, alpha=0.9, 
                    label='top PP', clip_on=False)

            ## Anotation은 바깥 (main figure)에서 하는걸로. hard-coding / fine-tuining할게 너무 많음.


        _ax.vlines(x[top_mask], ymin=0.0, ymax=y[top_mask],
                   colors=_color_top, linestyles='-', linewidth=(1 / len(x[top_mask])), alpha=0.7, zorder=2)


        ###### Credible set
        CredibleSet_99_mask = df_ToPlot['CredibleSet'].astype(bool).values & (~top_mask)

        # (b) 상위권: 진한 파랑 다이아몬드 (예시)
        _ax.scatter(x[CredibleSet_99_mask], y[CredibleSet_99_mask],
                    s=16, color=_color_top, marker='D', edgecolors='none', zorder=2, alpha=0.9, label='PP ≥ 0.10',
                    clip_on=False) # s=28, color='#1f77b4'

        ###### 나머지
        rest_mask = ~(top_mask | CredibleSet_99_mask)

        # --- 그리기 ---
        # (a) 나머지: 연한 회색 다이아몬드
        _ax.scatter(x[rest_mask], y[rest_mask],
                    s=16, color='#C9CDD2', marker='D', edgecolors='none', zorder=1, alpha=0.9, label='others')


        """
        (나중에 이렇게 확장.)
        # 예: 카테고리 우선순위 높은 순서대로 정의
        rules = [
            {"mask": lambda v: v >= 0.30, "label": "PP ≥ 0.30", "color": "#e41a1c", "size": 52},
            {"mask": lambda v: v >= 0.10, "label": "0.10 ≤ PP < 0.30", "color": "#377eb8", "size": 30},
            {"mask": lambda v: v >= 0.01, "label": "0.01 ≤ PP < 0.10", "color": "#4daf4a", "size": 22},
        ]
        used = np.zeros_like(y, dtype=bool)
        for r in rules:
            m = r["mask"](y) & (~used)
            _ax.scatter(x[m], y[m], marker='D', s=r["size"], color=r["color"],
                        edgecolors='none', zorder=2, alpha=0.9, label=r["label"])
            used |= m

        # 남은 것(최하위)은 연회색
        m = ~used
        _ax.scatter(x[m], y[m], marker='D', s=16, color='#C9CDD2',
                    edgecolors='none', zorder=1, alpha=0.9, label='others')        
        
        """


        # scatter
        # _ax.scatter(x, df["y"], s=20) # 맨 처음 한꺼번에 다 그리기.





        # 셀 중앙 정렬 (반칸 여백)
        # _ax.set_xlim(-0.5, len(df)-0.5) # 이거 잠깐 끔. (2025.09.30.)

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
        _ax.set_ylabel("Posterior Probability (PP)")

        _ax.set_xlabel("HLA variants")
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

        return x[top_mask], y[top_mask]


    def run(self):

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        self.x_top_mask, self.y_top_mask = self.plot_HLA_manhattan_PP(ax)


        plt.tight_layout()
        plt.show()

        return fig, ax



class plot_HLA_manhattan_Pvalue():

    def __init__(self, _df_Pvalue, _df_bim_HLAgenes, _figsize=(11, 8), _dpi=300):

        self.df_Pvalue = _df_Pvalue
        self.df_bim_HLAgenes = _df_bim_HLAgenes

        self.df_ToPlot = None

        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )


    def prepr_Pvalue(self):

        self.df_ToPlot = self.df_bim_HLAgenes.merge(self.df_Pvalue, on='SNP', how='left') \
                        .dropna(subset=["OR", "SE", "STAT", "P"])
            ## 이게 right로 해놓으면 다시 PP로 sorting해놓은 순서가 되어버림.
            ## 그 와중에 left로 해놓으면 intragenic SNPs들도 다시 남음.
            ## 걍 left로 해서 bim파일 전체 집합으로 match하고, 걍 dropna()하는게 낫겠음.

        return self.df_ToPlot
    

    def plot_HLA_manhattan_Pvalue(self, _ax, _color_top='#d62728'):

        df_ToPlot = self.prepr_Pvalue()


        # 예시 MultiIndex
        items  = [f"item{i}" for i in range(1, df_ToPlot.shape[0]+1)] # (deprecated)
        groups = df_ToPlot['HLA'].tolist()
        idx = pd.MultiIndex.from_arrays([items, groups], names=["item", "group"])

        arr_mlog10Pval = -np.log10(df_ToPlot['P'])
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
                    label='top PP', clip_on=False)

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
                    s=16, color='#C9CDD2', marker='D', edgecolors='none', zorder=0, alpha=0.9, label='others',
                    clip_on=False)



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
                offset = mtransforms.ScaledTranslation(+3/72, 0, _ax.figure.dpi_scale_trans)
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



class plot_Figure_ReA_CLE():

    def __init__(self, _df_PP_ReA, _df_PP_CLE, _df_bim_HLAgenes, _figsize=(11, 8), _dpi=300):

        self.plotter_ReA = plot_HLA_manhattan_PP(_df_PP_ReA, _df_bim_HLAgenes)
        self.plotter_CLE = plot_HLA_manhattan_PP(_df_PP_CLE, _df_bim_HLAgenes)
        

        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )
        self.arial_regular = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf"
        )


    def run(self):

        fig, ax = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        ##### (1) ReA
        x_top_mask, y_top_mask = self.plotter_ReA.plot_HLA_manhattan_PP(ax[0])

        ax[0].set_title("Reactive arthritis (ReA)", fontsize=14)

        ax[0].text(-0.2, 1.15, 'a', transform=ax[0].transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')


        print(x_top_mask)
        print(y_top_mask)

        marker_top = "HLA-B*27"

        ax[0].annotate(
            marker_top,
            xy=(x_top_mask, y_top_mask), xycoords="data",
            xytext=(x_top_mask + 200, y_top_mask - 0.05), textcoords="data",
            ha="left", va="center",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=1, shrinkB=8 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )




        ##### (2) CLE
        x_top_mask, y_top_mask = self.plotter_CLE.plot_HLA_manhattan_PP(ax[1])

        print(x_top_mask, y_top_mask)

        ## 여기서부터 자잘자잘한 post-processing
        ax[1].set_title("Cutaneous lupus erythematosus (CLE)", fontsize=14)
        ax[1].set_ylabel("")

        ax[1].text(-0.2, 1.15, 'b', transform=ax[1].transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        l_marker_top = ['HLA-DQB1*03', 'Pro55 in HLA-DQB1'] # AA_DQB1_55_32740672_P

        ax[1].annotate(
            'HLA-DQB1*03',
            xy=(x_top_mask[0], y_top_mask[0]), xycoords="data",
            xytext=(x_top_mask[0] - 500, y_top_mask[0] - 0.01), textcoords="data",
            ha="right", va="top",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=2, shrinkB=5 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )


        # from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker

        # xt, yt = x_top_mask[1], y_top_mask[1]

        # regular = dict(fontproperties=self.arial_regular, fontsize=20)
        # italic  = dict(fontproperties=self.arial_italic,  fontsize=20)

        # # 부분 텍스트를 서로 다른 폰트로
        # ta1 = TextArea("Pro55 in ", textprops=regular)      # 일반
        # ta2 = TextArea("HLA-DQB1",  textprops=italic)       # 여기만 Arial Italic → 숫자 1도 기울어짐

        # # 가로로 붙이기
        # box = HPacker(children=[ta1, ta2], align="center", pad=0, sep=0)

        # # 점(xt, yt)을 가리키고, 라벨은 (xt+200, yt-0.08)에 배치 (모두 데이터 좌표)
        # ab = AnnotationBbox(
        #     box,
        #     (xt, yt),                       # 화살표가 가리킬 점
        #     xybox=(xt - 500, yt - 0.08),    # 라벨 위치
        #     xycoords="data", boxcoords="data",
        #     arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#636363",
        #                     shrinkA=0, shrinkB=5),
        #     frameon=False
        # )

        # ax[1].add_artist(ab)        

        ax[1].annotate(
            # "Amino acid pos. 55 with P\n" + r"in HLA-DQ$\beta$1",
            "Amino acid pos. 55 with P\n" + r"in HLA-DQB1", # 걍 inkscape갈 생각하고.
            xy=(x_top_mask[1], y_top_mask[1]), xycoords="data",
            xytext=(x_top_mask[1]-1300, y_top_mask[1] - 0.08), textcoords="data",
            ha="center", va="top",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=8, shrinkB=4.5 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            # fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )



        plt.tight_layout()
        plt.show()

        return fig, ax





### Multi-indexing 예시
"""

    def run(self):




        self.prepr_PP()

        # 예시 MultiIndex
        items  = [f"item{i}" for i in range(1, self.df_PP.shape[0]+1)]
        groups = self.df_PP['HLA'].tolist()
        idx = pd.MultiIndex.from_arrays([items, groups], names=["item", "group"])

        # 예시 데이터 (y값)
        y = self.df_PP['PP'].tolist()
        df = pd.DataFrame({"y": y}, index=idx)
        print(df)

        # x좌표와 라벨
        x = np.arange(len(df))
        item_labels  = df.index.get_level_values("item")
        group_labels = df.index.get_level_values("group")

        # group 경계와 중심 계산
        boundaries = np.flatnonzero(group_labels[:-1].to_numpy() != group_labels[1:].to_numpy()) + 1
        # [4, 8] 같은 식으로 경계 인덱스가 나옵니다 (0-based, next-start 위치)
        runs = np.split(np.arange(len(df)), boundaries)
        group_centers = [r.mean() for r in runs]
        group_names   = [group_labels[i[0]] for i in runs]  # 각 run의 대표 group명

        # print(np.flatnonzero(group_labels[:-1].to_numpy() != group_labels[1:].to_numpy()))
        # print(boundaries)
        # print(runs)
        # print(group_centers)
        # print(group_labels)
        # print(group_names)




        # figure & 2개의 축 (아래: 상위 group 라벨만, 위: 실제 플롯과 item 라벨)
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 1])
        ax  = fig.add_subplot(gs[0, 0])  # 메인 산점도 축
        # ax2 = fig.add_subplot(gs[1, 0], sharex=ax)  # 하단 group 라벨 전용 축
        ax2 = fig.add_subplot(gs[1, 0])  # 하단 group 라벨 전용 축

        # 산점도
        ax.scatter(x, df["y"], s=20)
        ax.set_xlim(-0.5, len(df)-0.5)

        # 하위(item) 라벨: 메인 축의 xtick으로
        ax.set_xticks(x)
        ax.set_xticklabels(item_labels, rotation=45)
        ax.tick_params(axis='x', which='major', pad=2)

        # 상위(group) 라벨: 아래 축에 "중심만" 표시
        ax2.set_xticks(group_centers)
        ax2.set_xticklabels(group_names, fontsize=11)

        # # 아래 축의 나머지 장식 지우기
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.tick_params(axis='y', left=False, labelleft=False) # 'left => y-spline상의 tick의 mark (눈금/ 그어놓은 곳) / 'labelleft' => 눈금의 label such as '1' and '0'
        ax2.tick_params(axis='x', length=0)

        # group 경계선 표시(메인 축에)
        for b in boundaries:
            ax.axvline(b-0.5, linestyle="--", linewidth=0.8, alpha=0.4)

        ax.set_ylabel("value")
        plt.tight_layout()
        plt.show()


"""