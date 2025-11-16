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

import src.Util as Util


# %matplotlib inline
# import seaborn as sns

# import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import rc_context
# from matplotlib import rcParams
# import matplotlib as mpl
# import matplotlib.font_manager as fm



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


# mpl.rcParams['svg.fonttype'] = 'none'




"""
- (backlink)
    - PLOT_HLA_manhattan.py

- ReA 제외하고 MVP로 first-ever fine-mapping section의 내용을 주요하게 변경했음.
    - 여기서 the four diseases들을 내세우는 걸로 바뀜. 
- 이를 위해 HLA manhattan plot도 조금 수정해야함.
    - 자잘자잘한 수준인데, 여튼 변경사항을 반영해야함.
- Gemini한테 장기적으로 변경사항을 어떻게 반영할 수 있을지에 대해서 물어봄.
    - axis: (1) 데이터 받음, (2) 전처리, and (3) 핵심 기능 run
    - 저 axis들을 단위로 variation이 생길 것 같으면 class로 떼어내면 됨. 
        - 그리고 variations들끼리도 공유할 부분은 abstract class로, variational 해지는 부분은 이 ABC를 상속받고 필요한 helper member functions들만 추가하는 식으로 구현하면 됨.
    - (cf) 상속은 이런 version-up을 위한 수단으로 적절하지 않음.
        - 언급된 variations들의 superset을 abstract class로 만들고, these variations들이 share할 부분만 superset부분이 undertake해주면 됨.
        - 임의의 variation을 주입받아 단계적으로 수행될 수 있도록 구현하면 됨. => version관리.
- 근데 저번처럼 좀 무식하게 할거임. 제안해준 것까지 반영할만큼의 상황은 아님. (알아놓기만 하는걸로)
    - 전처리도 사용해야하는 column name하나 바뀐거임.
    - 걍 단계적으로 처리하는걸 helper member functions들로 나눠서 준비하는식으로만 수정하게.
- 그리고 나서 subplots들로 the 4 diseases들 그리게 할거임.
    - axis손보고 이런거는 어차피 subfigure-level에서 건드려야겠더라고.


근데 이게 잠재적 가치 되게 큰게, 나중에 batch_run 을 위한 class를 저렇게 share하는 부분만 남기고, 관련해서 전처리 및 run함수 들 부분만 다르게 만들어서 활용하면 훨씬 편해질듯.
"""


def extract_HLAgene(_marker):

    """
    아래와 같이 활용해서 검증함.

    sr_HLA_gene = df_T1DGC_bim_HLA['SNP'].map(lambda x: extract_HLAgene(x))
    print(sr_HLA_gene.isna().any())

    l_HLAgenes = ['A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1']
    print(sr_HLA_gene.map(lambda x: x in l_HLAgenes).all())

    display(sr_HLA_gene)


    (2025.11.01.; Comment)
    - 이 함수는 지금 이 source script내에서 안씀.
    - 해당 ipynb에서 T1DGC bim을 load해서 이걸 적용하고, 적용해놓은걸 계속 받아서씀.
    - 근데 이걸 약 한달정도 지나서 다시 보니까, 전혀 모르겠음.
    - 앞으로는 이걸 가져다 쓰는 부분까지 포함해서 source에 scripting해놔야겠음.
        - class와 멤버함수를 최대한 잘 활용하자구.

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
    


class plot_HLA_manhattan_PP_v2():

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


        self.config = {
            "colname_CredibleSet": 'CredibleSet(99%)', # 예전에 v1에서는 'CredibleSet' 이었음.
        }


    def prepr_ref_bim(self):


        ### (1) load check
        if isinstance(self.df_bim_HLAgenes, str) and os.path.exists(self.df_bim_HLAgenes):

            self.df_bim_HLAgenes = pd.read_csv(
                self.df_bim_HLAgenes, 
                sep='\t', header=None, names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2']
            )

        elif isinstance(self.df_bim_HLAgenes, pd.DataFrame):
            self.df_bim_HLAgenes = self.df_bim_HLAgenes.copy()

        else:
            return -1


        ### (2) HLA markers들만 남기기.
        f_is_HLA_markers = Util.is_HLA_locus(self.df_bim_HLAgenes['SNP'])

        self.df_bim_HLAgenes = self.df_bim_HLAgenes[f_is_HLA_markers]


        ### (3) HLA gene names들 extract하기
        sr_HLA_gene = self.df_bim_HLAgenes['SNP'].map(lambda x: extract_HLAgene(x)).rename("HLA")
        # print(sr_HLA_gene.isna().any())


        ### (4) concat (여기까지해서는 3,090 HLA markers들 남음.)
        self.df_bim_HLAgenes = pd.concat([self.df_bim_HLAgenes, sr_HLA_gene], axis=1)

        return self.df_bim_HLAgenes


    def prepr_PP(self):

        # self.df_PP = self.df_PP.merge(self.df_bim_HLAgenes, on='SNP', how='left')

        # self.df_ToPlot = self.df_bim_HLAgenes.merge(self.df_PP, on='SNP', how='left') \
        #                 .dropna(subset=["PP", "CredibleSet"])
            ## 이게 right로 해놓으면 다시 PP로 sorting해놓은 순서가 되어버림.
            ## 그 와중에 left로 해놓으면 intragenic SNPs들도 다시 남음.
            ## 걍 left로 해서 bim파일 전체 집합으로 match하고, 걍 dropna()하는게 낫겠음.


        _colname = self.config['colname_CredibleSet']

        self.df_ToPlot = self.df_bim_HLAgenes.merge(self.df_PP, on='SNP', how='left')

        ## fillna(..., inplace=True) 이렇게 쓰면 안된다고 함 이제.
        self.df_ToPlot['PP'] = self.df_ToPlot['PP'].fillna(0.0)
        self.df_ToPlot[_colname] = self.df_ToPlot[_colname].fillna(False)
            ## marker set을 3,090 HLA variants들로 통일하려면, 그냥 left_join시키고 NA있는대로 두셈.



        return self.df_ToPlot
    

    def plot_HLA_manhattan_PP(self, _ax, _color_top='#d62728'):

        _  = self.prepr_ref_bim()
        df_ToPlot = self.prepr_PP()

        _colname = self.config['colname_CredibleSet']



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
        CredibleSet_99_mask = df_ToPlot[_colname].astype(bool).values & (~top_mask)

        if np.any(CredibleSet_99_mask):
            """
            (2025.11.04.)
            - 이게 credible set없는데 ax.scatter하게 하니까, 전체 figure 좌하단에 이상한 점 하나가 찍힘.
            """
    
            # print("Credibe To Plot any?:", np.any(CredibleSet_99_mask))

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



class plot_Figure_FirstEverHLAfm_4diseases():

    """
    - Actinic keratosis (AK), 
    - Cutaneous lupus erythematosus (CLE), 
    - Mycoses, and 
    - Dermatophytosis (DP).
    
    """

    @classmethod
    def from_paths(cls, _fpath_AK, _fpath_CLE, _fpath_Mycoses, _fpath_DP, 
                   _df_bim_HLAgenes, **kwargs):
        
        """
        outdir: "/data02/wschoi/_hCAVIAR_v2/20251015_MVP_whole_EUR_run"

        - AK: "MVP.GCST90476204.Actinic_keratosis.AA+HLA.PP"
        - CLE: "MVP.GCST90476182.Cutaneous_lupus_erythematosus.AA+HLA.PP"
        - Mycoses: "MVP.GCST90477160.Mycoses.AA+HLA.PP"
        - Dermatophytosis: "MVP.GCST90475561.Dermatophytosis.AA+HLA.PP"
        
        """
        
        df_PP_AK = pd.read_csv(_fpath_AK, sep='\t', header=0)
        df_PP_CLE = pd.read_csv(_fpath_CLE, sep='\t', header=0)
        df_PP_Mycoses = pd.read_csv(_fpath_Mycoses, sep='\t', header=0)
        df_PP_DP = pd.read_csv(_fpath_DP, sep='\t', header=0)
        
        return cls(df_PP_AK, df_PP_CLE, df_PP_Mycoses, df_PP_DP, _df_bim_HLAgenes, **kwargs)



    def __init__(self, 
                 _df_PP_AK, _df_PP_CLE, _df_PP_Mycoses, _df_PP_DP, 
                 _df_bim_HLAgenes, 
                 _figsize=(11, 8), _dpi=300, _w_h_space_constrained_layout=(0.04, 0.06)):

        self.plotter_AK = plot_HLA_manhattan_PP_v2(_df_PP_AK, _df_bim_HLAgenes)
        self.plotter_CLE = plot_HLA_manhattan_PP_v2(_df_PP_CLE, _df_bim_HLAgenes)
        self.plotter_Mycoses = plot_HLA_manhattan_PP_v2(_df_PP_Mycoses, _df_bim_HLAgenes)
        self.plotter_DP = plot_HLA_manhattan_PP_v2(_df_PP_DP, _df_bim_HLAgenes)
        

        ### figure parameteres
        self.figsize = _figsize
        self.dpi = _dpi

        # 마지막으로 작업했을 때 (0.04, 0.06) 이 가장 무난했음. (default parameters들은 (0.02, 0.02)라고 함.)
        self.wspace_constraied_layout = _w_h_space_constrained_layout[0]
        self.hspace_constraied_layout = _w_h_space_constrained_layout[1]


        self.arial_italic = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf"
        )
        self.arial_regular = fm.FontProperties(
            fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf"
        )



    def run(self):

        """
        - 얘가 main figure 1임, 4개의 subplots들로 구성된.
        - 그냥 여기다가 hard-coding스럽게 갈거임.
            - plotting은 hard-coding을 똑똑하게 하는게 맞다는 내 생각에는 변함이 없음.
            - refactoring 및 일반화는 뒷전임. 왜냐하면, 이걸 할 일이 잘 없거든 ㅋㅋㅋ
        
            
        - constrained_layout만 그나마 w/hspace값이 조절 가능함. tight_layout은 사실상 조절 불가임.
        - 한편, 이 둘을 안쓰면 진짜 개판임. 여기서 subplots_adjust만 가지고 조절하려면 헬일듯.
        """

        fig, ax = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi, constrained_layout=True) # 2x2 (2025.11.03.)
        fig.set_constrained_layout_pads(
            wspace=self.wspace_constraied_layout, 
            hspace=self.hspace_constraied_layout
            )
        # plt.subplots_adjust(hspace=0.8) # tight_layout은 의미가 없음.

        _fontsize_subtitile = 15

        ## Dark2
        # _color_AK = "#1b9e77"
        # _color_CLE = "#d95f02"
        # _color_Mycoses = "#e7298a"
        # _color_DP = "#66a61e"


        ## Set2
        _color_AK = "#66c2a5"
        _color_CLE = "#fc8d62"
        _color_Mycoses = "#e78ac3"
        _color_DP = "#a6d854"



        ##### (1) AK (Actinic keratosis)

        _ax_temp = ax[0, 0]
        _subtitile = "Actinic keratosis"
        _PP_toPrint = 0.97
        _marker_top = f"HLA-DQB1*03:02\n(PP={_PP_toPrint})"
        _color = _color_AK
        _plotter = self.plotter_AK
        _subfig_label = 'a'


        x_top_mask, y_top_mask = _plotter.plot_HLA_manhattan_PP(_ax_temp, _color_top=_color)
        print(x_top_mask)
        print(y_top_mask)


        _ax_temp.set_title(_subtitile, fontsize=_fontsize_subtitile)
        _ax_temp.set_xlabel("")

        _ax_temp.text(-0.2, 1.15, _subfig_label, transform=_ax_temp.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')

        _ax_temp.annotate(
            _marker_top,
            xy=(x_top_mask, y_top_mask), xycoords="data",
            xytext=(x_top_mask - 1300, y_top_mask - 0.3), textcoords="data",
            ha="center", va="center",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=2, shrinkB=8 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            # fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )


        import matplotlib.transforms as mtransforms
        for label in _ax_temp.get_xticklabels(which="major"):

            label_text = label.get_text()

            # if label_text == "DPA1":
            #     offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
            #     label.set_transform(label.get_transform() + offset)

            if label_text == "DPB1":
                offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
                label.set_transform(label.get_transform() + offset)




        ##### (2) CLE

        _ax_temp = ax[0, 1]
        _subtitile = "Cutaneous lupus erythematosus"
        _color = _color_CLE
        _plotter = self.plotter_CLE
        _subfig_label = 'b'

        # _marker_top = "HLA-DRB1*07:01"
        # _marker_top = "HLA-DRB1*07:01, pos. 11G, 13Y,\nand 5 other variants\n(8 joint top; Total PP=0.56)"
        _marker_top = "HLA-DRB1*07:01,\nAmino acid pos. 11G, 13Y,\nand 5 other variants\n(8 joint top; Total PP=0.56)"
        _fontsize_text = 11



        x_top_mask, y_top_mask = _plotter.plot_HLA_manhattan_PP(_ax_temp, _color_top=_color)
        print(x_top_mask, y_top_mask)

        ## 여기서부터 자잘자잘한 post-processing
        _ax_temp.set_title(_subtitile, fontsize=_fontsize_subtitile)
        _ax_temp.set_ylabel("")
        _ax_temp.set_xlabel("")

        _ax_temp.text(-0.2, 1.15, _subfig_label, transform=_ax_temp.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')


        _ax_temp.annotate(
            _marker_top,
            xy=(x_top_mask[0], y_top_mask[0]), xycoords="data",
            xytext=(x_top_mask[0] - 1050, y_top_mask[0] - 0.025), textcoords="data",
            ha="center", va="center",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=2, shrinkB=5 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=_fontsize_text,
            # fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )


        for label in _ax_temp.get_xticklabels(which="major"):

            label_text = label.get_text()

            # if label_text == "DPA1":
            #     offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
            #     label.set_transform(label.get_transform() + offset)

            if label_text == "DPB1":
                offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
                label.set_transform(label.get_transform() + offset)




        # ax[1].annotate(
        #     # "Amino acid pos. 55 with P\n" + r"in HLA-DQ$\beta$1",
        #     "Amino acid pos. 55 with P\n" + r"in HLA-DQB1", # 걍 inkscape갈 생각하고.
        #     xy=(x_top_mask[1], y_top_mask[1]), xycoords="data",
        #     xytext=(x_top_mask[1]-1300, y_top_mask[1] - 0.08), textcoords="data",
        #     ha="center", va="top",
        #     arrowprops=dict(
        #         arrowstyle="-|>", 
        #         color="#636363", 
        #         lw=1.0,
        #         shrinkA=8, shrinkB=4.5 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
        #     ),
        #     fontsize=13,
        #     # fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        # )



        ##### (3) Mycoses

        _ax_temp = ax[1, 0]
        _subtitile = "Mycoses"
        _PP_toPrint = 1.0
        _marker_top = f"HLA-DQB1*06:02\n(PP={_PP_toPrint})"
        _color = _color_Mycoses
        _plotter = self.plotter_Mycoses
        _subfig_label = 'c'


        x_top_mask, y_top_mask = _plotter.plot_HLA_manhattan_PP(_ax_temp, _color_top=_color)
        print(x_top_mask, y_top_mask)

        ## 여기서부터 자잘자잘한 post-processing
        _ax_temp.set_title(_subtitile, fontsize=_fontsize_subtitile)
        # _ax_temp.set_ylabel("")
        # _ax_temp.set_xlabel("")

        _ax_temp.text(-0.2, 1.15, _subfig_label, transform=_ax_temp.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')


        _ax_temp.annotate(
            _marker_top,
            xy=(x_top_mask[0], y_top_mask[0]), xycoords="data",
            xytext=(x_top_mask[0] - 1300, y_top_mask[0] - 0.3), textcoords="data",
            ha="center", va="center",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=2, shrinkB=8 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            # fontproperties=self.arial_italic  # 앞서 정의한 Arial italic 폰트 적용 가능
        )


        for label in _ax_temp.get_xticklabels(which="major"):

            label_text = label.get_text()

            # if label_text == "DPA1":
            #     offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
            #     label.set_transform(label.get_transform() + offset)

            if label_text == "DPB1":
                offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
                label.set_transform(label.get_transform() + offset)



        ##### (4) Dermatophytosis

        _ax_temp = ax[1, 1]
        _subtitile = "Dermatophytosis"
        # _marker_top = "HLA-DRB1 a.a. pos. 67L"
        _PP_toPrint = 1.0
        _marker_top = f"Amino acid pos. 67L\nin HLA-DRB1\n(PP={_PP_toPrint})"
        _color = _color_DP
        _plotter = self.plotter_DP
        _subfig_label = 'd'


        x_top_mask, y_top_mask = _plotter.plot_HLA_manhattan_PP(_ax_temp, _color_top=_color)
        print(x_top_mask, y_top_mask)

        ## 여기서부터 자잘자잘한 post-processing
        _ax_temp.set_title(_subtitile, fontsize=_fontsize_subtitile)
        _ax_temp.set_ylabel("")
        # _ax_temp.set_xlabel("")

        _ax_temp.text(-0.2, 1.15, _subfig_label, transform=_ax_temp.transAxes,
                fontweight='bold', fontsize=16, va='bottom', ha='right')


        _ax_temp.annotate(
            _marker_top,
            xy=(x_top_mask[0], y_top_mask[0]), xycoords="data",
            xytext=(x_top_mask[0] - 1050, y_top_mask[0] - 0.3), textcoords="data",
            ha="center", va="center",
            arrowprops=dict(
                arrowstyle="-|>", 
                color="#636363", 
                lw=1.0,
                shrinkA=2, shrinkB=8 # shirnk B를 높일수록 점에서 화살표가 떨어지고, shrinkA를 줄일수록 textbox와 label이가까워짐.
            ),
            fontsize=13,
            # fontproperties=self.arial_italic  # 얘는 어차피 따로 넣어야 함
        )


        for label in _ax_temp.get_xticklabels(which="major"):

            label_text = label.get_text()

            # if label_text == "DPA1":
            #     offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
            #     label.set_transform(label.get_transform() + offset)

            if label_text == "DPB1":
                offset = mtransforms.ScaledTranslation(+1/72, 0, _ax_temp.figure.dpi_scale_trans)
                label.set_transform(label.get_transform() + offset)



        # plt.tight_layout() (deprecated; tight_layout은 subplots들 간의 간격을 조절할 수 없음, practically)
        plt.show()

        return fig, ax
