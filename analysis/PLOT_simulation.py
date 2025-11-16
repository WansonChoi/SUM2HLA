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



class plot_scenario_1():

    def __init__(self, _df_RRC_PP_Pval):

        self.df_RRC_PP_Pval = _df_RRC_PP_Pval



    @classmethod
    def the_usual_please(cls, **kwargs):

        _fpath_fixed = "/data02/wschoi/_hCAVIAR_v2/20250920_eval_sim_v4_3/ToPlot.scenario1.txt"

        df_RRC = pd.read_csv(_fpath_fixed, sep='\t', header=0)

        df_RRC = df_RRC.iloc[:-1, :] # 마지막 row하나만 제외.

        return cls(df_RRC, **kwargs)



    def plot_scenario1(self, _ax, _color_PP="red", _color_Pval="blue"):


        ##### Main plotting

        self.df_RRC_PP_Pval.plot.bar(ax = _ax, width=0.6, color=[_color_PP, _color_Pval])

        # _ax.set_title("Scenario 1", fontsize=16)
        _ax.set_xlabel("True assciation z-score (of the causal HLA variant)")
        # _ax.set_xlabel("True assciation z-score\n(of the causal HLA variant)")
        _ax.set_ylabel("Recall Rate")

        
        ##### xlabel 회전

        ## 45도
        # for lbl in _ax.get_xticklabels():
        #     lbl.set_rotation(45)
        #     lbl.set_ha('right')

        ## 90도
        _ax.tick_params(axis='x', labelrotation=0)   # = rotation=0
        for lbl in _ax.get_xticklabels():
            lbl.set_ha('center')  # 중앙 정렬



        ##### legend
        _handles, _labels = _ax.get_legend_handles_labels()

        # 원하는 표시명으로 매핑
        _label_map = {'SUM2HLA': 'PP', 'S_Imp': 'P-value'}
        _new_labels = [_label_map.get(l, l) for l in _labels]

        # 레전드 다시 배치 (폰트 크기 축소)
        fontsize = 10 # 기본
        fontsize = 8.5
        _ax.legend(_handles, _new_labels, title=None, fontsize=fontsize, frameon=True)        



        ##### 가에 없애기

        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)

        return 0
    


    def run(self, _style="default", _f_use_Arial=True):

        with plt.style.context(_style):

            rc = {}
            if _f_use_Arial:
                rc.update({
                    "font.family": "Arial",
                    "font.sans-serif": ["Arial"],
                    "pdf.fonttype": 42, "ps.fonttype": 42,
                })

            with mpl.rc_context(rc = rc):

                print(mpl.rcParams["font.family"])
                print(mpl.rcParams.get("font.sans-serif"))                

                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                
                self.plot_scenario1(ax)
                fig.tight_layout()

                plt.show()
                
                return fig, ax
    


    def plot_scenario1_seaborn(self, _ax, _palette):

        df_long = self.df_RRC_PP_Pval.reset_index().melt(
            id_vars='ncp',
            var_name='Method',
            value_name='Recall'
        )
        
        sns.barplot(data=df_long, x='ncp', y='Recall', hue='Method', palette=_palette, ax=_ax)
        _ax.set_title("Recall Rate by NCP", fontsize=16)
        _ax.set_xlabel("NCP")
        _ax.set_ylabel("Recall Rate")
        _ax.legend(title="Method")

        return 0




    def run_seaborn(self, _style="white", _context="paper", _palette="colorblind"):

        with mpl.rc_context():

            with sns.axes_style(_style), sns.plotting_context(_context):

                fig, ax = plt.subplots(1, 1, figsize=(6, 4))

                self.plot_scenario1_seaborn(ax, _palette=_palette)

                fig.tight_layout()

                plt.show()

                return fig, ax


        # sns.set_theme(style=_style)        

        # fig, axes = plt.subplots(1, 1)
        # self.plot_scenario1_seaborn(axes)

        # plt.tight_layout() # 얘는 여기서하는게 좋다고함. 나중에 main figure만들때도 최종 한번만 해야하는 애인가봄.
        # plt.show()

        # sns.reset_defaults()



class plot_scenario_2():

    def __init__(self, _df_scenario2):

        self.df_RRC_PP_Pval = _df_scenario2



    def plot_scenario2(self, _ax, _color_PP='#dc143c', _width_bar=0.3):

        self.df_RRC_PP_Pval.plot.bar(ax = _ax, width=_width_bar, color=[_color_PP,])
            # bar width를 manually 조정해줘야 함.

        # _ax.set_title("Scenario 2", fontsize=16)
        _ax.set_xlabel("True assciation z-score (of the independent HLA variant)")
        # _ax.set_xlabel("True assciation z-score\n(of the independent HLA variant)")
        _ax.set_ylabel("Recall Rate")


        ##### xlabel 회전

        ## 90도
        _ax.tick_params(axis='x', labelrotation=0)   # = rotation=0
        for lbl in _ax.get_xticklabels():
            lbl.set_ha('center')  # 중앙 정렬



        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)


        leg = _ax.get_legend()
        if leg is not None:
            leg.remove()        

        return 0
    


    def run(self, _style="default", _f_use_Arial=True):

        with plt.style.context(_style):

            rc = {}
            if _f_use_Arial:
                rc.update({
                    "font.family": "Arial",
                    "font.sans-serif": ["Arial"],
                    "pdf.fonttype": 42, "ps.fonttype": 42,
                })

            with mpl.rc_context(rc = rc):


                print(mpl.rcParams["font.family"])
                print(mpl.rcParams.get("font.sans-serif"))                

                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                
                self.plot_scenario2(ax)
                fig.tight_layout()

                plt.show()
                
                return fig, ax



class plot_Figure1():

    def __init__(self, _df_scenario1, _df_scenario2, _figsize=(8, 3), _dpi=300):

        self.plotter_scenario1 = plot_scenario_1(_df_scenario1)
        self.plotter_scenario2 = plot_scenario_2(_df_scenario2)

        ### setting
        self.figsize = _figsize
        self.dpi = _dpi



    def __repr__(self):

        print(self.plotter_scenario1.df_RRC_PP_Pval)
        print(self.plotter_scenario2.df_RRC_PP_Pval)    

        return ""



    def run(self, _style="default", _f_use_Arial=True):

        ## blue + grey
        color_sim1_PP = "#0072B2"
        color_sim1_Pval = "#999999"
        color_sim2_PP = "#56B4E9"

        ## orange + blue
        color_sim1_PP = "#D55E00"
        color_sim1_Pval = "#0072B2"
        color_sim2_PP = "#E69F00"

        ## 틸(Teal) + 퍼플'
        color_sim1_PP = "#1B9E77"
        color_sim1_Pval = "#7570B3"
        color_sim2_PP = "#66A61E"

        ## Set 3에서 내가 필요로하는 색 catch (빨간색)
            ## 위 색들이 채도가 좀 통일이 안되는 느낌.
        color_sim1_PP = "#fb8072"
        color_sim1_Pval = "#80b1d3"
        color_sim2_PP = "#fdb462"

        ## "Set 1"
        color_sim1_PP = '#e41a1c'
        color_sim1_Pval = '#377eb8'
        color_sim2_PP = '#ff7f00'

        ## "Set 2" (채도는 얘가 딱 좋단 말이지...)
        color_sim1_PP = '#66c2a5'
        color_sim1_Pval = '#b3b3b3'
        color_sim2_PP = '#a6d854'


        ## "Set 2"
        color_sim1_PP = '#fc8d62'
        color_sim1_Pval = '#8da0cb'
        color_sim2_PP = '#ffd92f'

        ## "Set 2"
        color_sim1_PP = '#66c2a5'
        color_sim1_Pval = '#ffd92f'
        color_sim2_PP = '#a6d854'


        _subfigure_label_height = 1.1 # 1.15가 다른 figure에서도 쓰던 값임.


        with plt.style.context(_style):

            rc = {}
            if _f_use_Arial:
                rc.update({
                    "font.family": "Arial",
                    "font.sans-serif": ["Arial"],
                    "pdf.fonttype": 42, "ps.fonttype": 42,
                })

            with mpl.rc_context(rc = rc):

                print(mpl.rcParams["font.family"])
                print(mpl.rcParams.get("font.sans-serif"))

                fig, ax = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
                # fig, ax = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi, constrained_layout=True)
                print(ax)

                ### Fig. 1a
                self.plotter_scenario1.plot_scenario1(ax[0], _color_PP=color_sim1_PP, _color_Pval=color_sim1_Pval)

                # ax[0].text(-0.08, 1.05, 'a', transform=ax[0].transAxes,
                #         fontweight='bold', fontsize=14, va='bottom', ha='right')

                ## axis 수정 (2025.11.04.)
                ax[0].text(-0.2, _subfigure_label_height, "a", transform=ax[0].transAxes,
                        fontweight='bold', fontsize=16, va='bottom', ha='right')



                ### Fig. 1b
                self.plotter_scenario2.plot_scenario2(ax[1], _color_PP=color_sim2_PP, _width_bar=0.28)

                # ax[1].text(-0.08, 1.05, 'b', transform=ax[1].transAxes,
                #         fontweight='bold', fontsize=14, va='bottom', ha='right')

                ## axis 수정 (2025.11.04.)
                ax[1].text(-0.2, _subfigure_label_height, "b", transform=ax[1].transAxes,
                        fontweight='bold', fontsize=16, va='bottom', ha='right')


                # fig.tight_layout()
                # fig.tight_layout(w_pad=1.0, h_pad=1.0)
                fig.tight_layout(w_pad=1.1, h_pad=1.0) # width좀만 넓히자.

                plt.show()
                
                return fig, ax

        return 0
    

"""



"""