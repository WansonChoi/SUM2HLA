# %load_ext autoreload
# %autoreload 2

import os, sys, re
from os.path import basename, dirname, join
import numpy as np
import scipy as sp
import pandas as pd

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
        _ax.set_xlabel("True association z-score\n(of the primary causal variant)")
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
        _label_map = {'SUM2HLA': 'APP', 'S_Imp': 'P-value'}
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

                return fig, ax



class plot_scenario_2():

    def __init__(self, _df_scenario2):

        self.df_RRC_PP_Pval = _df_scenario2



    def plot_scenario2(self, _ax, _color_PP='#dc143c', _width_bar=0.3):

        self.df_RRC_PP_Pval.plot.bar(ax = _ax, width=_width_bar, color=[_color_PP,])
            # bar width를 manually 조정해줘야 함.

        # _ax.set_title("Scenario 2", fontsize=16)
        _ax.set_xlabel("True association z-score\n(of the independent secondary causal variant)")
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

                return fig, ax

        return 0



if __name__ == "__main__":

    import matplotlib.font_manager as fm

    arial_paths = [
        "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf",
        "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Bold.ttf",
        "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Italic.ttf",
        "/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial Bold Italic.ttf",
    ]
    for path in arial_paths:
        fm.fontManager.addfont(path)
    mpl.rcParams['font.family'] = fm.FontProperties(fname="/data02/wschoi/_hCAVIAR_v2/Arial_fonts/Arial.ttf").get_name()
    mpl.rcParams['svg.fonttype'] = 'none'

    # --- data load ---
    _sim_dir = "/data02/wschoi/_hCAVIAR_v2/20260412_review_simulation"

    df_ToPlot_sc1 = pd.read_csv(
        f"{_sim_dir}/ToPlot.scenario1.txt",
        sep='\t', header=0, index_col=[0]
    ).iloc[:4, :]  # ncp 6, 8, 10, 20 (30은 saturated)

    df_ToPlot_sc2 = pd.read_csv(
        f"{_sim_dir}/ToPlot.scenario2.txt",
        sep='\t', header=0, index_col=[0]
    ).loc[:, ['ROUND_2_r2_0_99']].iloc[:4, :]  # ncp2 6, 8, 10, 20

    # --- plot ---
    plotter = plot_Figure1(df_ToPlot_sc1[["SUM2HLA", "S_Imp"]], df_ToPlot_sc2, _figsize=(8.65, 3.5))
    fig, ax = plotter.run()

    # --- 95% Clopper-Pearson error bars ---
    from scipy import stats as _stats

    def _cp95(p, n):
        k = round(p * n)
        lo = _stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi = _stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        return lo, hi

    def _add_errorbars(_ax, _containers_idx_p_n_pairs):
        for col_idx, (p_vals, n_vals) in _containers_idx_p_n_pairs:
            container = _ax.containers[col_idx]
            x_pos = [bar.get_x() + bar.get_width() / 2 for bar in container]
            y_pos = [bar.get_height() for bar in container]
            cis   = [_cp95(p, n) for p, n in zip(p_vals, n_vals)]
            lo_e  = [max(0.0, y - ci[0]) for y, ci in zip(y_pos, cis)]
            hi_e  = [max(0.0, ci[1] - y) for y, ci in zip(y_pos, cis)]
            _ax.errorbar(x_pos, y_pos, yerr=[lo_e, hi_e],
                         fmt='none', ecolor='black', elinewidth=0.8, capsize=3)

    N_sc1 = df_ToPlot_sc1['N_applicable'].values  # [70, 299, 459, 500]

    # scenario 1: col 0 = SUM2HLA, col 1 = S_Imp (same N_applicable)
    _add_errorbars(ax[0], [
        (0, (df_ToPlot_sc1['SUM2HLA'].values, N_sc1)),
        (1, (df_ToPlot_sc1['S_Imp'].values,   N_sc1)),
    ])

    # scenario 2: col 0 = ROUND_2_r2_0_99, n=500 for all
    _add_errorbars(ax[1], [
        (0, (df_ToPlot_sc2['ROUND_2_r2_0_99'].values, [500] * len(df_ToPlot_sc2))),
    ])

    # --- export ---
    _fpath_out = "/data02/wschoi/_hCAVIAR_v2/SUM2HLA/Fig.simulation.v3.pdf"
    fig.savefig(_fpath_out, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(_fpath_out.replace("pdf", "svg"), format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(_fpath_out.replace("pdf", "png"), format="png", dpi=300, bbox_inches="tight", facecolor='white')
    print(f"Saved: {_fpath_out}")