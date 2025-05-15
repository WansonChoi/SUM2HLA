import os
import re
import subprocess
from os.path import dirname

import pandas as pd

from src import SWCA_PostCalc as fine_mapping_SWCA
from src.SWCA import get_single_residue_markers



def sort_COJO_result_cond(_fpath_COJO_cond):

    df_COJO_cond = pd.read_csv(_fpath_COJO_cond, sep='\t', header=0)

    sr_zC = (df_COJO_cond['bC'] / df_COJO_cond['bC_se']).rename("zC")


    df_ToSort = pd.concat([df_COJO_cond['pC'], -sr_zC.abs()], axis=1).sort_values(['pC', 'zC'])

    df_RETURN = pd.concat([df_COJO_cond, sr_zC], axis=1).loc[df_ToSort.index, :]


    return df_RETURN



def iterate_GCTA_COJO(_fpath_ma, _initial_top_signal, _fpath_ref_bfile, _fpath_ref_ld, _fpath_ref_MAF, _out_prefix,
                      _N_max_iter=100, _f_use_finemapping=True, _f_single_factor_markers=True,
                      _gcta="/home/wschoi/bin/gcta64", _plink="/home/wschoi/miniconda3/bin/plink"):

    ## 얘가 main varaible

    ## (2025.04.25.) initial signal도 single residue marker로 전환해줘야 함.
    if _f_single_factor_markers:
        l_secondary_signals, df_ToRefer = get_single_residue_markers([_initial_top_signal], _fpath_ref_MAF, _fpath_ma)
    else:
        l_secondary_signals = [_initial_top_signal] if isinstance(_initial_top_signal, str) else _initial_top_signal



    ##### Main iteration

    for i in range(1, _N_max_iter+1):

        print("===[{}]: ROUND_{}.".format(i, i))
        # print("Current conditions:\n{}".format(l_secondary_signals))

        snplist_temp = os.path.join(_out_prefix + "{}.snplist".format(i))
        out_prefix_temp = re.sub(r'.snplist$', '', snplist_temp)

        sr_snplist = pd.Series(l_secondary_signals, name="snptlist")
        # display(sr_snplist)
        sr_snplist.to_csv(snplist_temp, header=False, index=False, na_rep="NA")


        # print("snplist: ", snplist_temp)
        # print("out_prefix: ", out_prefix_temp)


        cmd = [
            _gcta,
            "--bfile", _fpath_ref_bfile,
            "--cojo-file", _fpath_ma,
            "--cojo-cond", snplist_temp,
            "--out", out_prefix_temp
        ]
        # print("\nExecuting command:\n")
        # print(json.dumps(cmd, indent='\t'))


        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        except subprocess.CalledProcessError as e:
            # 에러 메시지를 Jupyter Notebook에 출력
            # print(f"Error occurred: {e}")

            # stderr을 파일로 기록
            with open(out_prefix_temp + ".stderr", 'w') as f_stderr:
                f_stderr.write(f"Error occurred while running GCTA64:\n{e}\n")

            """
            추후 여기서 마지막 fail한 gcta execution log에서 "colinearity" 관련 단어가 확인되면 "Done"이라고 출력해주는 코드 몇줄만 추가하셈.
            """

            break  # 에러 발생 시 루프 종료

        else:

            OUT_CMA = out_prefix_temp + ".cma.cojo"

            df_out_sort = sort_COJO_result_cond(OUT_CMA)
            df_out_sort.to_csv(OUT_CMA + ".sort", sep='\t', header=True, index=False, na_rep="NA")
            # display(df_out_sort.head(5))



            ### To use fine-mapping for the next top?
            if _f_use_finemapping:

                df_PP_cma, OUT_PP_cma = fine_mapping_SWCA.__MAIN__(
                    OUT_CMA,
                    _fpath_ref_bfile, _fpath_ref_ld,
                    dirname(OUT_CMA),
                    _plink
                )

                print(df_PP_cma.head(10))

                ## credible set에 있는 애들 확인.
                df_PP_cma = df_PP_cma[ df_PP_cma['CredibleSet'] ] # 2개 이상의 rows들일 수 있음.

                l_next_top_signal = df_PP_cma['SNP'].tolist()
                next_top_signal_pC = \
                        df_out_sort.loc[ df_out_sort['SNP'].isin(l_next_top_signal) , 'pC'].min()

            else:
                l_next_top_signal = [df_out_sort['SNP'].iat[0]]
                next_top_signal_pC = df_out_sort['pC'].iat[0]



            ### the top signal as single-factor markers? (without co-linearity)

            if _f_single_factor_markers:

                l_next_top_signal, df_ToRefer = get_single_residue_markers(l_next_top_signal, _fpath_ref_MAF, df_out_sort)
                print(f"df_ToRefer:\n{df_ToRefer}")


            ### End 조건: No more significant signals?
            if next_top_signal_pC < 5e-8:

                l_secondary_signals.extend(l_next_top_signal)
                print("\nNext conditions: ", l_secondary_signals)

            else:

                print("No more significant loci!")
                break

        # if i >= 2: break

    return l_secondary_signals





if __name__ == '__main__':

    ##### `iterate_GCTA_COJO` 테스트

    ### HLA only

    # r_temp = iterate_GCTA_COJO(
    #     "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST.HLA.ma",
    #     "AA_DRB1_13_32660109_HF",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.NoNA.PSD.ld",
    #     "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA.FRQ.frq",
    #     "/data02/wschoi/_hCAVIAR_v2/20250415_SWCA_v2/20250417_TEST_2.HLA.ROUND_",
    #     _f_use_finemapping=True, _f_single_factor_markers=True,
    #     _N_max_iter=2
    # )
    #
    # print("Final secondary signals:")
    # print(r_temp)

    pass
