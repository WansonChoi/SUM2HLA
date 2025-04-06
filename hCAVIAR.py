from os.path import basename
from datetime import datetime

import src.check_arguments as check_arguments
from src.mod_hCAVIAR_batch import hCAVIAR_batch
import argparse, textwrap



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent('''\
    ###########################################################################################

        hCAVIAR.py


    ###########################################################################################
                                     '''),
                                     add_help=False
                                     )

    parser.add_argument("--help", help="Show this help message and exit.", action='help')

    ### Necessary
    parser.add_argument("--sumstats", help="GWAS summary file of the target trait.", required=True, metavar="")

    parser.add_argument("--ref", help="Reference dataset", required=True, metavar="")

    parser.add_argument("--out", help="Output file name prefix.", required=True, metavar="")


    ### Optional
    parser.add_argument("--batch-size", help="The # of causal configurations to process in a batch.",
                        default=50, metavar="")

    parser.add_argument("--skip-SWCA", help="Skip the StepWise Conditional Analysis (SWCA).",
                        action="store_true")


    ##### [1] Argument parsing #####

    ### < for Debugging > ###

    # str_temp = [
    #     "--sumstats", "data/IMPUTED.WTCCC.58C+NBS+RA.hg19.chr6.29-34mb.N4798.SNP.No_BPdup.assoc.logistic.sort2",
    #     "--ref", "data/REF_T1DGC.hg19.SNP+HLA",
    #     "--out", "tests/20250405.test"
    # ]
    # args = parser.parse_args(str_temp)


    ### < for Publish > ###
    args = parser.parse_args()

    print(args)

    ### checking arguments beforehand.
    if not check_arguments.__MAIN__(args):
        raise RuntimeError('Some arguments are incorrect. Please check your arguments')



    ##### [2] Main #####

    t_start = datetime.now()
    print(f"[ {basename(__file__)} ]: Start. ({t_start})")

    a_batch_hCAVIAR = hCAVIAR_batch(
        args.sumstats, args.ref, args.out,
        _batch_size=args.batch_size, _f_run_SWCR=(not args.skip_SWCA),
    )
    print(a_batch_hCAVIAR)
    a_batch_hCAVIAR.__MAIN__()

    t_end = datetime.now()
    print(f"\n\n[ {basename(__file__)} ]: End. ({t_end})")
    print(f"[ {basename(__file__)} ]: Total time: {t_end - t_end}")