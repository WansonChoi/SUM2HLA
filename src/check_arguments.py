"""
A collection of functions to check given arguments
"""

import os
from os.path import exists, dirname
import pandas as pd



def check_sumstats(_sumstats) -> bool:

    ### Existence
    if not exists(_sumstats):
        raise FileNotFoundError(f'Sumstats file not found: {_sumstats}')



    ### Required columns
    l_headers_required = ["CHR", "SNP", "BP", "A1", "N", "SE", "Z", "P", "A2"]

    df_ss_temp = pd.read_csv(_sumstats, sep=r'\s+', header=0, nrows=5)
    sr_ss_headers = df_ss_temp.columns.to_series(index=None)

    if not pd.Series(l_headers_required).isin(sr_ss_headers).all():
        raise RuntimeError('The given sumstats file does not contain the following required columns: ["CHR", "SNP", "BP", "A1", "N", "SE", "Z", "P", "A2"]')


    return True



def check_reference_data(_ref) -> bool:

    ### Existence - 1: genotype files of (1) bed, (2) bim, and (3) fam
    BED = _ref + ".bed"
    BIM = _ref + ".bim"
    FAM = _ref + ".fam"
    FRQ = _ref + ".FRQ.frq"

    if not exists(BED):
        raise FileNotFoundError(f'Reference BED file not found: {BED}')

    if not exists(BIM):
        raise FileNotFoundError(f'Reference BIM file not found: {BIM}')

    if not exists(FAM):
        raise FileNotFoundError(f'Reference FAM file not found: {FAM}')

    if not exists(FRQ):
        raise FileNotFoundError(f'Reference FRQ file not found: {FRQ}')


    ### Existence - 2: LD file
    if not exists(_ref + ".NoNA.PSD.ld"):
        raise FileNotFoundError(f"Reference LD file not found: {_ref + '.NoNA.PSD.ld'}")

    return True



def check_outdir(_out_prefix) -> bool:

    if not exists(dirname(_out_prefix)):
        os.makedirs(dirname(_out_prefix), exist_ok=True)

    return True





def __MAIN__(_args):

    f_check_sumstats = check_sumstats(_args.sumstats)
    f_check_reference_data = check_reference_data(_args.ref)
    f_check_outdir = check_outdir(_args.out)

    f_ToCheck = \
        f_check_sumstats and \
        f_check_reference_data and \
        f_check_outdir

    return f_ToCheck