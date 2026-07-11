"""
A collection of functions to check given arguments
"""

import os
from os.path import exists, dirname
import pandas as pd

"""
- 아래 check_sumstats()와 check_reference_data()는 각자 INPUT class에서 하면 되잖아. (2025.07.23.)
- 시간이 없어서 걍 두기로 함. (2025.11.15.)
"""

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

    ### Existence - 1: always-required files (.bim, .FRQ.frq)
    BIM = _ref + ".bim"
    FRQ = _ref + ".FRQ.frq"

    if not exists(BIM):
        raise FileNotFoundError(f'Reference BIM file not found: {BIM}')

    if not exists(FRQ):
        raise FileNotFoundError(f'Reference FRQ file not found: {FRQ}')


    ### Existence - 2: LD matrix (fine-mapping; also the signed-r source for SWCA r/r^2)
    if not (exists(_ref + ".NoNA.PSD.ld") or exists(_ref + ".NoNA.PSD.ld.gz")):
        raise FileNotFoundError(f"Reference LD file not found: {_ref + '.NoNA.PSD.ld'}")


    ### Existence - 3: clumping backend
    ###  - DEFAULT (genotype-free, summary-based): needs the precomputed EM/haplotype r^2 matrix.
    ###  - legacy PLINK genotype path (env SUM2HLA_USE_PLINK): needs the .bed/.fam genotype instead.
    if os.environ.get("SUM2HLA_USE_PLINK"):
        BED = _ref + ".bed"
        FAM = _ref + ".fam"
        if not exists(BED):
            raise FileNotFoundError(f'Reference BED file not found: {BED}')
        if not exists(FAM):
            raise FileNotFoundError(f'Reference FAM file not found: {FAM}')
    else:
        EM_NPY = os.environ.get("SUM2HLA_PY_CLUMP_NPY", _ref + ".EM_haplotype_r2.npy")
        if not exists(EM_NPY):
            raise FileNotFoundError(
                f"Reference EM/haplotype r^2 matrix not found: {EM_NPY}\n"
                f"  It is required for the (default) genotype-free summary-based clumping.\n"
                f"  Provide it at '<ref>.EM_haplotype_r2.npy', or set SUM2HLA_PY_CLUMP_NPY to its path,\n"
                f"  or set SUM2HLA_USE_PLINK=1 to use the legacy PLINK genotype-based clumping instead.")

    return True



def check_outdir(_out_prefix) -> bool:

    if dirname(_out_prefix) == '':
        return True

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