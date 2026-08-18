"""
Summary-based (genotype-free) replacement for src/SWCA_calc_r_r2.py.

The original `calc_r_r2` shells out to `plink --r/--r2 --ld-snp <top> --bfile <ref>`
to get the GENOTYPIC signed-r (and r^2) between each round's index/top marker and its
clumped members. This module reproduces those values by LOOKUP into a precomputed
**signed-r square matrix** (the raw `plink --r square` output), so no genotype bfile
is needed at runtime.

Notes on exactness (validated on WTCCC RA/T1D):
  - Use the RAW `--r square` matrix (`LD.<ref>.ld`). The PSD-clipped `.NoNA.PSD.ld`
    differs at ~4th decimal and does NOT reproduce PLINK exactly.
  - r^2 = r*r (PLINK's --r2 equals the square of --r to printed precision).
  - `--ld-window-r2 0.2`: pairs with genotypic r^2 < 0.2 are NOT emitted by PLINK →
    they become NaN here too.
  - This matrix r is genotypic signed-r (NOT the EM/haplotype r^2 used for clumping).

Everything else (round/clumped-file bookkeeping) is reused unchanged from the original
module so behavior stays identical.
"""

import os, re, json, gzip
from os.path import dirname, basename, join, exists
import numpy as np
import pandas as pd

from src.SWCA_calc_r_r2 import (
    get_SWCA_Each_Round_Clumped_Files,
    get_clumped_markers_of_a_marker_v2,
)

R2_WINDOW_MIN = 0.2   # mirrors PLINK `--ld-window-r2 0.2` used by the original calc_r_r2


def _open_ld(_fpath_ld):
    """Open a square matrix that may be plain text or gzipped.
    Everything else on this path goes through `pandas.read_csv`, which already
    decompresses by extension; these two readers stream the file by hand, so they
    need this. The reference panel shipped in example/ is gzipped."""
    return gzip.open(_fpath_ld, 'rt') if _fpath_ld.endswith('.gz') else open(_fpath_ld)


def _detect_header(_fpath_ld):
    """Return True if the LD matrix file has a leading SNP-name header row.
    Raw `plink --r square` output is header-less (all-numeric first line);
    matrices written by src/Util.py carry a SNP-name header (index=False)."""
    with _open_ld(_fpath_ld) as f:
        toks = f.readline().rstrip('\n').split('\t')
    try:
        float(toks[0])
        return False
    except ValueError:
        return True


def load_signed_r_rows(_fpath_ld, _panel_snps, _needed_rows, _needed_cols):
    """Read ONLY the rows for `_needed_rows` from a signed-r square matrix.

    Returns {row_snp: {col_snp: signed_r}} for col_snp in `_needed_cols`.
    Supports header-less (raw plink, panel/.bim order) and header-present
    (index=False) square matrices. The matrix is symmetric; row/col order == bim order.
    """
    has_header = _detect_header(_fpath_ld)

    with _open_ld(_fpath_ld) as f:
        if has_header:
            hdr = f.readline().rstrip('\n').split('\t')
            row_names = hdr
            colpos = {s: i for i, s in enumerate(hdr)}
        else:
            row_names = _panel_snps
            colpos = {s: i for i, s in enumerate(_panel_snps)}

        col_idx = {c: colpos[c] for c in _needed_cols if c in colpos}
        need_rows = {s for s in _needed_rows if s in colpos}

        out = {}
        for i, line in enumerate(f):
            if i >= len(row_names):
                break
            rn = row_names[i]
            if rn not in need_rows:
                continue
            vals = np.fromstring(line, sep='\t')
            out[rn] = {c: vals[j] for c, j in col_idx.items()}
    return out


def _fmt6g(_x):
    """Match PLINK's numeric output: 6 significant figures (as PLINK prints --r/--r2)."""
    return float("%.6g" % _x)


def _pair_r_r2(_r):
    """(r, r2) with the r^2>=0.2 window filter and NaN propagation, matching PLINK.
    Values are rounded to 6 significant figures to mirror PLINK's printed --r/--r2
    (so the exported .SWCA.dict is byte-identical to the genotype-based run)."""
    if _r is None or not np.isfinite(_r):
        return np.nan, np.nan
    r2 = _r * _r
    if not np.isfinite(r2) or r2 < R2_WINDOW_MIN:
        return np.nan, np.nan
    return _fmt6g(_r), _fmt6g(r2)


def __MAIN__(_fpath_SWCA_out_dict, _out_dir_clumped, _fpath_signed_r_ld, _fpath_ref_bim,
             _f_old=False):
    """Drop-in for SWCA_calc_r_r2.__MAIN__, but sources r/r^2 from a precomputed
    signed-r matrix (`_fpath_signed_r_ld`) instead of `plink --r --bfile`.

    _fpath_ref_bim : the reference .bim (panel SNP order, for header-less matrices).
    Returns (d_step1, d_step2) exactly like the original.
    """
    ##### (0) load data
    if isinstance(_fpath_SWCA_out_dict, str):
        with open(_fpath_SWCA_out_dict, 'r') as f_dict:
            d_SWCA_out = json.load(f_dict)
    elif isinstance(_fpath_SWCA_out_dict, dict):
        d_SWCA_out = _fpath_SWCA_out_dict.copy()
    else:
        raise Exception("Wrong dictionary!")

    d_clumped_files = get_SWCA_Each_Round_Clumped_Files(_out_dir_clumped)

    if _f_old:
        d_SWCA_out = {k: v for i, (k, v) in enumerate(d_SWCA_out.items()) if i > 0}
        d_clumped_files = {(k + 1): v for k, v in d_clumped_files.items()}

    panel_snps = pd.read_csv(_fpath_ref_bim, sep='\t', header=None,
                             names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])['SNP'].tolist()

    # Round 1 제외
    d_SWCA_out_2 = {k: v for i, (k, v) in enumerate(d_SWCA_out.items()) if i > 0}

    ##### step1: index marker -> its clumped members (from each round's .clumped)
    d_RETURN_step1 = {}
    for _ROUND_N, _l_markers in d_SWCA_out_2.items():
        if _ROUND_N not in d_clumped_files:
            print(f"[WARNING] Clumped file for {_ROUND_N} not found. Skipping...")
            continue
        _fpath_clumped = d_clumped_files[_ROUND_N]
        df_temp_clumped = pd.read_csv(_fpath_clumped, sep=r'\s+', header=0)

        d_temp = {}
        for _marker in _l_markers:
            d_temp.update(get_clumped_markers_of_a_marker_v2(_marker, df_temp_clumped))
        d_RETURN_step1[_ROUND_N] = d_temp.copy()

    ##### step2: annotate each (index, member) with r/r^2 via matrix lookup
    d_RETURN_step2 = {}
    for _ROUND_N, _d_clumped in d_RETURN_step1.items():
        # all index markers (rows) and all members (cols) needed this round
        needed_rows = list(_d_clumped.keys())
        needed_cols = sorted({m for members in _d_clumped.values() for m in members})
        R_rows = load_signed_r_rows(_fpath_signed_r_ld, panel_snps, needed_rows, needed_cols)

        d_temp = {}
        for _marker, _l_clumped in _d_clumped.items():
            row = R_rows.get(_marker, {})
            l_RETURN = {}
            for m in _l_clumped:
                r, r2 = _pair_r_r2(row.get(m, np.nan))
                l_RETURN[m] = {"r": r, "r2": r2}
            d_temp[_marker] = l_RETURN
        d_RETURN_step2[_ROUND_N] = d_temp.copy()

    ##### step3: prepend ROUND_1 (has no clumping to reference)
    d_RETURN_step1_2 = {"ROUND_1": d_SWCA_out["ROUND_1"]}
    d_RETURN_step2_2 = {"ROUND_1": d_SWCA_out["ROUND_1"]}
    d_RETURN_step1_2.update(d_RETURN_step1)
    d_RETURN_step2_2.update(d_RETURN_step2)

    return d_RETURN_step1_2, d_RETURN_step2_2
