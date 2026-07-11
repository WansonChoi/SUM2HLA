#!/usr/bin/env python
"""
Replicate PLINK 1.9 `--clump` using only summary statistics:
    - per-variant P-value (from the ToClump file)
    - per-variant BP position (from the reference .bim)
    - an r (or r^2) LD matrix among the variants

PLINK defaults replicated: --clump-p1 1e-4, --clump-p2 1e-2,
--clump-r2 0.50, --clump-kb 250.

Algorithm (PLINK 1.9 plink_clump.c::clump_reports, no --clump-allow-overlap):
  1. sort variants by P ascending (stable; ties keep ToClump file order)
  2. take most significant unclaimed variant with P < p1 -> index SNP
  3. members = unclaimed variants with P < p2, |BP-BP_idx| <= kb*1000,
     r2(idx, .) >= r2_thresh
  4. mark index + members as claimed; repeat
"""
import argparse, sys
import numpy as np
import pandas as pd


def load_ld_subset(fpath_ld, fpath_bim, snps_needed, is_r2=False):
    """Load the LD submatrix (as r^2) restricted to `snps_needed`.

    Reads the square LD matrix line-by-line, keeping only required rows/cols.
    Row/col order of the matrix == order of variants in the .bim.
    Returns (snp_list, r2_matrix) where snp_list is the panel order of the
    kept variants.
    """
    df_bim = pd.read_csv(fpath_bim, sep='\t', header=None,
                         names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])
    panel_snps = df_bim['SNP'].tolist()
    snp2idx = {s: i for i, s in enumerate(panel_snps)}

    need = set(snps_needed)
    keep_idx = [i for i, s in enumerate(panel_snps) if s in need]   # panel order
    keep_set = set(keep_idx)
    # map global panel index -> position in kept submatrix
    pos_in_sub = {gi: k for k, gi in enumerate(keep_idx)}
    n = len(keep_idx)
    R = np.empty((n, n), dtype=np.float64)

    with open(fpath_ld) as f:
        for i, line in enumerate(f):
            if i not in keep_set:
                continue
            vals = np.fromstring(line, sep='\t')
            sub = vals[keep_idx]          # pick needed columns
            R[pos_in_sub[i], :] = sub
    if not is_r2:
        R = R * R
    kept_snps = [panel_snps[i] for i in keep_idx]
    kept_bp = df_bim['BP'].values[keep_idx]
    return kept_snps, kept_bp, R


def clump(snps, P, BP, R2, p1=1e-4, p2=1e-2, r2_thresh=0.50, kb=250):
    """Return list of clumps. Each clump: dict(index, members[list of idx into arrays]).
    `snps`, `P`, `BP` indexed identically; R2 is len(snps) x len(snps)."""
    n = len(snps)
    kb_bp = kb * 1000
    # stable sort by P -> ties keep original (ToClump) order
    order = np.argsort(P, kind='stable')
    claimed = np.zeros(n, dtype=bool)
    clumps = []
    for idx in order:
        # PLINK --clump-p1 is INCLUSIVE: a variant with P == p1 is still an index
        # candidate (confirmed on Consortium MS, rs4959053 P=1e-4). Break only when
        # strictly above p1 (order is P-ascending, so all remaining are too).
        if P[idx] > p1:
            break
        if claimed[idx]:
            continue
        # candidate members: ALL unclaimed variants in LD+kb (any P).
        # PLINK claims the whole r2+kb neighborhood; SP2 displays only P<p2,
        # while TOTAL/NSIG/S05/... bin the full neighborhood by P.
        within = np.abs(BP - BP[idx]) <= kb_bp
        ld_ok = R2[idx] >= r2_thresh
        cand = within & ld_ok & (~claimed)
        cand[idx] = False
        members = np.where(cand)[0].tolist()
        claimed[idx] = True
        claimed[members] = True
        clumps.append({'index': int(idx), 'members': members})
    return clumps


def plink_p(p):
    """Format a P-value like PLINK's .clumped column: 3 significant digits,
    lowercase 'e', signed >=2-digit exponent, trailing zeros stripped.
    (Round-half-even; PLINK's David-Gay dtoa rounds a handful of messy-double
    inputs differently in the 3rd sig-fig -- display only, see notes.)"""
    s = '%.3g' % float(p)
    if 'e' in s:
        mant, exp = s.split('e')
        ei = int(exp)
        return "%se%s%02d" % (mant, '-' if ei < 0 else '+', abs(ei))
    return s


def write_clumped(clumps, snps, P, BP, df_bim_full, out_path, p2=1e-2):
    """Write a PLINK-format .clumped file (whitespace, right-justified)."""
    snp2chr = dict(zip(df_bim_full['SNP'], df_bim_full['CHR']))
    rows = []
    for f_idx, c in enumerate(clumps, start=1):
        i = c['index']
        mem = c['members']
        # bins over the FULL neighborhood (PLINK semantics)
        pm = P[mem]
        nsig = int(np.sum(pm > 0.05))
        s05 = int(np.sum((pm > 0.01) & (pm <= 0.05)))
        s01 = int(np.sum((pm > 0.001) & (pm <= 0.01)))
        s001 = int(np.sum((pm > 0.0001) & (pm <= 0.001)))
        s0001 = int(np.sum(pm <= 0.0001))
        # SP2: only members with P < clump-p2, ordered by BP ascending
        mem_sp2 = sorted([m for m in mem if P[m] < p2], key=lambda m: BP[m])
        sp2 = ",".join("{}(1)".format(snps[m]) for m in mem_sp2) if mem_sp2 else "NONE"
        rows.append({
            'CHR': snp2chr.get(snps[i], 6), 'F': 1, 'SNP': snps[i],
            'BP': BP[i], 'P': P[i], 'TOTAL': len(mem),
            'NSIG': nsig, 'S05': s05, 'S01': s01, 'S001': s001, 'S0001': s0001,
            'SP2': sp2,
        })
    df = pd.DataFrame(rows)
    cols = ['CHR', 'F', 'SNP', 'BP', 'P', 'TOTAL', 'NSIG', 'S05', 'S01', 'S001', 'S0001', 'SP2']
    with open(out_path, 'w') as fo:
        fo.write(" %3s %4s %24s %10s %8s %8s %6s %6s %6s %6s %6s %s\n" %
                 tuple(cols))
        for _, r in df.iterrows():
            fo.write(" %3d %4d %24s %10d %8s %8d %6d %6d %6d %6d %6d %s\n" % (
                int(r['CHR']), int(r['F']), r['SNP'], int(r['BP']), plink_p(r['P']),
                int(r['TOTAL']), int(r['NSIG']), int(r['S05']), int(r['S01']),
                int(r['S001']), int(r['S0001']), r['SP2']))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--toclump', required=True)
    ap.add_argument('--ld', required=True)
    ap.add_argument('--bim', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--is-r2', action='store_true', help='LD matrix already r^2 (else signed r)')
    ap.add_argument('--p1', type=float, default=1e-4)
    ap.add_argument('--p2', type=float, default=1e-2)
    ap.add_argument('--r2', type=float, default=0.50)
    ap.add_argument('--kb', type=float, default=250)
    args = ap.parse_args()

    tc = pd.read_csv(args.toclump, sep='\t')
    tc = tc.dropna(subset=['P']).reset_index(drop=True)
    df_bim_full = pd.read_csv(args.bim, sep='\t', header=None,
                              names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])

    snps, bp, R2 = load_ld_subset(args.ld, args.bim, tc['SNP'].tolist(), is_r2=args.is_r2)
    # `snps` are in panel (bim/bfile) order -- PLINK breaks P-value ties by that order
    p_map = dict(zip(tc['SNP'], tc['P']))
    P = np.array([p_map[s] for s in snps], dtype=np.float64)
    BP = np.array(bp, dtype=np.float64)

    clumps = clump(snps, P, BP, R2, p1=args.p1, p2=args.p2,
                   r2_thresh=args.r2, kb=args.kb)
    write_clumped(clumps, snps, P, BP, df_bim_full, args.out)
    print("clumps formed: {}  (from {} variants with P<{:g})".format(
        len(clumps), int(np.sum(P < args.p1)), args.p1))
    print("written:", args.out)


if __name__ == '__main__':
    main()
