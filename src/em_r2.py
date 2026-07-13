#!/usr/bin/env python
"""
EM haplotype-frequency-based r^2 between biallelic variants -- the statistic
PLINK uses internally for `--clump` and reports via `--ld` (NOT the genotypic
`--r2`).  Computed directly from .bed genotypes, pairwise-complete on missing.

For a SNP pair, only the double heterozygote is phase-ambiguous; a standard
2-locus EM resolves it.  All pairs are processed in a vectorized fashion.
"""
import numpy as np
import pandas as pd


def read_bed_subset(bed_prefix, snp_subset):
    """Return (snps_kept_in_panel_order, dosage MxN float32 with np.nan missing)."""
    bim = pd.read_csv(bed_prefix + ".bim", sep='\t', header=None,
                      names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])
    n_samp = sum(1 for _ in open(bed_prefix + ".fam"))
    bpv = (n_samp + 3) // 4
    snp2idx = {s: i for i, s in enumerate(bim['SNP'])}
    need = [s for s in bim['SNP'] if s in set(snp_subset)]  # panel order
    code2dos = np.array([0, np.nan, 1, 2], dtype=np.float32)  # 00,01,10,11
    M = len(need)
    G = np.empty((M, n_samp), dtype=np.float32)
    with open(bed_prefix + ".bed", "rb") as f:
        for k, s in enumerate(need):
            v = snp2idx[s]
            f.seek(3 + v * bpv)
            raw = np.frombuffer(f.read(bpv), dtype=np.uint8)
            bits = np.empty(bpv * 4, dtype=np.uint8)
            bits[0::4] = raw & 3
            bits[1::4] = (raw >> 2) & 3
            bits[2::4] = (raw >> 4) & 3
            bits[3::4] = (raw >> 6) & 3
            G[k, :] = code2dos[bits[:n_samp]]
    bp = bim.set_index('SNP').loc[need, 'BP'].values
    return need, G, bp


def em_r2_matrix(G, n_iter=60):
    """G: M x N dosage (0/1/2, np.nan=missing). Returns MxM EM r^2."""
    M, N = G.shape
    # indicator matrices per dosage (missing -> all zero -> excluded pairwise)
    I0 = (G == 0).astype(np.float32)
    I1 = (G == 1).astype(np.float32)
    I2 = (G == 2).astype(np.float32)
    # joint genotype counts n_ij[s,t] = #samples with dosage i at s and j at t
    n00 = I0 @ I0.T; n01 = I0 @ I1.T; n02 = I0 @ I2.T
    n10 = I1 @ I0.T; n11 = I1 @ I1.T; n12 = I1 @ I2.T
    n20 = I2 @ I0.T; n21 = I2 @ I1.T; n22 = I2 @ I2.T
    n00 = n00.astype(np.float64); n01 = n01.astype(np.float64); n02 = n02.astype(np.float64)
    n10 = n10.astype(np.float64); n11 = n11.astype(np.float64); n12 = n12.astype(np.float64)
    n20 = n20.astype(np.float64); n21 = n21.astype(np.float64); n22 = n22.astype(np.float64)
    # dosage = count of allele A1 (reference). haplotypes: 11=AB,12=Ab,21=aB,22=ab
    N11 = 2 * n22 + n21 + n12      # wait: with dosage=#A1, (2,2)=A1A1/A1A1 -> AB,AB
    N12 = 2 * n20 + n21 + n10
    N21 = 2 * n02 + n12 + n01
    N22 = 2 * n00 + n10 + n01
    D = n11                         # double heterozygotes (ambiguous)
    Nind = n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22  # complete-pair count
    tot2 = 2 * Nind
    f = np.full_like(D, 0.5)
    for _ in range(n_iter):
        a = N11 + D * f            # AB
        d = N22 + D * f            # ab
        b = N12 + D * (1 - f)      # Ab
        c = N21 + D * (1 - f)      # aB
        num = a * d
        den = a * d + b * c
        with np.errstate(invalid='ignore', divide='ignore'):
            f_new = np.where(den > 0, num / den, 0.5)
        f = f_new
    a = N11 + D * f; d = N22 + D * f; b = N12 + D * (1 - f); c = N21 + D * (1 - f)
    with np.errstate(invalid='ignore', divide='ignore'):
        p11 = a / tot2; p12 = b / tot2; p21 = c / tot2; p22 = d / tot2
        pA = p11 + p12; pB = p11 + p21
        Dld = p11 * p22 - p12 * p21
        denom = pA * (1 - pA) * pB * (1 - pB)
        r2 = np.where(denom > 0, Dld * Dld / denom, 0.0)
    r2 = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, 1.0)
    np.fill_diagonal(r2, 1.0)
    return r2


def wrapper_make_EM_haplotype_r2_matrix(_bed_prefix, _out_npy):
    """Build the panel-wide EM/haplotype r^2 matrix from a PLINK bfile and save it as a
    .npy (row/col order == the reference .bim order; diagonal = 1). This matrix is the
    clumping input for the genotype-free (summary-based) SUM2HLA pipeline — it must be
    placed at `<ref>.EM_haplotype_r2.npy` (or pointed to via env SUM2HLA_PY_CLUMP_NPY).
    Returns (out_path, R2)."""
    bim = pd.read_csv(_bed_prefix + ".bim", sep='\t', header=None,
                      names=['CHR', 'SNP', 'GD', 'BP', 'A1', 'A2'])
    snps, G, bp = read_bed_subset(_bed_prefix, bim['SNP'].tolist())
    assert snps == bim['SNP'].tolist(), "marker order mismatch between .bed and .bim"
    R2 = em_r2_matrix(G)
    np.save(_out_npy, R2)
    return _out_npy, R2


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description="Build the EM/haplotype r^2 matrix (SUM2HLA summary-based clumping input).")
    ap.add_argument('--bfile', help="reference PLINK bfile prefix (.bed/.bim/.fam)")
    ap.add_argument('--out', help="output .npy path (recommended: <ref>.EM_haplotype_r2.npy)")
    args = ap.parse_args()

    if args.bfile and args.out:
        out, R2 = wrapper_make_EM_haplotype_r2_matrix(args.bfile, args.out)
        print(f"written: {out}  shape={R2.shape}  diag={np.diag(R2)[:2]}  "
              f"sym_err={np.abs(R2 - R2.T).max():.1e}")
    else:
        # quick self-test against PLINK --ld values
        bedp = "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA"
        test = ['rs9268844', 'rs3806157', 'rs9262582', 'rs9263600', 'rs9268494', 'rs9268497']
        snps, G, bp = read_bed_subset(bedp, test)
        R2 = em_r2_matrix(G)
        pos = {s: i for i, s in enumerate(snps)}
        for a, b, exp in [('rs9268844', 'rs3806157', 0.502944),
                          ('rs9262582', 'rs9263600', 0.520921),
                          ('rs9268844', 'rs9268494', 0.507579),
                          ('rs9268844', 'rs9268497', 0.507538)]:
            print(f"{a}-{b}: EM_py={R2[pos[a],pos[b]]:.6f}  PLINK_ld={exp:.6f}")
