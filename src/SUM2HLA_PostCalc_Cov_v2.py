"""
SUM2HLA_PostCalc_Cov_v2.py  —  Plan A: Rank-1 Update Optimized LL computation

[동기]
기존 SUM2HLA_PostCalc_Cov.py의 N_causal=1 경로는 다음과 같은 구조적 비효율이 있음:
  - K개 causal configuration을 batch_size씩 나눠 JAX 배치 루프 반복
  - 배치마다 (batch_size, K, K) 크기의 희소 diagC 행렬 full allocation 후 matmul 2회
  - 배치마다 O(M³) SVD + O(M³) solve

N_causal=1일 때 각 configuration은 rank-1 update이므로:
  R'_c[SNPs,SNPs] = R_SNP + ncp * v_c @ v_c^T   (v_c = R[SNPs, c])

다음 두 항등식으로 모든 K개 config의 LL을 루프 없이 O(M*K) 전처리 + O(K) 벡터 연산으로 계산:

  ① Matrix Determinant Lemma:
     log det(R'_c) = log det(R_SNP) + log(1 + ncp * v_c^T @ R_SNP^{-1} @ v_c)

  ② Woodbury Identity:
     Z^T @ (R'_c)^{-1} @ Z
       = Z^T @ R_SNP^{-1} @ Z  -  ncp * (u_c^T @ Z)^2 / (1 + ncp * alpha_c)
     여기서 u_c = R_SNP^{-1} @ v_c,  alpha_c = v_c^T @ u_c

[결과]
  - yield_configure_batch() 불필요 → 삭제
  - JAX 배치 루프 불필요 → 삭제
  - 배치당 SVD 불필요 → 삭제
  - INPUT_LDmatrix에 이미 있는 eigendecomposition 재활용

[인터페이스]
  __MAIN__()와 postprepr_LL()의 signature는 기존과 동일 (drop-in replacement).
  SUM2HLA_batch.py에서:
    import src.SUM2HLA_PostCalc_Cov_v2 as mod_PostCal_Cov
  으로 한 줄만 바꾸면 됨.

[제한]
  N_causal >= 2는 미구현. N_causal=1 전용.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

from src.INPUT_LDmatrix import INPUT_LDmatrix
from src.INPUT_GWAS_summary import INPUT_GWAS_summary

# NOTE: calc_PP / postprepr_LL을 원본(SUM2HLA_PostCalc_Cov.py)에서 import하지 않고
# 직접 포함시킴. 원본은 module-level에서 'import jax'를 수행하므로,
# JAX 미설치 환경(ex. MLX-only Mac, CPU-only 서버)에서도 v2.py가 독립적으로 동작하도록.



########## [] Postprocessing (copied from SUM2HLA_PostCalc_Cov.py — no modification)

def calc_PP(_sr_LL_prior):
    """log-sum-exp trick으로 posterior probability 계산."""

    C           = np.max(_sr_LL_prior)
    shifted_exp = np.exp(_sr_LL_prior - C)
    log_sum_exp = C + np.log(np.sum(shifted_exp))

    sr_logPP = (_sr_LL_prior - log_sum_exp).rename("logAPP")
    sr_PP    = pd.Series(np.exp(sr_logPP), name='APP', index=_sr_LL_prior.index)

    return sr_PP, sr_logPP


def postprepr_LL(_df_result, _rho=0.99, _col_CredibleSet="CredibleSet(99%)",
                 _l_type=('whole', 'SNP', 'HLAtype', 'HLA', 'AA', 'intraSNP'),
                 _f_AA_only_positive_pos=True) -> dict:
    """
    LL → PP 변환 + credible set 계산. 마커 타입별 subset.
    원본 SUM2HLA_PostCalc_Cov.postprepr_LL()과 동일.
    """

    print("\nPostprocessing the calculated LLs.")

    df_result_sort = _df_result.sort_values("LL+Lprior", ascending=False)

    f_HLA      = df_result_sort['SNP'].str.match(r'^HLA_\w+_\d{4}$')
    f_HLAh     = df_result_sort['SNP'].str.startswith("HLAh_")  # HLA-haplotype markers
    f_intraSNP = df_result_sort['SNP'] \
                     .map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x)) \
                     .map(lambda x: bool(x))
    if _f_AA_only_positive_pos:
        f_AA = df_result_sort['SNP'].map(lambda x: bool(re.match(r"AA_(\w+)_(\d+)_", x)))
    else:
        f_AA = df_result_sort['SNP'].str.startswith("AA")

    f_HLAtype = f_HLA | f_AA | f_intraSNP | f_HLAh
    f_SNP     = ~f_HLAtype
    f_whole   = f_SNP | f_HLAtype
    f_AA_HLA  = f_AA | f_HLA

    d_flag_target_group = {
        "whole":        f_whole,
        "SNP":          f_SNP,
        "HLAtype":      f_HLAtype,
        "HLA":          f_HLA,
        "AA":           f_AA,
        "intraSNP":     f_intraSNP,
        "AA+HLA":       f_AA_HLA,
        "HLAh":         f_HLAh,
        "HLA+HLAh":     f_HLA | f_HLAh,
        "AA+HLA+HLAh":  f_AA | f_HLA | f_HLAh,
    }

    if len(_l_type) > 0:
        d_flag_target_group = {k: v for k, v in d_flag_target_group.items() if k in _l_type}

    def postprepr_LL_subgroup(_df_sub, _rho):

        l_LL = _df_sub['LL+Lprior'].tolist()
        sr_diff_abs = pd.Series(
            [0.0] + [l_LL[i-1] - l_LL[i] for i in range(1, _df_sub.shape[0])],
            index=_df_sub.index, name='LL+Lprior_diff'
        )

        acc_temp, l_temp = 0.0, []
        for _diff in sr_diff_abs:
            acc_temp += _diff
            l_temp.append(acc_temp)
        sr_diff_abs_acc = pd.Series(l_temp, index=_df_sub.index, name='LL+Lprior_diff_acc')

        sr_PP, sr_logPP = calc_PP(_df_sub['LL+Lprior'])

        def get_credible_set(_sr_PP, _rho):
            if _sr_PP.iat[0] >= _rho:
                return [True] + [False] * (_sr_PP.shape[0] - 1)
            acc_PP, l_cs = 0.0, []
            for _pp in _sr_PP:
                acc_PP += _pp
                l_cs.append(True)
                if acc_PP >= _rho:
                    break
            return l_cs + [False] * (len(_sr_PP) - len(l_cs))

        sr_CredibleSet = pd.Series(
            get_credible_set(sr_PP, _rho), index=sr_PP.index, name=_col_CredibleSet
        )
        sr_rank   = pd.Series(range(_df_sub.shape[0]), name='rank',   index=_df_sub.index)
        sr_rank_p = (sr_rank / _df_sub.shape[0]).rename("rank_p")
        sr_rank   = sr_rank + 1

        return pd.concat(
            [sr_rank, sr_rank_p, _df_sub, sr_PP, sr_CredibleSet,
             sr_diff_abs, sr_diff_abs_acc, sr_logPP], axis=1
        ).loc[:, ['rank', 'rank_p', 'SNP', 'APP', _col_CredibleSet,
                  'LL+Lprior', 'LL+Lprior_diff', 'LL+Lprior_diff_acc', 'logAPP']]

    for _key, _sr_flag in d_flag_target_group.items():
        try:
            d_flag_target_group[_key] = postprepr_LL_subgroup(df_result_sort[_sr_flag], _rho)
        except Exception:
            d_flag_target_group[_key] = None

    return d_flag_target_group



########## [] Core: Rank-1 Update LL Computation

def calc_LL_all_rank1(
    _GWASsummary: INPUT_GWAS_summary,
    _LDmatrix:    INPUT_LDmatrix,
    _LL_0:        float,
    _gamma:       float = 0.01,
    _ncp:         float = 5.2,
) -> tuple:
    """
    N_causal=1 전용. K개 causal configuration의 LL을 루프 없이 한 번에 계산.

    Parameters
    ----------
    _GWASsummary : INPUT_GWAS_summary
        sr_GWAS_summary (M-dim Z-score Series, SNP order = LD SNP order)
    _LDmatrix : INPUT_LDmatrix
        df_LD (K×K), l_ix_SNPs, eigenvalues, eigenvectors, term2 활용
    _LL_0 : float
        Baseline LL (N_causal=0 + Lprior_0), SUM2HLA_batch에서 계산해서 넘겨줌
    _gamma : float
        P(causal) prior per variant
    _ncp : float
        Non-centrality parameter

    Returns
    -------
    arr_PIP_acc : np.ndarray, shape (K,)
        (LL_1(c) - LL_0 + Lprior) per marker c — 기존 __MAIN__ return과 동일 의미
    acc_LL_N_causal : float
        sum(arr_PIP_acc)
    """

    # ── Dimensions ────────────────────────────────────────────────────────────
    K       = _LDmatrix.df_LD.shape[0]             # total markers (SNP + HLA)
    SNP_idx = np.array(_LDmatrix.l_ix_SNPs)        # integer index of SNP markers, shape (M,)

    # ── Prior ─────────────────────────────────────────────────────────────────
    # Lprior = log(gamma) + (K-1)*log(1-gamma)   [N_causal=1]
    Lprior = np.log(_gamma) + (K - 1) * np.log(1.0 - _gamma)

    # ── Eigendecomposition of R_SNP ───────────────────────────────────────────
    # INPUT_LDmatrix에서 이미 np.linalg.eigh()로 계산됨.
    # R_SNP = V @ diag(lam) @ V^T,  R_SNP^{-1} = V @ diag(1/lam) @ V^T
    V   = _LDmatrix.eigenvectors                   # (M, M)
    lam = np.maximum(_LDmatrix.eigenvalues, 1e-12) # (M,) — 수치 안정성

    # ── Data ──────────────────────────────────────────────────────────────────
    Z      = _GWASsummary.sr_GWAS_summary.values   # (M,)
    R_full = _LDmatrix.df_LD.values                # (K, K)

    # ── Precomputation (O(M*K)) ───────────────────────────────────────────────

    # V_full = R_full[SNPs, :]  = [v_0 | v_1 | ... | v_{K-1}]   shape (M, K)
    # v_c = R[SNPs, c] = c번째 column을 SNP index로 subset
    V_full = R_full[SNP_idx, :]                             # (M, K)

    # U = R_SNP^{-1} @ V_full = [u_0 | u_1 | ... | u_{K-1}]   shape (M, K)
    # = V @ diag(1/lam) @ V^T @ V_full
    U = V @ ((V.T @ V_full) / lam[:, None])                 # (M, K)

    # alpha_c = v_c^T @ u_c = diag(V_full^T @ U)   shape (K,)
    alpha = np.sum(V_full * U, axis=0)                      # (K,)

    # b = R_SNP^{-1} @ Z   shape (M,)
    b = V @ ((V.T @ Z) / lam)                               # (M,)

    # Z_b = Z^T @ R_SNP^{-1} @ Z   scalar
    Z_b = float(Z @ b)

    # beta_c = u_c^T @ Z = (U^T @ Z)_c   shape (K,)
    beta = U.T @ Z                                          # (K,)

    # ── LL for all K configurations (fully vectorized) ─────────────────────────
    #
    # LL_1(c) = term2(R_SNP)  — 0.5*log(denom_c)  — 0.5*Z_b  + 0.5*ncp*beta_c^2/denom_c
    # 여기서 term2(R_SNP) = _LDmatrix.term2  (= -0.5*sum(log(eigenvalues)))
    #        denom_c      = 1 + ncp * alpha_c   (Matrix Det. Lemma의 correction factor)
    #        beta_c^2/denom_c                   (Woodbury의 Mahalanobis correction)

    denom    = np.maximum(1.0 + _ncp * alpha, 1e-12)       # (K,) — clip for safety
    LL_1_all = (
        _LDmatrix.term2
        - 0.5 * np.log(denom)
        - 0.5 * Z_b
        + 0.5 * _ncp * (beta ** 2) / denom
    )                                                       # (K,)

    # ── PIP accumulation ──────────────────────────────────────────────────────
    # N_causal=1이면 configuration c = (c,): arr_PIP_acc[c] = LL(c)
    arr_PIP_acc     = LL_1_all - _LL_0 + Lprior            # (K,)
    acc_LL_N_causal = float(np.sum(arr_PIP_acc))

    return arr_PIP_acc, acc_LL_N_causal



########## [] Main wrapper (drop-in replacement for SUM2HLA_PostCalc_Cov.__MAIN__)

def __MAIN__(
    _N_causal,
    _GWASsummary: INPUT_GWAS_summary,
    _LDmatrix:    INPUT_LDmatrix,
    _LL_0,
    _batch_size=None,   # ignored — kept for interface compatibility
    _gamma=0.01,
    _ncp=5.2,
    _engine=None,       # ignored — kept for interface compatibility
):
    """
    SUM2HLA_PostCalc_Cov.__MAIN__()와 동일한 signature의 drop-in replacement.
    N_causal=1 한정. _batch_size, _engine은 무시됨.

    SUM2HLA_batch.py에서 아래 한 줄만 바꾸면 이 모듈로 전환:
        import src.SUM2HLA_PostCalc_Cov_v2 as mod_PostCal_Cov
    """

    if _N_causal != 1:
        raise NotImplementedError(
            f"SUM2HLA_PostCalc_Cov_v2 supports only N_causal=1 (got N_causal={_N_causal}).\n"
            "N_causal >= 2 is not implemented in this version."
        )

    K = _LDmatrix.df_LD.shape[0]
    M = len(_LDmatrix.l_ix_SNPs)

    print(f"\n[v2 / Rank-1 Optimized]  N_causal=1,  K={K} total markers,  M={M} SNPs")
    print("  No batch loop / no SVD per config — single vectorized pass.")

    t_start = datetime.now()

    arr_PIP_acc, acc_LL_N_causal = calc_LL_all_rank1(
        _GWASsummary, _LDmatrix, _LL_0,
        _gamma=_gamma, _ncp=_ncp,
    )

    t_end = datetime.now()
    print(f"  Elapsed: {t_end - t_start}")

    return arr_PIP_acc, acc_LL_N_causal
