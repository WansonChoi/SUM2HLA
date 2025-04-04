import os, sys, re
import numpy as np
import pandas as pd
import math
import json
from datetime import datetime

from itertools import combinations

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax


from src.mod_LDmatrix_class import LDmatrix
from src.mod_GWAS_summary import GWAS_summary


########## [] Causal configuration generation 

def yield_configure_batch(M, P, batch_size, _f_as_list=False):
    """P개의 요소를 선택하는 M개의 binary configure를 batch 단위로 yield."""
    
    if P > M or P < 0:
        raise ValueError("P must be between 0 and M inclusive.")
    
    iterator = combinations(range(M), P)  # 조합 생성기
    batch = []  # 배치를 저장할 리스트
    
    for comb in iterator:
        if _f_as_list:
            binary_list = [0] * M
            for index in comb:
                binary_list[index] = 1
            batch.append(binary_list)
        else:
            batch.append(comb)

        # batch_size에 도달하면 yield
        if len(batch) == batch_size:
            yield batch
            batch = []  # 새로운 batch 초기화

    # 남은 데이터가 있다면 추가로 yield
    if batch:
        yield batch

"""
(Mid-conclusion)
- combination tuple로 다루도록 수정할거임 => 더 memory efficient
    - combination tuples => N_causal x batch_size
    - combination list => M x batch_size
- JAX같은걸로 vectorized computing할 때, configutation batch만큼은 memory위에 올려놓고 시작해야 함.

"""





########## [] Numpy for log-Likelihood

def generate_LD_matrices(_LDmatrix, _l_batch_configures, _ncp=5.2):
    """
    벡터 연산을 활용하여 batch_size 개의 LD 행렬을 한꺼번에 계산.
    """
    batch_size = len(_l_batch_configures)  # 배치 크기
    K = _LDmatrix.df_LD.shape[0]  # 전체 행렬 크기
    M = len(_LDmatrix.l_ix_SNPs)  # Subset 크기

    # ✅ (1) diagC 생성: (batch_size, K, K)
    diagC = np.zeros((batch_size, K, K))

    # ✅ (2) Configure를 이용해 diagC의 diagonal 값 설정
    for i, configure in enumerate(_l_batch_configures):
        diagC[i, configure, configure] = abs(_ncp)

    # ✅ (3) R을 브로드캐스팅하여 벡터 연산 적용
    R = _LDmatrix.df_LD.values  # (K, K)
    R_new = R + np.matmul(np.matmul(R, diagC), R)  # (batch_size, K, K)

    # ✅ (4) Subsetting (batch-wise로 (K, K) → (M, M) 변환)
    SNP_indices = np.array(_LDmatrix.l_ix_SNPs)  # (M,)
    LD_matrices = R_new[:, SNP_indices[:, None], SNP_indices]  # (batch_size, M, M)

    return LD_matrices



def check_and_fix_psd_batch(LD_matrices, threshold=1e-10):
    """Batch-wise PSD 변환"""
    # 대칭 행렬 보정
    LD_matrices = (LD_matrices + LD_matrices.transpose(0, 2, 1)) / 2  # (batch_size, M, M)

    # Batch-wise 고유값 & 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(LD_matrices)  # (batch_size, M), (batch_size, M, M)

    # 음수 고유값을 threshold로 치환
    eigenvalues_fixed = np.maximum(eigenvalues, threshold)  # (batch_size, M)

    # 행렬 복원 (V * Λ * V.T)
    LD_matrices_fixed = np.einsum('bik,bk,bjk->bij', eigenvectors, eigenvalues_fixed, eigenvectors)  # (batch_size, M, M)

    return eigenvalues_fixed, LD_matrices_fixed



def calc_LL_given_batch_numpy_optimized(_l_batch_configures, _GWASsummary, _LDmatrix, _Lprior, _LL_0, _ncp=5.2):

    # ✅ (1) 벡터 연산으로 LD matrices 생성
    t_start = datetime.now()
    LD_matrices = generate_LD_matrices(_LDmatrix, _l_batch_configures, _ncp)  # (batch_size, M, M)
    t_end = datetime.now()
    print("✅ (1) 벡터 연산으로 LD matrices 생성: {}".format(t_end - t_start))

    
    # ✅ (2) Batch-wise PSD 변환 적용
    t_start = datetime.now()
    # eigenvalues_fixed, LD_matrices_fixed = check_and_fix_psd_batch(LD_matrices)

    eigenvalues = np.linalg.svd(LD_matrices, compute_uv=False) # Eigen-values만 계산.
    f_need_psd_fix = np.any(eigenvalues < 0, axis=1) # <0인 애가 하나라도 있으면 PSD_fix해야 함.

    if np.any(f_need_psd_fix):
        print(f"⚠️ {np.sum(f_need_psd_fix)} 개의 행렬에서 PSD 변환이 필요합니다!")
        print(f_need_psd_fix)
        
        # **PSD 변환이 필요한 행렬만 수정**
        eigenvalues_temp, LD_matrices_temp = check_and_fix_psd_batch(LD_matrices[f_need_psd_fix])  
        
        # **해당 위치의 eigenvalues와 행렬 업데이트**
        eigenvalues[f_need_psd_fix] = eigenvalues_temp  # (batch_size, M)에서 필요한 부분만 수정
        LD_matrices[f_need_psd_fix] = LD_matrices_temp  # (batch_size, M, M)에서도 업데이트

    # 변수 명들만 re-labeling (No Copy)
    eigenvalues_fixed = eigenvalues
    LD_matrices_fixed = LD_matrices
    
    
    t_end = datetime.now()
    print("✅ (2) Batch-wise PSD 변환 적용: {}".format(t_end - t_start))

    # ✅ (3) Batch-wise log-determinant 계산
    t_start = datetime.now()
    term2_values = -0.5 * np.sum(np.log(eigenvalues_fixed), axis=1)  # (batch_size,)
    t_end = datetime.now()
    print("✅ (3) Batch-wise log-determinant 계산: {}".format(t_end - t_start))

    # ✅ (4) GWAS summary 변환
    t_start = datetime.now()
    GWAS_summary = _GWASsummary.sr_GWAS_summary.values  # (M,)
    GWAS_summary_batched = np.broadcast_to(GWAS_summary, (LD_matrices.shape[0], len(GWAS_summary)))  # (batch_size, M)
    t_end = datetime.now()
    print("✅ (4) GWAS summary 변환: {}".format(t_end - t_start))

    # ✅ (5) Solve multiple linear systems in batch
    t_start = datetime.now()
    solutions = np.linalg.solve(LD_matrices_fixed, GWAS_summary_batched)  # (batch_size, M)
    t_end = datetime.now()
    print("✅ (5) Solve multiple linear systems in batch: {}".format(t_end - t_start))

    # ✅ (6) Compute Mahalanobis distance using einsum
    t_start = datetime.now()
    LL_1 = term2_values - 0.5 * np.einsum('bi,bi->b', GWAS_summary_batched, solutions)
    t_end = datetime.now()
    print("✅ (6) Compute Mahalanobis distance using einsum: {}".format(t_end - t_start))

    # ✅ (7) Compute final log-likelihood
    return (LL_1 - _LL_0) + _Lprior



## 완전 raw하게 짰던거 (derepcated)
def calc_LL_given_batch_numpy(_l_batch_configures, _GWASsummary:GWAS_summary, _LDmatrix:LDmatrix, _Lprior, _LL_0):

    iter_LL_1_items = map(lambda x: _LDmatrix.calc_new_cov(x), _l_batch_configures)

    def calc_LL(_term2, _arr_LD_SNP, _arr_GWAS_summary, _LL_0, _Lprior):
        
        LL_1 = _term2 + -0.5 * ( _arr_GWAS_summary.T @ (np.linalg.solve(_arr_LD_SNP, _arr_GWAS_summary)) )
        
        return (LL_1 - _LL_0) + _Lprior

    l_LL = [
        calc_LL(_term2, _df_LD_SNP.values, _GWASsummary.sr_GWAS_summary.values, _LL_0, _Lprior) \
            for _term2, _df_LD_SNP in iter_LL_1_items
    ]

    return l_LL



########## [] JAX for log-Likelihood

@jit
def generate_LD_matrices_jax(_l_batch_configures:jnp.array, _df_LD:jnp.array, _l_ix_SNPs:jnp.array, _ncp=5.2):

    """
    - @jit을 적용받을 수 있도록 수정해봄.
    - argument로 받을때 jnp.array여야 @jit이 효과를 본다고 함.
    """
    
    batch_size = len(_l_batch_configures)  # 배치 크기
    K = _df_LD.shape[0]  # 전체 행렬 크기
    M = len(_l_ix_SNPs)  # Subset 크기
    
    # ✅ (1) diagC 생성: (batch_size, K, K)
    diagC = jnp.zeros((batch_size, K, K))

    # ✅ (2) Configure를 이용해 diagC의 diagonal 값 설정
    batch_indices = jnp.arange(batch_size)[:, None]  # (batch_size, 1)
    configure_indices = _l_batch_configures  # (batch_size, N_causal) 형태

    # ✅ JAX 방식으로 diagonal 값 설정 (Numpy의 for-loop 대체)
    diagC = diagC.at[batch_indices, configure_indices, configure_indices].set(abs(_ncp))

    # ✅ (3) R을 브로드캐스팅하여 벡터 연산 적용
    R = _df_LD  # (K, K) / numpy array를 jnp array로 변환.
    R_new = R + jnp.matmul(jnp.matmul(R, diagC), R)  # (batch_size, K, K)

    # ✅ (4) Subsetting (batch-wise로 (K, K) → (M, M) 변환)
    SNP_indices = _l_ix_SNPs  # (M,)
    LD_matrices = R_new[:, SNP_indices[:, None], SNP_indices]  # (batch_size, M, M)

    return LD_matrices



@jit
def check_and_fix_psd_batch_jax(LD_matrices, threshold=1e-10):
    """Batch-wise PSD 변환"""
    # 대칭 행렬 보정
    LD_matrices = (LD_matrices + LD_matrices.transpose(0, 2, 1)) / 2  # (batch_size, M, M)
    
    # Batch-wise 고유값 & 고유벡터 계산
    eigenvalues, eigenvectors = jnp.linalg.eigh(LD_matrices)  # (batch_size, M), (batch_size, M, M)
    
    # 음수 고유값을 threshold로 치환
    eigenvalues_fixed = jnp.maximum(eigenvalues, threshold)  # (batch_size, M)
    
    # 행렬 복원 (V * Λ * V.T)
    LD_matrices_fixed = jnp.einsum('bik,bk,bjk->bij', eigenvectors, eigenvalues_fixed, eigenvectors)  # (batch_size, M, M)
    
    return eigenvalues_fixed, LD_matrices_fixed



@jit
def calc_LL_given_batch_jax(_GWASsummary_jax:jnp.array, _LDmatrix_jax:jnp.array, _Lprior, _LL_0, _ncp=5.2):

    LD_matrices = _LDmatrix_jax
    
    # ✅ (2) Batch-wise PSD 변환 적용
    eigenvalues = jnp.linalg.svd(LD_matrices, compute_uv=False)
    # f_need_psd_fix = jnp.any(eigenvalues < 0, axis=1)

    eigenvalues_fixed = eigenvalues 
    LD_matrices_fixed = LD_matrices
    
    # ✅ Boolean Masking 대신 `lax.select()`로 적용
    # def fix_psd(LD_matrices, eigenvalues):
    #     # ✅ 모든 행렬을 대상으로 PSD 변환을 적용하는 대신, 필요 없는 곳에는 원본 유지
    #     eigenvalues_temp, LD_matrices_temp = check_and_fix_psd_batch_jax(LD_matrices)

    #     # ✅ Boolean Mask의 Shape을 맞춰서 적용
    #     mask_eigenvalues = f_need_psd_fix[:, None].repeat(eigenvalues.shape[1], axis=1)
    #     mask_matrices = f_need_psd_fix[:, None, None].repeat(LD_matrices.shape[1], axis=1).repeat(LD_matrices.shape[2], axis=2)

    #     eigenvalues_fixed = lax.select(mask_eigenvalues, eigenvalues_temp, eigenvalues)
    #     LD_matrices_fixed = lax.select(mask_matrices, LD_matrices_temp, LD_matrices)

    #     return eigenvalues_fixed, LD_matrices_fixed

    # # ✅ 조건에 따라 PSD 변환 적용
    # eigenvalues_fixed, LD_matrices_fixed = lax.cond(
    #     jnp.any(f_need_psd_fix), 
    #     lambda _: fix_psd(LD_matrices, eigenvalues), 
    #     lambda _: (eigenvalues, LD_matrices),
    #     operand=None
    # )    

    
    # ✅ (3) Batch-wise log-determinant 계산
    term2_values = -0.5 * jnp.sum(jnp.log(eigenvalues_fixed), axis=1)
    
    # ✅ (4) GWAS summary 변환
    GWAS_summary_batched = jnp.broadcast_to(_GWASsummary_jax, (LD_matrices.shape[0], len(_GWASsummary_jax)))
    
    # ✅ (5) Solve multiple linear systems in batch
    solutions = jnp.linalg.solve(LD_matrices_fixed, GWAS_summary_batched[..., None])[..., 0]
    
    # ✅ (6) Compute Mahalanobis distance using einsum
    LL_1 = term2_values - 0.5 * jnp.einsum('bi,bi->b', GWAS_summary_batched, solutions)
    
    # ✅ (7) Compute final log-likelihood
    return (LL_1 - _LL_0) + _Lprior
    

    
########## [] Main wrapper of this '*.py' file.

def __MAIN__(_N_causal, _GWASsummary:GWAS_summary, _LDmatrix:LDmatrix, _LL_0,
                           _batch_size=100, _gamma=0.01, _ncp=5.2, _engine="numpy"):

    """
    - (1) iter on batches of configures.
    - (2) Given a batch of configure, calc LL of the batch configures.
        - 여기를 `calc_LL_given_batch_XX()`로 가져가서 계산.
        - (3) subtract the LL_0
        - (4) add the Lprior
    - (5) accumulate the LL to each SNP (PIP 계산)
    
    """

    ##### (1) Main variables
    M = _LDmatrix.df_LD.shape[0]
    Lprior = _N_causal*np.log(_gamma) + (M - _N_causal)*np.log(1-_gamma)
        # Lprior는 N_causal이 given되는 지금 계산하는게 나을 듯.
    
    ## accumulation
    arr_PIP_acc = np.zeros(M)
    acc_LL_N_causal = 0.0 # `N_causal`이 주어졌을 때의 LL 누적. 나중에 N_causal이 몇일때 가장 LL이 높게 나오는지도 알고싶음.


    
    ##### (2) Main iteration
    
    iter_batch_configures = yield_configure_batch(M, _N_causal, _batch_size, _f_as_list=False)

    for i, _batch_configures in enumerate(iter_batch_configures):

        if i % 100 == 0: 
            print("=====[{}]: {}-th batch / First 5 items: {} ({})".format(i, i, _batch_configures[:5], datetime.now()))
            # print("First 5 items: {}".format(_batch_configures[:5]))
            # display(_batch_configures)

        ### (2-1) calc LL for the configure batch.

        if _engine == 'numpy':
            l_LL_batch = func_calc_LL(_batch_configures, _GWASsummary, _LDmatrix, Lprior, _LL_0, _ncp=_ncp)

        elif _engine.startswith('jax'):
            
            GWAS_summary_jax = jnp.array(_GWASsummary.sr_GWAS_summary.values)

            # ✅ (1) 벡터 연산으로 LD matrices 생성
            t_start_1 = datetime.now()
            _LDmatrix_jax = generate_LD_matrices_jax(
                jnp.array(_batch_configures), jnp.array(_LDmatrix.df_LD.values), jnp.array(_LDmatrix.l_ix_SNPs), _ncp
            )
            # print("✅ (1) LD matrices 생성 (R + R @ diagC @ R) && Subset: {}".format(datetime.now() - t_start_1))

            # ✅ (2) LL 계산.
            t_start_2 = datetime.now()
            l_LL_batch = calc_LL_given_batch_jax(GWAS_summary_jax, _LDmatrix_jax, Lprior, _LL_0, _ncp=_ncp)
            # print("✅ (2) LL 계산: {}".format(datetime.now() - t_start_2))

            # print("Total Time of this batch: {}".format(datetime.now() - t_start_1))

        else:
            raise ValueError("Wrong engine! ({})".format(_engine))


        
        ### (2-2) accumulation for PIP.
        for j, (_configure, _LL) in enumerate(zip(_batch_configures, l_LL_batch)):

            # print("===[{}]: {} / {}".format(j, _configure, _LL))
            
            for _idx_SNP in _configure:

                # print(_idx_SNP)

                arr_PIP_acc[_idx_SNP] += _LL

        # display(arr_PIP_acc)


        ### (2-3) acc for the prob. of `N_causal`
        acc_LL_N_causal += np.sum(l_LL_batch)

        
        # if i >= 1: break
        
    

    return arr_PIP_acc, acc_LL_N_causal



########## Postprocessing
def calc_PP(_sr_LL_prior):
    
    """
    - 얘는 걍 고대로 베겨옴.
    - `calc_PP()` in mod_PostCal_NcpMean.py
    
    """

    # display(_sr_LL_prior)

    ## 1. max(LLs)
    C = np.max(_sr_LL_prior)
    # print("\nMax LL: {}".format(C))


    ## 2. LL 값에서 C를 뺀 값으로 exp 계산
    shifted_exp = np.exp(_sr_LL_prior - C)
    # print("shifted_exp: ")
    # display(shifted_exp)

    ## 3. 합산 후 로그 값 복원
    log_sum_exp = C + np.log(np.sum(shifted_exp))    
    # print("log_sum_exp:")
    # print(log_sum_exp)


    ## log Posterior prob.
    sr_logPP = (_sr_LL_prior - log_sum_exp).rename("logPP")
    # print("sr_logPP: ")
    # display(sr_logPP)


    ## Posterior prob.
    sr_PP = pd.Series( np.exp(sr_logPP), name='PP', index=_sr_LL_prior.index)
    # print("sr_PP: ")
    # display(sr_PP)

    # print(sr_PP.sum())

    return sr_PP, sr_logPP    



def postprepr_LL(_df_result, _rho=0.95, 
                 _l_type=('whole', 'SNP', 'HLAtype', 'HLA', 'AA', 'intraSNP')) -> dict:
    
    """
    - mod_PostCal_NcpMean.py에 있는 `postprepr_LL()`함수 거의 그대로 가져옴.
    - cov-model 상황에 맞춰서 조금만 수정함. (완전히 같은 함수는 아님.)
    
    """
    
    print("\nPostprocessing the calculated LLs.")    
    # display(_df_result)
    
    
    ##### (1.5) sorting
    df_result_sort = _df_result \
        .sort_values("LL+Lprior", ascending=False) \
        .rename_axis("index", axis=0) \
        .reset_index(drop=False) \
        .rename_axis("rank", axis=0) \
        .reset_index(drop=False)
    
    print(df_result_sort.head(10))
    # display(df_result_sort.sort_values("LL", ascending=False)) # (Conclusion) 똑같이 나옴, 당연히
    
    
    
    """
    ## 다양한 sub-group of markers들에 대해서 PP를 구할 거임.
    
    - 다음 두 가지를 구분 해야 함.
        - (1) 특정 group of markers들에 한정해서 PP를 계산. (PP를 여러번 구함.)
        - (2) 그냥 전체 PP 구한 후, 특정 group of markers들의 PP proportion을 구함. (전체 PP 한번 구한거를 나눠 보는거임.)

    - target group of markers related to the (1).
        - Whole: SNP + HLA type markers (HLA + AA + intraSNPs)
        - HLA type markers
        - HLA
        - AA
        - intraSNP
    
    - 근데 (2)는 좀만 나중에.
    """
        
    f_HLA = df_result_sort['SNP'].str.startswith("HLA")
    f_AA = df_result_sort['SNP'].str.startswith("AA")
    f_intraSNP = df_result_sort['SNP'] \
                    .map(lambda x: re.match(r'^SNP_(\S+)_(\d+)', x)) \
                    .map(lambda x: bool(x))
    
    f_HLAtype = f_HLA | f_AA | f_intraSNP
    f_SNP = ~f_HLAtype
    f_whole = f_SNP | f_HLAtype
    
    d_flag_target_group = {
        "whole": f_whole,
        "SNP": f_SNP,
        "HLAtype": f_HLAtype,
        "HLA": f_HLA,
        "AA": f_AA,
        "intraSNP": f_intraSNP,
    }

    if 0 < len(_l_type) and len(_l_type) < 6:
        ## simulation같은거 할 때는 그냥 'whole'만 return했으면 좋겠음.
        d_flag_target_group = {k: v for k, v in d_flag_target_group.items() if k in _l_type}
    

    
    def postprepr_LL_subgroup(_df_LL_Lprior_sort_sub, _rho):

        # print("\nSummarizing posterior probabilities.")
        # display(_df_LL_Lprior_sort)
        
        ##### (2) diff (abs)
        l_LL_Lprior = _df_LL_Lprior_sort_sub['LL+Lprior'].tolist()
        sr_diff_abs = pd.Series(
            [0.0] + [l_LL_Lprior[i-1] - l_LL_Lprior[i] for i in range(1, _df_LL_Lprior_sort_sub.shape[0])],
            index = _df_LL_Lprior_sort_sub.index,
            name='LL+Lprior_diff'
        )

    
        ##### (3) diff (abs) acc
        acc_temp = 0.0

        l_temp = []
        for i, _diff in enumerate(sr_diff_abs):
            acc_temp += _diff
            l_temp.append(acc_temp)

        sr_diff_abs_acc = pd.Series(
            l_temp, 
            index=_df_LL_Lprior_sort_sub.index, name='LL+Lprior_diff_acc'
        )
        # display(sr_diff_abs_acc)
        
        
        ##### (4) Posterior Prob. (PP)
        sr_PP, sr_logPP = calc_PP(_df_LL_Lprior_sort_sub['LL+Lprior'])
        
        
        ##### (5) credible set
        def get_credible_set(_sr_PP, _rho) -> list:
            
            l_CredibleSet = []
            
            if _sr_PP.iat[0] >= _rho:
                
                return [True] + [False] * (_sr_PP.shape[0] - 1)
                
            
            acc_PP = 0.0

            for _pp in sr_PP:

                acc_PP += _pp
                                
                l_CredibleSet.append(True)
                
                if acc_PP >= _rho: break

            l_CredibleSet = l_CredibleSet + [False] * (len(_sr_PP) - len(l_CredibleSet))
        
            return l_CredibleSet
        
        sr_CredibleSet = pd.Series(
            get_credible_set(sr_PP, _rho), index=sr_PP.index, name='CredibleSet'
        )
        
        
        df_RETURN = pd.concat(
            [
                _df_LL_Lprior_sort_sub, 
                sr_diff_abs, 
                sr_diff_abs_acc, 
                sr_logPP,
                sr_PP,
                sr_CredibleSet
            ],axis=1
        )        


        return df_RETURN # 여기서 한번 끊자.    
    
    
    
    
    for i, (_key, _sr_flag) in enumerate(d_flag_target_group.items()):

        # print("\n===[{}]: {}".format(i, _key))
        
        d_flag_target_group[_key] = \
            postprepr_LL_subgroup(df_result_sort[_sr_flag], _rho)

        # display(d_flag_target_group[_key])


    
    return d_flag_target_group    




