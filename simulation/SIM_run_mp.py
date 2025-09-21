import os, sys, re
import numpy as np
import pandas as pd

import subprocess
from multiprocessing import Pool

# %load_ext autoreload
# %autoreload 2

"""
- (1) SSFN파일에 disjoint하게 GPU_ID할당.
- (2) mp로 run

아직 class로 wrapping해야할만한 수준을 아니라서 그냥 함수들로 준비함.
"""



def alloc_GPU(_df_SSFN):

    l_GPU_ToUse = [0, 3, 5, 7]
    
    d_GPU_ToMap = {
        0: 2, # 2로 잡아야 0이 나옴.
        3: 1, # 1로 잡아야 3이 나옴.
    }    

    print(_df_SSFN)

    arr_split = np.array_split(_df_SSFN.index.tolist(), len(l_GPU_ToUse))

    sr_GPU_ID = pd.Series(
        [_gpu_id for (_arr, _gpu_id) in zip(arr_split, l_GPU_ToUse) for _ in range(len(_arr))],
        name='gpu_id',
        index=_df_SSFN.index
    )
    sr_GPU_ID = sr_GPU_ID.map(lambda x: d_GPU_ToMap[x] if x in d_GPU_ToMap else x)
    # display(sr_GPU_ID)
    print(sr_GPU_ID.value_counts())


    df_RETURN = pd.concat([_df_SSFN, sr_GPU_ID], axis=1)

    return df_RETURN



### a single run of batch
## 이 함수는 상황에 따라 만들 여지가 높음. (복붙해서 가져다 쓰는 용도.)
def run_SUM2HLA(_args):

    _index, _Sim_No, _ncp_1st, _fpath_IN, _fpath_OUT, _gpu_id = _args

    print(f"=====[ index: {_index} / NCP_1st: {_ncp_1st} / Sim_No: {_Sim_No} ]")

    cmd = [
        "conda", "run", "-n", "jax_gpu", # 'jax_gpu' 환경에서 실행하도록 지정
        "python",
        "SUM2HLA.py",
        "--sumstats", _fpath_IN,
        "--ref", "/data02/wschoi/_ClusterPhes_v4/LD_from_HLA_reference_panel/REF_T1DGC.hg19.SNP+HLA",
        "--out", _fpath_OUT,
        "--gpu-id", str(_gpu_id)
    ]

    # print(cmd)

    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    
    if result.returncode != 0:
        # 에러가 발생한 경우, 에러 메시지를 출력
        return False
    return True



def print_PID():

    print(os.getpid())

    return 0


def do_multiprocessing(_df_SSFN, _chunksize, _func, _l_GPU_ToUse = [0, 3, 5, 7]):

    """
    - ipynb에서 쓸 경우, 얘는 반드시 "if __name == `__main__`:" 의 guard line내에서 써야 함.
    
    """
    print(f"✅ 이 커널의 PID는 {os.getpid()} 입니다. 이 번호를 기록해두세요.")    
    
    tasks = _df_SSFN.itertuples(name=None)
    
    pool = Pool(processes=len(_l_GPU_ToUse))

    try:
        # pool.map을 실행. 이 함수가 완료될 때까지 여기서 대기합니다.
        results = pool.map(_func, tasks, chunksize=_chunksize)
        print("\n--- 모든 작업 정상 완료 ---")

        sr_SUCCESS = pd.Series(list(results), name="success")
        print(sr_SUCCESS)
        print(sr_SUCCESS.all())

    except KeyboardInterrupt:
        # Ctrl+C 또는 Jupyter의 중단 버튼을 누르면 이 블록이 실행됩니다.
        print("\n🚨 사용자에 의해 작업이 중단되었습니다. 모든 자식 프로세스를 종료합니다...")
        
        # pool.terminate() : 현재 진행 중인 작업들을 즉시 강제 종료합니다. (가장 중요!)
        pool.terminate()
        
        # pool.join() : 자식 프로세스들이 완전히 종료될 때까지 기다립니다.
        pool.join()
        
        print("✅ 모든 프로세스가 성공적으로 종료되었습니다.")

    finally:
        # 정상 종료 되거나, 예외(중단)가 발생해도 항상 실행됩니다.
        # 열려있는 풀을 닫아줍니다.
        pool.close()
        pool.join()


    return sr_SUCCESS