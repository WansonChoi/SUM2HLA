import scipy as sp
import numpy as np
import pandas as pd



def calc_Z_CMVN(_df_ma, _df_LD, _l_ToCondition):

    if 'z' in _df_ma or 'Z' in _df_ma:
        df_ma_2 = _df_ma
    elif 'b' in _df_ma and 'se' in _df_ma:
        df_ma_2 = pd.concat(
            [_df_ma, (_df_ma['b'] / _df_ma['se']).rename("Z")],
            axis=1
        )
    else:
        print("Neither 'z' (or 'Z') nor ('b', 'se') are in 'df_ma'.")
        return -1

    ##### (1) ma 파일을 `df_ToIter`와 `df_ToCond`으로 나누기.
    f_condition = df_ma_2['SNP'].isin(_l_ToCondition)
    df_ToIter = df_ma_2[~f_condition]
    df_ToCond = df_ma_2[f_condition]

    # display(_df_ma)
    # display(df_ToCond)



    ##### (2) The `df_ToCond`의 LD matrix 준비하기.

    ## 원래는 `df_ToCond`을 `df_LD`의 order로 맞춰야 하지만,
    ## _l_ToCondition순으로 `df_ToCond`을 `df_LD` 둘 다 순서를 맞추는 걸로.

    df_ToCond = df_ToCond.set_index('SNP', drop=False).loc[_l_ToCondition, :].reset_index("SNP", drop=True)
    df_LD_ToCond = _df_LD.loc[df_ToCond['SNP'], df_ToCond['SNP']]

    # display(df_ToIter)
    # display(df_ToCond)
    # display(df_LD_ToCond)



    ##### (3) Main iteration

    l_conditional_Z = []
    df_ToIter_2 = df_ToIter[['SNP', 'Z']]

    for i, (_index, _a_SNP, _Z) in enumerate(df_ToIter_2.itertuples()):

        # print(f"===[{i}]: {_a_SNP}")

        """
        - group2: a marker (condition 받을 대상)
        - group1: conditioning markers

        """

        ##### the `_a_SNP` x `df_ToCond['SNP']` 들 간의 correlation (LD_group2; 1 x len(df_ToCond['SNP']))
        df_LD_SNP_Conds = _df_LD.loc[[_a_SNP] , df_ToCond['SNP']]
        # df_LD_SNP_Conds = _df_LD.loc[_a_SNP , df_ToCond['SNP']] # 이렇게 해도 값은 똑같이 나옴.
        # display(df_LD_SNP_Conds)

        ##### (3-1) Conditional mean (Z) 계산

        ## Conditional_Z = _Z - (mu_g2 + df_LD_SNP_Conds @ np.linalg.inv(df_LD_ToCond) @ (df_ToCond['Z'] - mu_g1))
        ## 여기서, mu_g2 = mu_g1 = 0

        Z_tagging_effect = df_LD_SNP_Conds @ np.linalg.inv(df_LD_ToCond) @ df_ToCond['Z']
        Conditional_Z = _Z - Z_tagging_effect.iat[0]

        # print(f"Conditional_Z: {Conditional_Z}")

        ### Conditional variance는 계산 생략. (Z의 variance는 필요 없음.)

        l_conditional_Z.append(Conditional_Z)


        # if i >= 1: break


    sr_cZ = pd.Series(l_conditional_Z, name='cZ', index=df_ToIter_2.index)
    sr_cBETA = (sr_cZ * df_ToIter['se']).rename("cBETA")
    sr_cP = sr_cZ.abs().map(lambda x: 2 * sp.stats.norm.cdf(-x)).rename("cP")


    df_RETURN = pd.concat([df_ToIter, sr_cZ, sr_cBETA, sr_cP], axis=1)

    return df_RETURN



def iterate_calc_Z_CMVN():

    """
    가장 naive하게 pC만 이용해서 iterate하는 함수를 여기서 짤거임.


    """

    return 0





if __name__ == '__main__':

    pass