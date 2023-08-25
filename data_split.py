import numpy as np
import pandas as pd
import math

def k_fold_split(df : pd.DataFrame, k : int, random_state : np.random.Generator = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (k < 2) : raise ValueError("k must be >= 2. The value of k was: {}".format(k))
    df         = df.copy()
    df         = df.sample(frac=1) if random_state is None \
                    else df.sample(frac=1, random_state=random_state)
    df_size    = df.shape[0]
    fold_times = math.ceil(df_size / k)
    train_df   = pd.DataFrame(columns=df.columns)
    test_df    = pd.DataFrame(columns=df.columns)
    for i in range(fold_times):
        curr_fold = df.iloc[i*k:(i+1)*k]
        train_df  = pd.concat([train_df, curr_fold.iloc[0:k-1]])
        test_df   = pd.concat([test_df,  curr_fold.iloc[k-1].to_frame().T]) if curr_fold.shape[0] >= k \
                        else test_df
    return train_df, test_df