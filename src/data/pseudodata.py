from __future__ import annotations

import string

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification


columns = 4
num_train = 300
seed = 3657

def get_pseudodata(num_train, columns, seed):
    sample = 1000
    X, y = make_classification(n_samples=sample, n_features=columns, random_state=seed)
    X = np.round(X, 2)
    y = np.round(y, 2)

    column_name = list(string.ascii_lowercase[:columns])

    df = pd.DataFrame(X, columns = column_name)
    df['target'] = y

    sample_num = int(num_train/2)
    df_1 = df[df['target'] == 1].sample(n=sample_num, random_state=seed)
    df_0 = df[df['target'] == 0].sample(n=sample_num, random_state=seed)

    df_train = pd.DataFrame()
    df_len = len(df_1)
    for i in range(df_len):
        df1 = pd.DataFrame([df_1.iloc[i]])
        df0 = pd.DataFrame([df_0.iloc[i]])
        df_train = pd.concat([df_train, df1, df0])

    df_train['target'] = df_train['target'].astype(int)
    df_test = df.drop(df_train.index)

    print(f'train data:{len(df_train)}')
    print(f'test data:{len(df_test)}')
    return df_train, df_test
