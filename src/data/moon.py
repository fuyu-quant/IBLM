from __future__ import annotations

import pandas as pd

from sklearn.datasets import make_moons


num_train = 300
seed = 3657

def moon_data(num_train, seed):
    sample = 1000
    X, y = make_moons(n_samples = sample, noise=0.05, random_state = seed)

    df = pd.DataFrame(data=X, columns=['Feature_1', 'Feature_2']).round(3)
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
