import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

sample = 300
sample_num = int(sample/2)
seed = 3655

def titanic_data(df, sample, seed):
    sample_num = int(sample/2)

    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

    df = pd.get_dummies(df, columns=['sex'])
    df = pd.get_dummies(df, columns=['embarked'])
    df = pd.get_dummies(df, columns=['alive'])
    df = pd.get_dummies(df, columns=['alone'])
    df = pd.get_dummies(df, columns=['adult_male'])
    df = pd.get_dummies(df, columns=['who'])
    df = pd.get_dummies(df, columns=['class'])
    df = pd.get_dummies(df, columns=['deck'])
    df = pd.get_dummies(df, columns=['embark_town'])
    df = df.replace({True: 1, False: 0})

    cols = list(df.columns)
    cols.remove('survived')
    cols.append('survived')
    df = df[cols]

    df_1 = df[df['survived'] == 1].sample(n = sample_num, random_state = seed)
    df_0 = df[df['survived'] == 0].sample(n = sample_num, random_state = seed)

    df_train = pd.DataFrame()
    df_len = len(df_1)
    for i in range(df_len):
        df1 = pd.DataFrame([df_1.iloc[i]])
        df0 = pd.DataFrame([df_0.iloc[i]])
        df_train = pd.concat([df_train, df1, df0])

    df_train['survived'] = df_train['survived'].astype(int)
    df_test = df.drop(df_train.index)

    print(f'train data:{len(df_train)}')
    print(f'test data:{len(df_test)}')
    return df_train, df_test


