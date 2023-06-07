import numpy as np
import pandas as pd

def model(x):
    df = x.copy()

    # Preprocessing
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['class'] = df['class'].map({'First': 1, 'Second': 2, 'Third': 3})
    df['who'] = df['who'].map({'man': 0, 'woman': 1, 'child': 2})
    df['adult_male'] = df['adult_male'].astype(int)
    df['alone'] = df['alone'].astype(int)

    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        pclass = row['pclass']
        sex = row['sex']
        age = row['age']
        sibsp = row['sibsp']
        parch = row['parch']
        fare = row['fare']
        embarked = row['embarked']
        who = row['who']
        adult_male = row['adult_male']
        alone = row['alone']

        # Prediction logic
        y = 0
        if sex == 1:
            y += 1
        if pclass == 1:
            y += 0.5
        elif pclass == 2:
            y += 0.25
        if age <= 16:
            y += 0.5
        if fare > df['fare'].median():
            y += 0.25
        if embarked == 1:
            y += 0.25
        if who == 1 or who == 2:
            y += 0.5
        if adult_male == 0:
            y += 0.25
        if alone == 1:
            y += 0.25

        y = 1 / (1 + np.exp(-y))
        output.append(y)
        
    return output