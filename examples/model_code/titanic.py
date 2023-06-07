import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    df.columns = range(df.shape[1])

    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        pclass = row[7]
        sex = row[1]
        age = row[2]
        fare = row[5]
        embarked = row[6]
        alone = row[13]

        # Prediction logic
        y = 0
        if pclass == 'First':
            y += 0.3
        elif pclass == 'Second':
            y += 0.15

        if sex == 'female':
            y += 0.35

        if age <= 16:
            y += 0.1
        elif age > 16 and age <= 32:
            y += 0.05

        if fare > 50:
            y += 0.1

        if embarked == 'C':
            y += 0.05

        if alone:
            y += 0.05

        y = 1 / (1 + np.exp(-y))
        output.append(y)

    output = np.array(output)

    return output