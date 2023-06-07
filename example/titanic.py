import numpy as np
import pandas as pd

def titanic(data):
    df = pd.DataFrame(data, columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck", "Title", "Alone", "Cabin", "Embark_town", "Alive", "IsAlone", "Target"])

    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        sex = 1 if row["Sex"] == "male" else 0
        age = row["Age"]
        if pd.isna(age):
            age = 30
        fare = row["Fare"]
        if pd.isna(fare):
            fare = 14
        pclass = row["Pclass"]
        sibsp = row["SibSp"]
        parch = row["Parch"]
        embarked = 0
        if row["Embarked"] == "C":
            embarked = 1
        elif row["Embarked"] == "Q":
            embarked = 2

        # Prediction logic
        y = -1.5 + 0.1 * sex - 0.02 * age + 0.15 * fare - 0.5 * pclass + 0.3 * sibsp - 0.1 * parch + 0.2 * embarked

        y = 1 / (1 + np.exp(-y))
        output.append(y)
        
    return output