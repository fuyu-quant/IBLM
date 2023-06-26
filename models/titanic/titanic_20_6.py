import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Feature weights
        weights = {
            'pclass': -0.15,
            'age': -0.01,
            'sibsp': -0.05,
            'parch': 0.02,
            'fare': 0.002,
            'sex_female': 0.5,
            'sex_male': -0.5,
            'embarked_C': 0.1,
            'embarked_Q': 0.05,
            'embarked_S': -0.05,
            'alive_no': -0.5,
            'alive_yes': 0.5,
            'alone_False': -0.1,
            'alone_True': 0.1,
            'adult_male_False': 0.3,
            'adult_male_True': -0.3,
            'who_child': 0.2,
            'who_man': -0.2,
            'who_woman': 0.2,
            'class_First': 0.2,
            'class_Second': 0.1,
            'class_Third': -0.1,
            'deck_A': 0.05,
            'deck_B': 0.1,
            'deck_C': 0.05,
            'deck_D': 0.1,
            'deck_E': 0.1,
            'deck_F': 0.05,
            'deck_G': 0.05,
            'embark_town_Cherbourg': 0.1,
            'embark_town_Queenstown': 0.05,
            'embark_town_Southampton': -0.05
        }

        # Calculate the weighted sum
        weighted_sum = sum(row[key] * value for key, value in weights.items())

        # Apply the logistic function to get the probability
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)