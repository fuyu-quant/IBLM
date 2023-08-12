import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, passengers in first class (pclass=1) or female passengers (sex_female=1) had higher survival rates.
        # Similarly, passengers who were alone (alone_True=1) or who embarked at Cherbourg (embark_town_Cherbourg=1) also had higher survival rates.
        # The weights for these features are set to 0.2, 0.3, 0.1, and 0.1 respectively.
        # The weights for all other features are set to 0.
        # The predicted probability of survival is then the sum of the weighted features.
        weights = {
            'pclass': 0.2,
            'sex_female': 0.3,
            'alone_True': 0.1,
            'embark_town_Cherbourg': 0.1,
        }
        y = sum(row[feature] * weight for feature, weight in weights.items())

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)