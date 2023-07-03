import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # We will use a simple rule-based system to predict the target.
        # If the passenger is female, in first class, and embarked from Cherbourg, we predict a high probability of survival.
        # Otherwise, we predict a low probability of survival.
        y = 0.1  # default low probability
        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 0.9  # high probability

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)