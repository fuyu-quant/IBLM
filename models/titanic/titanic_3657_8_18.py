import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based system to predict the target.
        # The rules are based on the observation of the data.
        # For example, if the passenger is female, embarked from Cherbourg, and is in first class, 
        # the probability of survival is high.
        # On the other hand, if the passenger is male, embarked from Southampton, and is in third class, 
        # the probability of survival is low.
        # These rules are not perfect and may not work well on unseen data.

        if row['sex_female'] == 1.0 and row['embarked_C'] == 1.0 and row['class_First'] == 1.0:
            y = 0.9
        elif row['sex_male'] == 1.0 and row['embarked_S'] == 1.0 and row['class_Third'] == 1.0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)