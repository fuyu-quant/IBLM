import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that if the passenger is female, in first class, and embarked from Cherbourg, they have a high probability of survival.
        # This is based on historical data from the Titanic disaster, where women, children, and first-class passengers were given priority for lifeboats.
        # We are also assuming that if the passenger is male, in third class, and embarked from Southampton, they have a low probability of survival.
        # This is also based on historical data, where men and third-class passengers had lower survival rates.
        # This is a very simplistic approach and would not be accurate for all cases, but it serves as a starting point for this exercise.

        if row['sex_female'] == 1.0 and row['class_First'] == 1.0 and row['embark_town_Cherbourg'] == 1.0:
            y = 1.0
        elif row['sex_male'] == 1.0 and row['class_Third'] == 1.0 and row['embark_town_Southampton'] == 1.0:
            y = 0.0
        else:
            y = 0.5  # If none of the conditions are met, we assign a neutral probability.

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)