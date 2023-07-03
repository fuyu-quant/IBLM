import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and embarked from Cherbourg.
        # These conditions are based on the historical fact that women, children and first class passengers were given priority during the evacuation of the Titanic.
        # We are also assuming that the target is less likely to be 1 if the passenger is male, is in third class, and embarked from Southampton.
        # These assumptions are simplistic and may not hold true for all cases, but they serve as a starting point for our prediction.

        if row['sex_female'] == 1 and row['class_First'] == 1 and row['embark_town_Cherbourg'] == 1:
            y = 0.9
        elif row['sex_male'] == 1 and row['class_Third'] == 1 and row['embark_town_Southampton'] == 1:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)