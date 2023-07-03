import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and embarked from Cherbourg.
        # This is a simple heuristic and may not be accurate for all cases.
        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.3
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.3

        # We also consider the age of the passenger, assuming that younger passengers are more likely to be the target.
        # We normalize the age to be between 0 and 1.
        y += (1.0 - row['age'] / 100.0) * 0.1

        # Finally, we consider the fare paid by the passenger, assuming that passengers who paid a higher fare are more likely to be the target.
        # We normalize the fare to be between 0 and 1.
        y += row['fare'] / 500.0 * 0.1

        # We limit the output to be between 0 and 1.
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)