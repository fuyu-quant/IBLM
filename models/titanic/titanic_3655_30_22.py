import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and embarked from Cherbourg.
        # This is a simple heuristic and may not be accurate for all cases.
        # A more accurate model would require training a machine learning model on the data.
        y = 0.0
        if row['sex_female'] == 1:
            y += 0.3
        if row['class_First'] == 1:
            y += 0.3
        if row['embark_town_Cherbourg'] == 1:
            y += 0.3
        if row['fare'] > 30:
            y += 0.1

        # Ensure the probability is between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)