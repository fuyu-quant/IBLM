import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and embarked from Cherbourg.
        # This is based on the historical data from the Titanic disaster, where women, children, and first-class passengers were more likely to survive.
        # Of course, this is a very simplistic approach and a real-world solution would likely use a machine learning model.

        p = 0.0
        if row['sex_female'] == 1:
            p += 0.3
        if row['class_First'] == 1:
            p += 0.3
        if row['embark_town_Cherbourg'] == 1:
            p += 0.3
        if row['fare'] > 30:
            p += 0.1

        # The probability should be between 0 and 1
        p = min(max(p, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(p)
    return np.array(output)