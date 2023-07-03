import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # This is a very simple model and does not take into account interactions between variables or non-linear relationships
        y = 0.0
        if row['sex_female'] == 1:
            y += 0.3
        if row['pclass'] == 1:
            y += 0.3
        if row['embarked_C'] == 1:
            y += 0.3
        if row['fare'] > 30:
            y += 0.1
        if y > 1.0:
            y = 1.0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)