import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # This is a very simplistic model and would not perform well in a real-world scenario
        y = 0.5
        if row['sex_female'] == 1:
            y += 0.3
        if row['class_First'] == 1:
            y += 0.1
        if row['embark_town_Cherbourg'] == 1:
            y += 0.1

        # Ensure the probability is within [0,1]
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)