import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # We also consider age, with younger passengers given a slightly higher probability
        # This is a very basic prediction and does not take into account many other factors

        y = 0.5  # start with a base probability of 0.5

        # increase probability for females
        if row['sex_female'] == 1:
            y += 0.3

        # increase probability for first class passengers
        if row['class_First'] == 1:
            y += 0.1

        # increase probability for passengers embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1:
            y += 0.05

        # slightly increase probability for younger passengers
        if row['age'] < 30:
            y += 0.05

        # ensure probability is within [0,1]
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)