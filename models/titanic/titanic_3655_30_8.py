import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on the data, and in a real scenario, a machine learning model would be trained to find these patterns
        prob = 0.5
        if row['sex_female'] == 1.0:
            prob += 0.3
        if row['pclass'] == 1.0:
            prob += 0.1
        if row['embarked_C'] == 1.0:
            prob += 0.1

        # Ensure probability is between 0 and 1
        prob = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)