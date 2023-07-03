import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # as these factors are generally associated with higher survival rates in the Titanic disaster.
        # The age and fare are also considered, younger and higher fare are assumed to have higher survival rates.
        # This is a simple heuristic and does not guarantee high accuracy.

        prob = 0.5  # start with a base probability

        # increase probability for females
        if row['sex_female'] == 1.0:
            prob += 0.3

        # increase probability for first class passengers
        if row['class_First'] == 1.0:
            prob += 0.1

        # increase probability for passengers embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.05

        # decrease probability for older passengers
        if row['age'] > 30.0:
            prob -= 0.05

        # increase probability for passengers with higher fare
        if row['fare'] > 30.0:
            prob += 0.05

        # ensure probability is within [0,1]
        prob = min(max(prob, 0), 1)

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)