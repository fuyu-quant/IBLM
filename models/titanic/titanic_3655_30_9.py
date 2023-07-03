import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on the historical data of Titanic survivors
        # In reality, a machine learning model should be used to find the best features for prediction

        prob = 0.5  # start with a base probability of 0.5

        # increase probability if passenger is female
        if row['sex_female'] == 1.0:
            prob += 0.3

        # increase probability if passenger is in first class
        if row['class_First'] == 1.0:
            prob += 0.1

        # increase probability if passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.1

        # decrease probability if passenger is alone
        if row['alone_True'] == 1.0:
            prob -= 0.1

        # make sure probability is within [0, 1]
        prob = max(0, min(1, prob))

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)