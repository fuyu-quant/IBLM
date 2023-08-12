import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority for lifeboats.
        # The conditions can be adjusted based on the specific dataset and the correlations between the features and the target variable.

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

        # ensure probability stays within [0, 1]
        prob = max(0, min(prob, 1))

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)