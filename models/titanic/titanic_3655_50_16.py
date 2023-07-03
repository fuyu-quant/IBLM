import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority during the evacuation.
        # The conditions are simplified for the purpose of this task and do not take into account all possible factors that could influence survival.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.3
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.3
        if row['fare'] > 30.0:
            y += 0.1

        # The probability is capped at 1.0
        y = min(y, 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)