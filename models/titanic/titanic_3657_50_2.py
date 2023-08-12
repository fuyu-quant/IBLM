import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that if the passenger is a female, in first class, and embarked from Cherbourg, 
        # they have a higher chance of survival. This is based on the historical data from the Titanic disaster.
        # We also consider the age of the passenger, assuming that younger passengers have a higher chance of survival.
        # The fare is also considered, assuming that passengers who paid more have a higher chance of survival.
        # This is a very basic prediction and does not take into account many other factors.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.3
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 18.0:
            y += 0.1
        if row['fare'] >= 50.0:
            y += 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)