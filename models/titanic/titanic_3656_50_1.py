import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority during the evacuation.
        # The condition for embarkation from Cherbourg is based on the data provided, which shows a higher survival rate for these passengers.
        # The age and fare are also considered, with younger and higher-paying passengers given a higher probability of survival.
        # The resulting probability is a simple weighted sum of these factors.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['class_First'] == 1.0:
            y += 0.3
        if row['embark_town_Cherbourg'] == 1.0:
            y += 0.2
        y += 0.1 * (1.0 - row['age'] / 80.0)  # assuming age is between 0 and 80
        y += 0.1 * (row['fare'] / 500.0)  # assuming fare is between 0 and 500

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)