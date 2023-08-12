import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers were given priority for lifeboats.
        # The embarkation point is also considered as passengers from Cherbourg had a higher survival rate.
        # The age and fare are also considered where younger passengers and passengers who paid higher fares had higher survival rates.
        # The conditions are weighted according to their perceived impact on the survival rate.

        y = 0.0
        y += row['sex_female'] * 0.35
        y += row['class_First'] * 0.25
        y += row['embark_town_Cherbourg'] * 0.15
        y += (1 - row['age']/80) * 0.15  # assuming the oldest passenger is 80
        y += row['fare']/500 * 0.10  # assuming the highest fare is 500

        # The probability is capped at 1
        y = min(y, 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)