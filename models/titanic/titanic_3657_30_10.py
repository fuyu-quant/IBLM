import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The age of the passenger is also considered, giving higher survival probability for younger passengers.
        # The fare paid by the passenger is also considered, assuming that passengers who paid higher fares might have been given priority during the evacuation.
        # The conditions are weighted according to their perceived impact on the survival rate.

        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.2
        y += row['embark_town_Cherbourg'] * 0.1
        y += (1 - row['age']/80) * 0.2  # assuming the oldest passenger is 80
        y += row['fare']/500 * 0.2  # assuming the highest fare is 500

        # The final probability is capped at 1.0
        y = min(y, 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)