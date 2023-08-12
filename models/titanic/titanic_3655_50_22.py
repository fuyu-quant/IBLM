import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The age and fare are also considered, younger and passengers with higher fare are assumed to have higher survival rates.
        # The conditions are arbitrarily weighted based on their perceived impact on the survival rate.

        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.2
        y += row['embark_town_Cherbourg'] * 0.1
        y += (row['age'] <= 30) * 0.1
        y += (row['fare'] >= 30) * 0.1

        # The probability is capped at 1
        y = min(y, 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)