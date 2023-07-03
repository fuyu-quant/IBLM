import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data that women, children, and first-class passengers had higher survival rates.
        # The age and fare are also considered, younger and passengers with higher fare are assumed to have higher survival rates.
        # The values are normalized to be between 0 and 1 to represent probabilities.

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embark_town_Cherbourg']
        y -= row['age'] / 100  # assuming age is less than 100
        y += row['fare'] / 500  # assuming fare is less than 500

        # normalize the result to be between 0 and 1
        y = (y + 2) / 4

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)