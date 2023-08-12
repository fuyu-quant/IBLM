import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # The age, fare, and number of siblings/spouses/parents/children are also considered
        # The coefficients for each feature are determined based on their relative importance

        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.2
        y += row['embark_town_Cherbourg'] * 0.1
        y += row['age'] / 80 * 0.1
        y += row['fare'] / 500 * 0.1
        y -= row['sibsp'] / 8 * 0.1
        y -= row['parch'] / 6 * 0.1

        # The probability is capped at 1
        y = min(y, 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)