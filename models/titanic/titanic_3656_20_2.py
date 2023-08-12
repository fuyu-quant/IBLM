import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to assign higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # This is based on the historical data that women, children, and the upper-class passengers were the first to be evacuated.
        # The age, fare, and number of siblings/spouses/parents/children are also considered.
        # The probability is calculated as a weighted sum of these factors.

        y = 0.0
        y += row['sex_female'] * 0.35
        y += row['class_First'] * 0.25
        y += row['embark_town_Cherbourg'] * 0.15
        y += row['age'] / 100 * 0.1
        y += row['fare'] / 100 * 0.1
        y -= row['sibsp'] / 10 * 0.05
        y -= row['parch'] / 10 * 0.05

        # The probability is capped between 0 and 1.
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)