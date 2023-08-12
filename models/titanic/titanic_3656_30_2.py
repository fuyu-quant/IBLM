import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # The age and fare are also considered, younger and passengers who paid higher fare are given higher probability.
        # This is a simple logic and does not guarantee high accuracy.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['embarked_C'] == 1.0:
            y += 0.1
        if row['age'] <= 30.0:
            y += 0.2
        if row['fare'] >= 30.0:
            y += 0.2

        # Limit the probability to 1
        if y > 1.0:
            y = 1.0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)