import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on the historical data of Titanic survivors
        # In reality, a machine learning model should be trained to find the best features for prediction

        y = 0.0
        if row['sex_female'] == 1:
            y += 0.3
        if row['pclass'] == 1:
            y += 0.3
        if row['embarked_C'] == 1:
            y += 0.3
        if row['fare'] > 30:
            y += 0.1

        # Normalize the probability to be between 0 and 1
        y = min(max(y, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)