import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, 
        # belong to first class, embarked from Cherbourg and are adults.
        # This is based on the historical data that such passengers had higher survival rates.
        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.3
        if row['embarked_C'] == 1.0:
            y += 0.2
        if row['who_adult'] == 1.0:
            y += 0.2

        # Normalize the probability to be between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)