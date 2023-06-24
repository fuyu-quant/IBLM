import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}[row['pclass']]
        sex_prob = {True: 0.74, False: 0.19}[row['sex_female']]
        age_prob = 0.5 if row['age'] <= 16 else 0.36
        fare_prob = 0.5 if row['fare'] <= 20 else 0.4
        embarked_prob = {True: 0.55, False: 0.39}[row['embarked_C'] or row['embarked_Q']]
        alone_prob = {True: 0.3, False: 0.51}[row['alone_True']]

        # Combine the probabilities
        y = pclass_prob * sex_prob * age_prob * fare_prob * embarked_prob * alone_prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)