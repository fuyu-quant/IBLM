import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}
        sex_prob = {'male': 0.19, 'female': 0.74}
        age_prob = {0: 0.67, 1: 0.42, 2: 0.40, 3: 0.44, 4: 0.09}
        fare_prob = {0: 0.20, 1: 0.43, 2: 0.45, 3: 0.58}
        
        pclass = row['pclass']
        sex = row['sex']
        age = row['age']
        fare = row['fare']

        # Age probability
        if age <= 16:
            age_group = 0
        elif age <= 32:
            age_group = 1
        elif age <= 48:
            age_group = 2
        elif age <= 64:
            age_group = 3
        else:
            age_group = 4

        # Fare probability
        if fare <= 7.91:
            fare_group = 0
        elif fare <= 14.454:
            fare_group = 1
        elif fare <= 31:
            fare_group = 2
        else:
            fare_group = 3

        # Calculate the final probability
        y = pclass_prob[pclass] * sex_prob[sex] * age_prob[age_group] * fare_prob[fare_group]

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)