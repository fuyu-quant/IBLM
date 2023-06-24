import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}
        sex_prob = {True: 0.74, False: 0.19}
        age_prob = {0: 0.67, 1: 0.42, 2: 0.40, 3: 0.44, 4: 0.09}
        fare_prob = {0: 0.68, 1: 0.51, 2: 0.42, 3: 0.32}

        # Assign age group based on age
        age = row['age']
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

        # Assign fare group based on fare
        fare = row['fare']
        if fare <= 7.91:
            fare_group = 0
        elif fare <= 14.45:
            fare_group = 1
        elif fare <= 31:
            fare_group = 2
        else:
            fare_group = 3

        # Calculate the probability of survival based on the given features
        y = pclass_prob[row['pclass']] * sex_prob[row['sex_female']] * age_prob[age_group] * fare_prob[fare_group]

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)