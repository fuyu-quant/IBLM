import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}
        age_prob = {0: 0.55, 1: 0.34, 2: 0.42, 3: 0.44, 4: 0.37}
        fare_prob = {0: 0.68, 1: 0.51, 2: 0.42, 3: 0.31}
        sex_prob = {0: 0.74, 1: 0.19}
        embarked_prob = {0: 0.55, 1: 0.39, 2: 0.34}

        # Assign age group based on age
        age_group = 0
        if row['age'] <= 16:
            age_group = 0
        elif row['age'] <= 32:
            age_group = 1
        elif row['age'] <= 48:
            age_group = 2
        elif row['age'] <= 64:
            age_group = 3
        else:
            age_group = 4

        # Assign fare group based on fare
        fare_group = 0
        if row['fare'] <= 7.91:
            fare_group = 0
        elif row['fare'] <= 14.454:
            fare_group = 1
        elif row['fare'] <= 31:
            fare_group = 2
        else:
            fare_group = 3

        # Calculate the probability based on the given features
        pclass = pclass_prob[row['pclass']]
        age = age_prob[age_group]
        fare = fare_prob[fare_group]
        sex = sex_prob[row['sex_female']]
        embarked = embarked_prob[row['embarked_C'] + 2 * row['embarked_Q']]

        # Calculate the final probability
        y = pclass * age * fare * sex * embarked

        # Normalize the probability to be between 0 and 1
        y = y / (y + (1 - y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)