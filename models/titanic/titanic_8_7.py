import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}
        age_prob = {0: 0.55, 1: 0.34, 2: 0.42, 3: 0.44, 4: 0.37, 5: 0.49, 6: 0.45, 7: 0.5}
        fare_prob = {0: 0.68, 1: 0.51, 2: 0.42, 3: 0.31}
        sex_prob = {0: 0.19, 1: 0.74}
        embarked_prob = {0: 0.55, 1: 0.39, 2: 0.34}

        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']

        # Age group
        age_group = int(age // 10)
        if age_group > 7:
            age_group = 7

        # Fare group
        fare_group = int(fare // 10)
        if fare_group > 3:
            fare_group = 3

        # Sex group
        sex_group = int(sex_female)

        # Embarked group
        if embarked_C:
            embarked_group = 0
        elif embarked_Q:
            embarked_group = 1
        else:
            embarked_group = 2

        # Calculate the probability
        y = pclass_prob[pclass] * age_prob[age_group] * fare_prob[fare_group] * sex_prob[sex_group] * embarked_prob[embarked_group]

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)