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
        fare_prob = {0: 0.68, 1: 0.43, 2: 0.42, 3: 0.39}

        age_group = int(row['age'] // 20)
        fare_group = int(row['fare'] // 10)

        pclass = row['pclass']
        sex_female = row['sex_female']

        prob = pclass_prob[pclass] * sex_prob[sex_female] * age_prob.get(age_group, 0.4) * fare_prob.get(fare_group, 0.4)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)