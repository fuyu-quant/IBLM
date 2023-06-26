import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.6, 2: 0.5, 3: 0.3}
        age_prob = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2}
        fare_prob = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2}
        sex_female_prob = 0.7
        sex_male_prob = 0.3
        embarked_C_prob = 0.5
        embarked_Q_prob = 0.4
        embarked_S_prob = 0.6

        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        sex_male = row['sex_male']
        embarked_C = row['embarked_C']
        embarked_Q = row['embarked_Q']
        embarked_S = row['embarked_S']

        age_group = int(age // 10)
        fare_group = int(fare // 10)

        prob = pclass_prob[pclass] * age_prob.get(age_group, 0.1) * fare_prob.get(fare_group, 0.1)

        if sex_female:
            prob *= sex_female_prob
        else:
            prob *= sex_male_prob

        if embarked_C:
            prob *= embarked_C_prob
        elif embarked_Q:
            prob *= embarked_Q_prob
        else:
            prob *= embarked_S_prob

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)