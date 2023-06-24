import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.63, 2: 0.47, 3: 0.24}
        age_prob = {0: 0.55, 1: 0.34, 2: 0.42, 3: 0.44, 4: 0.38, 5: 0.29, 6: 0.00, 7: 0.50}
        fare_prob = {0: 0.20, 1: 0.42, 2: 0.45, 3: 0.58}
        sex_female_prob = 0.74
        sex_male_prob = 0.19
        embarked_C_prob = 0.55
        embarked_Q_prob = 0.39
        embarked_S_prob = 0.34

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

        prob = pclass_prob[pclass] * age_prob[age_group] * fare_prob[fare_group]
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

        # Normalize the probability to be between 0 and 1
        y = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)