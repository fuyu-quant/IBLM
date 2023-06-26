import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_prob = {1: 0.6, 2: 0.5, 3: 0.3}[row['pclass']]
        age_prob = 0.5 if row['age'] <= 30 else 0.3
        sibsp_prob = 0.5 if row['sibsp'] == 0 else 0.4
        parch_prob = 0.5 if row['parch'] == 0 else 0.4
        fare_prob = 0.5 if row['fare'] <= 20 else 0.4
        sex_female_prob = 0.7 if row['sex_female'] else 0.3
        embarked_S_prob = 0.5 if row['embarked_S'] else 0.4
        alive_yes_prob = 0.6 if row['alive_yes'] else 0.4
        alone_True_prob = 0.5 if row['alone_True'] else 0.4
        adult_male_True_prob = 0.4 if row['adult_male_True'] else 0.6
        who_man_prob = 0.4 if row['who_man'] else 0.6
        class_Third_prob = 0.3 if row['class_Third'] else 0.5
        embark_town_Southampton_prob = 0.5 if row['embark_town_Southampton'] else 0.4

        # Combine the probabilities
        y = pclass_prob * age_prob * sibsp_prob * parch_prob * fare_prob * sex_female_prob * embarked_S_prob * alive_yes_prob * alone_True_prob * adult_male_True_prob * who_man_prob * class_Third_prob * embark_town_Southampton_prob

        # Normalize the probability
        y = y / (y + (1 - y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)