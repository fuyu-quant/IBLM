import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.

        # Initialize the probability of target being 1
        prob = 0

        # If the passenger is a woman, increase the probability
        if row['sex'] == 'female':
            prob += 0.6

        # If the passenger is in first or second class, increase the probability
        if row['pclass'] in [1, 2]:
            prob += 0.3

        # If the passenger is a child, increase the probability
        if row['age'] < 18:
            prob += 0.2

        # If the passenger is not alone, increase the probability
        if not row['alone']:
            prob += 0.1

        # If the passenger is an adult male, decrease the probability
        if row['adult_male']:
            prob -= 0.4

        # If the passenger embarked from Cherbourg, increase the probability
        if row['embark_town'] == 'Cherbourg':
            prob += 0.1

        # If the passenger embarked from Queenstown, decrease the probability
        if row['embark_town'] == 'Queenstown':
            prob -= 0.1

        # Clip the probability between 0 and 1
        y = np.clip(prob, 0, 1)

        output.append(y)
    return