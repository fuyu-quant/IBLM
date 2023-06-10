import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass = row['pclass']
        sex = row['sex']
        age = row['age']
        fare = row['fare']
        embarked = row['embarked']
        who = row['who']
        adult_male = row['adult_male']
        alone = row['alone']

        # Initialize probability
        prob = 0

        # Consider the factors that may affect the probability of survival
        if pclass == 1:
            prob += 0.3
        elif pclass == 2:
            prob += 0.2

        if sex == 'female':
            prob += 0.35

        if age <= 16:
            prob += 0.1
        elif age >= 60:
            prob -= 0.1

        if fare >= 50:
            prob += 0.1

        if embarked == 'C':
            prob += 0.05

        if who == 'child':
            prob += 0.1

        if not adult_male:
            prob += 0.05

        if alone:
            prob -= 0.05

        # Clip the probability between 0 and 1
        prob = np.clip(prob, 0, 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)