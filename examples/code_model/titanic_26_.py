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

        # Consider the factors that increase the probability of survival
        if pclass == 1:
            prob += 0.3
        if sex == 'female':
            prob += 0.35
        if age <= 15:
            prob += 0.1
        if fare >= 50:
            prob += 0.1
        if embarked == 'C':
            prob += 0.05
        if who == 'woman':
            prob += 0.05
        if not adult_male:
            prob += 0.03
        if not alone:
            prob += 0.02

        # Normalize the probability to be between 0 and 1
        prob = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)