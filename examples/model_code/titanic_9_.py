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
        adult_male = row['adult_male']
        alone = row['alone']

        # Initialize probability
        prob = 0

        # Consider the effect of pclass
        if pclass == 1:
            prob += 0.6
        elif pclass == 2:
            prob += 0.4
        else:
            prob += 0.2

        # Consider the effect of sex
        if sex == 'female':
            prob += 0.35
        else:
            prob -= 0.35

        # Consider the effect of age
        if age <= 16:
            prob += 0.1
        elif age >= 60:
            prob -= 0.1

        # Consider the effect of fare
        if fare >= 50:
            prob += 0.1
        elif fare <= 10:
            prob -= 0.1

        # Consider the effect of embarked
        if embarked == 'C':
            prob += 0.05
        elif embarked == 'Q':
            prob -= 0.05

        # Consider the effect of adult_male
        if adult_male:
            prob -= 0.1

        # Consider the effect of alone
        if alone:
            prob -= 0.05

        # Normalize the probability to be between 0 and 1
        prob = max(0, min(1, prob))

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)