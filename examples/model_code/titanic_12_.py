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

        # Check if passenger is a female
        if sex == 'female':
            prob += 0.6

        # Check if passenger is in first class
        if pclass == 1:
            prob += 0.3
        elif pclass == 2:
            prob += 0.1

        # Check if passenger is a child
        if age <= 16:
            prob += 0.3

        # Check if passenger is not alone
        if not alone:
            prob += 0.1

        # Check if passenger embarked from Cherbourg
        if embarked == 'C':
            prob += 0.1

        # Normalize probability to be between 0 and 1
        prob = min(max(prob, 0), 1)

        # Do not change the code after this point.
        output.append(prob)
    return np.array(output)