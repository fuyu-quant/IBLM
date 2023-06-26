import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass = row['pclass']
        age = row['age']
        fare = row['fare']
        sex_female = row['sex_female']
        embarked_S = row['embarked_S']
        alive_yes = row['alive_yes']
        alone_True = row['alone_True']
        adult_male_True = row['adult_male_True']
        
        # Weights for each feature
        weights = {
            'pclass': -0.2,
            'age': -0.01,
            'fare': 0.01,
            'sex_female': 0.5,
            'embarked_S': -0.1,
            'alive_yes': 0.8,
            'alone_True': -0.1,
            'adult_male_True': -0.3
        }
        
        # Calculate the weighted sum
        weighted_sum = (
            pclass * weights['pclass'] +
            age * weights['age'] +
            fare * weights['fare'] +
            sex_female * weights['sex_female'] +
            embarked_S * weights['embarked_S'] +
            alive_yes * weights['alive_yes'] +
            alone_True * weights['alone_True'] +
            adult_male_True * weights['adult_male_True']
        )
        
        # Calculate the probability using the sigmoid function
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)