import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given features
        pclass_factor = 0.9 if row['pclass'] == 1 else 0.7 if row['pclass'] == 2 else 0.5
        age_factor = 0.9 if row['age'] <= 16 else 0.7 if row['age'] <= 40 else 0.5
        fare_factor = 0.9 if row['fare'] >= 50 else 0.7 if row['fare'] >= 20 else 0.5
        sex_factor = 0.9 if row['sex_female'] else 0.5
        embarked_factor = 0.9 if row['embarked_C'] else 0.7 if row['embarked_Q'] else 0.5
        alone_factor = 0.9 if row['alone_True'] else 0.7
        
        # Combine the factors to calculate the final probability
        y = pclass_factor * age_factor * fare_factor * sex_factor * embarked_factor * alone_factor
        
        # Normalize the probability to be between 0 and 1
        y = (y - 0.5) / 0.4
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)