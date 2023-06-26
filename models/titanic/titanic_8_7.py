import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        pclass_weight = 0.3 if row['pclass'] == 1 else 0.6 if row['pclass'] == 2 else 1
        age_weight = 0.5 if row['age'] <= 10 else 1 if row['age'] >= 60 else 0.8
        fare_weight = 0.5 if row['fare'] <= 10 else 1 if row['fare'] >= 50 else 0.8
        sex_weight = 0.7 if row['sex_female'] else 1
        embarked_weight = 0.8 if row['embarked_C'] else 1 if row['embarked_Q'] else 0.9
        alone_weight = 0.9 if row['alone_True'] else 1
        class_weight = 0.3 if row['class_First'] else 0.6 if row['class_Second'] else 1
        deck_weight = 0.5 if row['deck_A'] or row['deck_B'] or row['deck_C'] or row['deck_D'] or row['deck_E'] else 1
        
        # Calculate the final probability
        y = pclass_weight * age_weight * fare_weight * sex_weight * embarked_weight * alone_weight * class_weight * deck_weight
        y = 1 - (y / 10)
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)