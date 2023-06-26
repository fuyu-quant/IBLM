import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given features
        pclass_weight = 0.3 if row['pclass'] == 1 else 0.6 if row['pclass'] == 2 else 0.1
        age_weight = 0.5 if row['age'] <= 30 else 0.3 if row['age'] <= 60 else 0.2
        fare_weight = 0.4 if row['fare'] <= 10 else 0.3 if row['fare'] <= 30 else 0.2 if row['fare'] <= 60 else 0.1
        sex_weight = 0.6 if row['sex_female'] else 0.4
        embarked_weight = 0.5 if row['embarked_C'] else 0.3 if row['embarked_Q'] else 0.2
        alone_weight = 0.6 if row['alone_True'] else 0.4
        adult_male_weight = 0.6 if row['adult_male_True'] else 0.4
        class_weight = 0.4 if row['class_First'] else 0.3 if row['class_Second'] else 0.3
        deck_weight = 0.1 if row['deck_A'] else 0.2 if row['deck_B'] else 0.3 if row['deck_C'] else 0.1 if row['deck_D'] else 0.1 if row['deck_E'] else 0.1 if row['deck_F'] else 0.1
        
        # Calculate the final probability
        y = pclass_weight * age_weight * fare_weight * sex_weight * embarked_weight * alone_weight * adult_male_weight * class_weight * deck_weight
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)