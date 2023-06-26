import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the probability based on the given features
        pclass_weight = 0.3 if row['pclass'] == 1 else 0.6 if row['pclass'] == 2 else 0.1
        age_weight = 0.5 if row['age'] <= 18 else 0.3 if row['age'] <= 40 else 0.2
        fare_weight = 0.4 if row['fare'] <= 10 else 0.3 if row['fare'] <= 30 else 0.2 if row['fare'] <= 50 else 0.1
        sex_weight = 0.6 if row['sex_female'] else 0.4
        embarked_weight = 0.5 if row['embarked_C'] else 0.3 if row['embarked_Q'] else 0.2
        alone_weight = 0.6 if row['alone_True'] else 0.4
        class_weight = 0.5 if row['class_First'] else 0.3 if row['class_Second'] else 0.2

        # Combine the weights to calculate the final probability
        y = pclass_weight * age_weight * fare_weight * sex_weight * embarked_weight * alone_weight * class_weight

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)