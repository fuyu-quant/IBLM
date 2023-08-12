import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple heuristic to predict the target.
        # We are taking the sum of all the features and if the sum is greater than 0, we predict the target as 1 (high probability) else we predict the target as 0 (low probability).
        # This is a very simple heuristic and may not give accurate results for complex datasets.
        # For more accurate results, a machine learning model should be trained on the data.
        
        y = 1 if row['a'] + row['b'] + row['c'] + row['d'] > 0 else 0

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)