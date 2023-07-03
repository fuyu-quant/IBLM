import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model for prediction
        # We are assuming that the target is a linear combination of the features a, b, c, d
        # The coefficients of the linear combination are determined by observing the data
        y = 0.3*row['a'] - 0.2*row['b'] + 0.4*row['c'] - 0.1*row['d']
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to map the result to a probability between 0 and 1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)