import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model for prediction.
        # The coefficients are assumed based on the data.
        y = 0.3*row['a'] - 0.2*row['b'] + 0.4*row['c'] - 0.1*row['d']
        
        # Convert the output to probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)