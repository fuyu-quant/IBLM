import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        a = row['a']
        b = row['b']
        c = row['c']
        d = row['d']
        
        # Here we use a simple linear regression model for prediction
        # The coefficients are assumed based on the data distribution
        y = 0.3*a + 0.1*b + 0.2*c + 0.4*d
        
        # Convert the output to probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)