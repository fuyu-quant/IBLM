import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen based on the observation that higher values of Feature_1 and lower values of Feature_2 are more likely to result in target 1.
        y = 0.6 * row['Feature_1'] - 0.4 * row['Feature_2']
        
        # Convert the linear regression output to a probability using the sigmoid function
        y = 1 / (1 + np.exp(-y))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)