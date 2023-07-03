import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        # Here we are using a simple linear regression model for prediction
        # We are assuming that the target is a linear combination of the features a, b, c, d
        # The coefficients of the linear combination are determined by the mean values of the features for target 1 and 0
        mean_values_target_1 = df[df['target'] == 1].mean()
        mean_values_target_0 = df[df['target'] == 0].mean()
        coefficients = mean_values_target_1 - mean_values_target_0
        y = coefficients['a']*row['a'] + coefficients['b']*row['b'] + coefficients['c']*row['c'] + coefficients['d']*row['d']
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to map the result to a probability between 0 and 1
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)