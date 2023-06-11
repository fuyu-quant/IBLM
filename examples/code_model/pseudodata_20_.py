import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        A, B, C, D = row['A'], row['B'], row['C'], row['D']
        
        # Calculate the weighted sum of the features
        weighted_sum = A * 0.25 + B * 0.25 + C * 0.25 + D * 0.25
        
        # Apply the sigmoid function to the weighted sum to get the probability
        y = 1 / (1 + np.exp(-weighted_sum))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)