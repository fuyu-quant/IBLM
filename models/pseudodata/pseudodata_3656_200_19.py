import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Calculate the sum of the values in columns a, b, c, and d
        sum_values = row['a'] + row['b'] + row['c'] + row['d']
        
        # Calculate the average of the values in columns a, b, c, and d
        avg_values = sum_values / 4
        
        # If the average is greater than 0, assign a high probability (0.9)
        # If the average is less than or equal to 0, assign a low probability (0.1)
        y = 0.9 if avg_values > 0 else 0.1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)