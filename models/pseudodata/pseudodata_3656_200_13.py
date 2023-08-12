import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Calculate the sum of the values in the row
        row_sum = row['a'] + row['b'] + row['c'] + row['d']
        
        # Normalize the sum to a value between 0 and 1
        y = (row_sum + 10) / 20
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)