import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the average of A, B, C, and D
        avg = (row['A'] + row['B'] + row['C'] + row['D']) / 4
        
        # Calculate the probability based on the average
        y = 1 / (1 + np.exp(-avg))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)