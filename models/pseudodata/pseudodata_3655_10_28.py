import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Calculate the sum of the absolute values of the first four columns
        sum_abs = abs(row['a']) + abs(row['b']) + abs(row['c']) + abs(row['d'])
        
        # Calculate the sum of the squares of the first four columns
        sum_squares = row['a']**2 + row['b']**2 + row['c']**2 + row['d']**2
        
        # Calculate the average of the first four columns
        avg = (row['a'] + row['b'] + row['c'] + row['d']) / 4
        
        # Calculate the product of the first four columns
        product = row['a'] * row['b'] * row['c'] * row['d']
        
        # Calculate the probability based on the above four values
        y = (sum_abs + sum_squares + avg + product) / 4
        
        # Normalize the probability to be between 0 and 1
        y = (y - min(y, 0)) / (max(y, 1) - min(y, 0))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)