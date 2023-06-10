import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the sum of the absolute values of the first four columns (A, B, C, D)
        sum_abs_values = abs(row['A']) + abs(row['B']) + abs(row['C']) + abs(row['D'])

        # Calculate the average of the absolute values
        avg_abs_values = sum_abs_values / 4

        # Normalize the average by dividing it by the maximum possible average (assuming the maximum value for each column is 3)
        normalized_avg = avg_abs_values / 3

        # Calculate the probability of the target being 1 by using the normalized average
        y = 1 - normalized_avg

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)