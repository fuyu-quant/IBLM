import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the sum of the absolute values of the first four columns
        sum_abs = np.sum(np.abs(row[:4]))

        # Calculate the average of the sum
        avg_sum = sum_abs / 4

        # Normalize the average by dividing it by the maximum possible sum (assuming the range of each column is -3 to 3)
        normalized_avg = avg_sum / 6

        # Calculate the probability using a sigmoid function
        y = 1 / (1 + np.exp(-normalized_avg))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)