import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching, sums and products of features, linear relationships
        y = A * B + C * D

        # Apply sigmoid function to map the result to a probability between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Binary classification threshold
        if y >= 0.5:
            output.append(1)
        else:
            output.append(0)

    return np.array(output)