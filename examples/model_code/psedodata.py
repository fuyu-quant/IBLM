import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching
        if A > 0:
            y = A * 0.5 + B * 0.3 + C * 0.1 + D * 0.1
        else:
            y = A * 0.1 + B * 0.3 + C * 0.5 + D * 0.1

        # Sum of features
        y += A + B + C + D

        # Multiply features by a constant
        y *= 0.1

        # Linear relationships
        y = 0.5 * A + 0.3 * B + 0.1 * C + 0.1 * D

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)