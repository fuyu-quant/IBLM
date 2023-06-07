import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row[0], row[1], row[2], row[3]

        # Conditional branching, sums and products of features, linear relationships
        if A > 0:
            y = A * D - B * C
        else:
            y = A * C + B * D

        # As many formulas as possible
        y += A * B * C * D
        y += A * B + C * D
        y += A * C + B * D
        y += A * D + B * C

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)