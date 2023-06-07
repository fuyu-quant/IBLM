import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching of features and linear formulas
        if A > 0 and C < 0:
            y = 0.8 * A - 0.6 * C
        elif A < 0 and C > 0:
            y = 0.6 * B - 0.8 * D
        elif A > 0 and C > 0:
            y = 0.4 * A + 0.4 * C
        else:
            y = 0.4 * B + 0.4 * D

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)