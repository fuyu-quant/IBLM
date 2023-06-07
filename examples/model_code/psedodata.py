import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching of features and linear relationships
        if A > 0 and B > 0:
            y = A * 0.5 + B * 0.3 - C * 0.2 + D * 0.1
        elif A < 0 and B < 0:
            y = A * 0.3 + B * 0.5 - C * 0.1 + D * 0.2
        elif A > 0 and B < 0:
            y = A * 0.4 - B * 0.3 - C * 0.1 + D * 0.2
        else:
            y = A * 0.3 - B * 0.4 - C * 0.2 + D * 0.1

        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)