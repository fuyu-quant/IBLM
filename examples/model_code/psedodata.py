import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching and linear relational expressions
        if A > 0 and B < 0 and C > 0:
            y = 1
        elif A < 0 and B > 0 and C < 0:
            y = 0
        elif A * B * C * D > 0:
            y = 1
        elif A * B * C * D < 0:
            y = 0
        else:
            y = 0.5

        # Complex relational expressions
        if A > 0 and B > 0 and C > 0 and D > 0:
            y = 1 - y

        # Sigmoid function to convert y to probability
        y = 1 / (1 + np.exp(-y))
        output.append(y)
    return np.array(output)