import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():

        # Feature creation and data preprocessing
        A, B, C, D = row['A'], row['B'], row['C'], row['D']

        # Conditional branching and linear relational expressions
        if A > 0 and C < 0:
            y = 0
        elif A < 0 and C > 0:
            y = 1
        elif B > 0 and D < 0:
            y = 0
        elif B < 0 and D > 0:
            y = 1
        else:
            y = 0 if A + B + C + D > 0 else 1

        output.append(y)
    return np.array(output)