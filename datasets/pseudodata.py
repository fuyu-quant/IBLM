import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.
        A, B, C, D = row[0], row[1], row[2], row[3]

        # Define the logic for prediction
        if A > 0 and B > 0 and C > 0 and D > 0:
            y = 1
        elif A < 0 and B < 0 and C < 0 and D < 0:
            y = 0
        elif A * B * C * D > 0:
            y = 1
        else:
            y = 0

        output.append(y)
    return np.array(output)