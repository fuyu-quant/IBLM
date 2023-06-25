import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        a, b, c, d, _ = row
        y = (a + b + c + d) / 4
        y = (y + 2) / 4  # Normalize the value to be between 0 and 1

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)