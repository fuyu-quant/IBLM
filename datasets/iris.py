import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.
        sepal_length = row[0]
        sepal_width = row[1]
        petal_length = row[2]
        petal_width = row[3]

        if petal_length <= 2.0:
            y = 0
        elif petal_length > 2.0 and petal_length <= 4.9:
            y = 1
        else:
            y = 2

        output.append(y)
    return np.array(output)