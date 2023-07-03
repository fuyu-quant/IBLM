import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Based on the given data, it seems that when Feature_1 is high and Feature_2 is low, the target is likely to be 1.
        # Conversely, when Feature_1 is low and Feature_2 is high, the target is likely to be 0.
        # Therefore, we can use a simple heuristic to predict the target: if Feature_1 > Feature_2, predict 1, else predict 0.
        # To convert this into a probability, we can use the sigmoid function, which maps any real number into the range [0, 1].
        # The sigmoid function is defined as sigmoid(x) = 1 / (1 + exp(-x)).
        # We can compute x as the difference between Feature_1 and Feature_2, so that a large positive difference gives a high probability,
        # and a large negative difference gives a low probability.

        x = row['Feature_1'] - row['Feature_2']
        y = 1 / (1 + np.exp(-x))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)