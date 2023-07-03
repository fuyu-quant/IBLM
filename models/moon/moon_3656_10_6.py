import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model for prediction
        # y = mx + c, where m is the slope and c is the intercept
        # We are assuming m1 and m2 as the slopes for Feature_1 and Feature_2 respectively
        # and c as the intercept. These values are assumed based on the given data.
        m1 = 0.5
        m2 = -0.4
        c = 0.6

        # Calculating the probability using the linear regression equation
        y = m1*row['Feature_1'] + m2*row['Feature_2'] + c

        # Converting the output to a probability between 0 and 1 using the sigmoid function
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)