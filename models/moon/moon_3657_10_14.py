import numpy as np
import pandas as pd

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple linear regression model to predict the target.
        # The coefficients of the model are estimated by minimizing the sum of the squared residuals.
        # The intercept is set to 0.5, which is the midpoint between 0 and 1.
        # The coefficients for Feature_1 and Feature_2 are estimated based on the observed data.
        # The predicted probability is then calculated as the dot product of the feature vector and the coefficient vector, plus the intercept.
        # The result is then passed through the sigmoid function to ensure that it lies between 0 and 1.

        intercept = 0.5
        coef = np.array([0.5, -0.5])  # Coefficients for Feature_1 and Feature_2
        y = 1 / (1 + np.exp(-(np.dot(row[['Feature_1', 'Feature_2']], coef) + intercept)))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)