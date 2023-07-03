import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Here we are using a simple linear regression model to predict the target.
        # The coefficients are chosen based on the observation that higher values of Feature_1 and lower values of Feature_2 tend to result in target 1.
        # The intercept is chosen to be 0.5 so that when both features are 0, the predicted probability is 0.5.
        y = 0.5 + 0.3 * row['Feature_1'] - 0.2 * row['Feature_2']
        
        # The predicted probability should be between 0 and 1.
        y = max(0, min(1, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)