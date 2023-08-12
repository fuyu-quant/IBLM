import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that if the passenger is female, her survival probability is high.
        # If the passenger is male, his survival probability is low.
        # We are also considering the passenger class, assuming that first class passengers have higher survival probability.
        # This is a very basic approach and may not give accurate results for all cases.

        if row['sex_female'] == 1.0:
            if row['pclass'] == 1.0:
                y = 0.9
            elif row['pclass'] == 2.0:
                y = 0.8
            else:
                y = 0.7
        else:
            if row['pclass'] == 1.0:
                y = 0.4
            elif row['pclass'] == 2.0:
                y = 0.3
            else:
                y = 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)