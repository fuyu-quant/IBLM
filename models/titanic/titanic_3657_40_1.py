import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        y = 0
        if row['sex_female'] == 1:
            y += 0.3
        if row['pclass'] == 1:
            y += 0.2
        if row['fare'] > 30:
            y += 0.1
        if row['age'] < 10 or row['age'] > 50:
            y += 0.1
        if row['sibsp'] == 0 and row['parch'] == 0:
            y += 0.1
        if row['embarked_C'] == 1:
            y += 0.1
        if row['alone_True'] == 1:
            y += 0.1
        y = min(y, 1)  # Ensure probability is not more than 1
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)