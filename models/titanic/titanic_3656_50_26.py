import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are in first class, female, embarked from Cherbourg, and are alone.
        # These factors are chosen based on the general knowledge about the Titanic incident.
        # This is a very simple and naive prediction and may not give accurate results.
        # For more accurate results, a machine learning model should be trained using the data.

        p = 0.0
        if row['pclass'] == 1.0:
            p += 0.3
        if row['sex_female'] == 1.0:
            p += 0.3
        if row['embarked_C'] == 1.0:
            p += 0.2
        if row['alone_True'] == 1.0:
            p += 0.2

        # Do not change the code after this point.
        output.append(p)
    return np.array(output)