import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on general knowledge about the Titanic incident, and may not reflect the actual data
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.3*row['embark_town_Cherbourg'] + 0.1*row['fare']

        # Normalize the probability to be between 0 and 1
        y = max(min(y, 1), 0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)