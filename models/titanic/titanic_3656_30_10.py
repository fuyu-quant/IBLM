import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival factors from the Titanic disaster
        # The age and fare are also considered, younger and higher fare are more likely to survive
        # The values are normalized to be between 0 and 1 by dividing by the maximum value

        y = 0.0
        y += row['sex_female']
        y += row['class_First'] * 0.8
        y += row['embarked_C'] * 0.6
        y += row['age'] / df['age'].max() * 0.4
        y += row['fare'] / df['fare'].max() * 0.2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)