import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are in first class, female, younger and paid higher fare.
        # These are usually the people who are given priority during life-threatening situations.
        # The values are normalized to be between 0 and 1 by dividing by the maximum value in the dataset.
        pclass = 1 - row['pclass'] / df['pclass'].max()
        sex_female = row['sex_female']
        age = 1 - row['age'] / df['age'].max()
        fare = row['fare'] / df['fare'].max()

        # The final probability is a weighted sum of these factors.
        y = 0.3 * pclass + 0.3 * sex_female + 0.2 * age + 0.2 * fare

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)