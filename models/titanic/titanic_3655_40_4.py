import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # We also consider age, with younger passengers having a slightly higher chance of survival
        # Fare is also considered, with higher fare indicating higher social-economic status and thus higher survival rate

        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.3
        y += row['embark_town_Cherbourg'] * 0.2
        y -= row['age'] / 100
        y += row['fare'] / 100

        # Ensure the probability is within [0,1]
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)