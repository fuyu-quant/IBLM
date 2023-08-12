import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # We also consider age, with younger passengers being more likely to survive
        # Fare is also considered, with higher fare indicating higher social-economic status and thus higher survival rate

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embarked_C']
        y -= row['age'] / 80  # assuming age is between 0 and 80
        y += row['fare'] / 500  # assuming fare is between 0 and 500

        # Normalize the output to between 0 and 1
        y = (y + 2) / 4

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)