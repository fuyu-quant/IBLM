import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival statistics from the Titanic disaster
        # We also consider age, with children being more likely to survive
        # Fare is also considered, with higher fare indicating higher social-economic status and thus higher survival rate

        y = 0.0
        y += row['sex_female']
        y += row['class_First']
        y += row['embarked_C']
        if row['age'] <= 16:
            y += 0.5
        y += row['fare'] / 100

        # Normalize the output to range between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)