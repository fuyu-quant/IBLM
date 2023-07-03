import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, in first class, and embarked from Cherbourg
        # as they have higher survival rate based on the data. The age, fare, and number of siblings/spouses also play a role in the survival rate.
        # The logic can be adjusted based on the actual data analysis.

        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.2
        y += row['embarked_C'] * 0.1
        y += row['fare'] / 100 * 0.1
        y += row['sibsp'] / 10 * 0.1
        y -= row['age'] / 100 * 0.1

        # Limit the probability between 0 and 1
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)