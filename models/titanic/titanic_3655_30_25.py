import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, in first class, and embarked from Cherbourg
        # as these categories of passengers had higher survival rates in the Titanic disaster.
        # The age, fare, and number of siblings/spouses aboard are also considered.
        # The weights for each category are determined based on their importance in survival.

        y = 0.3*row['sex_female'] + 0.2*row['class_First'] + 0.1*row['embarked_C'] - 0.1*row['age']/50 - 0.1*row['fare']/100 - 0.1*row['sibsp']

        # The probability is calculated by applying the sigmoid function to the result.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)