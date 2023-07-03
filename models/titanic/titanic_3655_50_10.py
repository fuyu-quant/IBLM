import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are based on the known survival factors from the Titanic disaster
        # The age, fare, and number of siblings/spouses/parents/children are also considered
        # The coefficients for each factor are determined based on their perceived impact on survival
        y = 0.3*row['sex_female'] + 0.2*row['class_First'] + 0.1*row['embarked_C'] - 0.1*row['age']/50 - 0.05*row['fare']/100 - 0.05*row['sibsp'] - 0.05*row['parch']

        # The output is then passed through a sigmoid function to ensure it's between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)