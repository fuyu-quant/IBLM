import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # and lower probability for those who are male, in third class, and embarked from Southampton.
        # The age, sibsp, parch, fare are also considered in the prediction.
        # The weights for each feature are determined based on their importance in survival rate.
        y = 0.3*row['sex_female'] - 0.3*row['sex_male'] + 0.2*row['class_First'] - 0.2*row['class_Third'] + 0.1*row['embarked_C'] - 0.1*row['embarked_S'] + 0.05*row['age'] + 0.05*row['sibsp'] + 0.05*row['parch'] + 0.05*row['fare']
        y = 1 / (1 + np.exp(-y))  # Apply sigmoid function to convert the output into a probability between 0 and 1.

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)