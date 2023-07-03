import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are some of the factors that were found to have higher survival rates in the Titanic disaster
        # The age, fare, and number of siblings/spouses/parents/children are also considered
        # The weights for each factor are determined based on their perceived importance
        
        y = 0.3 * row['sex_female'] + 0.2 * row['class_First'] + 0.1 * row['embarked_C'] - 0.1 * row['age'] / 80 - 0.1 * row['fare'] / 500 - 0.1 * row['sibsp'] / 8 - 0.1 * row['parch'] / 6
        
        # The resulting value is then scaled to be between 0 and 1 using the sigmoid function
        y = 1 / (1 + np.exp(-y))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)