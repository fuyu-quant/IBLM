import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that we are giving more weightage to the passengers who are female, 
        # belong to first class, embarked from Cherbourg and are adults. 
        # These are the passengers who have a higher chance of survival based on the data.
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.2*row['embark_town_Cherbourg'] + 0.2*row['who_adult']
        
        # We are subtracting the weightage of passengers who are male, belong to third class, embarked from Southampton and are children. 
        # These are the passengers who have a lower chance of survival based on the data.
        y -= 0.3*row['sex_male'] + 0.3*row['class_Third'] + 0.2*row['embark_town_Southampton'] + 0.2*row['who_child']
        
        # We are normalizing the output to be between 0 and 1 using the sigmoid function.
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)