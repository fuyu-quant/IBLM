import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that we are giving more weightage to the passengers who are female, 
        # belong to first class, embarked from Cherbourg and are adults. 
        # These factors are considered based on the historical data of the Titanic disaster where 
        # it was observed that females, first class passengers and adults had higher survival rates.
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.2*row['embark_town_Cherbourg'] + 0.2*row['who_adult']
        
        # Normalizing the output to be between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)