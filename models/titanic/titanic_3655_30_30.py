import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and embarked from Cherbourg
        # These are just assumptions based on general knowledge about the Titanic incident, 
        # in a real scenario, we would need to perform an exploratory data analysis to understand the data better
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.3*row['embark_town_Cherbourg'] + 0.1*row['fare']

        # Normalizing the output to be between 0 and 1
        y = 1 / (1 + np.exp(-y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)