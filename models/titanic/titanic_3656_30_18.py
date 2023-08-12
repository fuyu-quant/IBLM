import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for the passengers who are female, belong to first class, 
        # embarked from Cherbourg, travelling alone and are adults. These factors are chosen based on the general 
        # survival statistics of the Titanic disaster.
        y = 0.1 * row['sex_female'] + 0.1 * row['class_First'] + 0.1 * row['embarked_C'] + 0.1 * row['alone_True'] + 0.1 * row['who_woman']
        
        # Subtracting factors that are likely to decrease the survival rate
        y -= 0.1 * row['sex_male'] + 0.1 * row['class_Third'] + 0.1 * row['embarked_S'] + 0.1 * row['alone_False'] + 0.1 * row['who_man']
        
        # Normalizing the output to range between 0 and 1
        y = (y + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)