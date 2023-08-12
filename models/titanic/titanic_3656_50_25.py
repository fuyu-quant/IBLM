import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is that we are giving more weightage to the passengers who are female, who are children, who are travelling in first class, who have embarked from Cherbourg and who are alone. These factors are considered based on the historical data of the Titanic disaster where women, children and first class passengers were given priority during the rescue.

        y = 0.3*row['sex_female'] + 0.3*row['who_child'] + 0.2*row['class_First'] + 0.1*row['embark_town_Cherbourg'] + 0.1*row['alone_True']

        # Normalizing the output to be between 0 and 1
        y = y / (0.3 + 0.3 + 0.2 + 0.1 + 0.1)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)