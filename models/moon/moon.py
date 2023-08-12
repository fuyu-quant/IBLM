import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Based on the given data, it seems that when Feature_1 is greater than 0 and Feature_2 is less than 0, 
        # the target is more likely to be 1. Conversely, when Feature_1 is less than 0 and Feature_2 is greater than 0, 
        # the target is more likely to be 0. We can use these observations to make a simple prediction.
        if row['Feature_1'] > 0 and row['Feature_2'] < 0:
            y = 0.9  # High probability for target 1
        elif row['Feature_1'] < 0 and row['Feature_2'] > 0:
            y = 0.1  # Low probability for target 1
        else:
            y = 0.5  # Equal probability for target 0 and 1 if none of the above conditions are met

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)