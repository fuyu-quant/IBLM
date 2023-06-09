import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Please describe the process required to make the prediction below.
        
        # If the passenger is a female, has a fare greater than 50, and is not in the third class, predict 1
        if row['sex'] == 'female' and row['fare'] > 50 and row['pclass'] != 3:
            y = 1
        # If the passenger is a male, younger than 10, and is not in the third class, predict 1
        elif row['sex'] == 'male' and row['age'] < 10 and row['pclass'] != 3:
            y = 1
        # If the passenger is a female, younger than 18, and is in the first or second class, predict 1
        elif row['sex'] == 'female' and row['age'] < 18 and row['pclass'] != 3:
            y = 1
        # If the passenger is a male, older than 50, and is in the first class, predict 1
        elif row['sex'] == 'male' and row['age'] > 50 and row['pclass'] == 1:
            y = 1
        # Otherwise, predict 0
        else:
            y = 0

        output.append(y)
    return np.array(output)