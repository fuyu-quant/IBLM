import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # These conditions are based on the historical data of the Titanic disaster where women, children, and first-class passengers had higher survival rates.
        # This is a simple heuristic and does not guarantee accurate results for all cases.
        
        prob = 0.0
        if row['sex_female'] == 1.0:
            prob += 0.3
        if row['class_First'] == 1.0:
            prob += 0.3
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.3
        if row['age'] < 10.0:
            prob += 0.1
        if prob > 1.0:
            prob = 1.0

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)