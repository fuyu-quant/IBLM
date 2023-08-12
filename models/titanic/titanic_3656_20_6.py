import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give higher probability for survival if the passenger is a female, in first class, and embarked from Cherbourg.
        # This is based on the historical data that women, children, and the upper-class passengers were given priority during the evacuation.
        # The age, fare, and number of siblings/spouses/parents/children are also considered.
        # The values are normalized to be between 0 and 1.
        
        p = 0.0
        p += row['sex_female']
        p += row['class_First']
        p += row['embarked_C']
        p -= row['age'] / 80
        p += row['fare'] / 500
        p -= row['sibsp'] / 8
        p -= row['parch'] / 6
        
        # The probability is then clipped to be between 0 and 1.
        p = max(0.0, min(1.0, p))
        
        y = p

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)