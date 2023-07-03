import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, if the passenger is female, if the passenger is in first class, if the passenger is a child, 
        # if the passenger embarked from Cherbourg, if the passenger is alone, and if the passenger's deck is B, C, D, or E.
        # These features are chosen based on the historical data of the Titanic disaster.
        # The weights for these features are chosen arbitrarily and can be fine-tuned for better accuracy.
        
        y = 0.0
        y += row['sex_female'] * 0.3
        y += row['class_First'] * 0.2
        y += row['who_child'] * 0.2
        y += row['embark_town_Cherbourg'] * 0.1
        y += row['alone_True'] * 0.1
        y += row['deck_B'] * 0.05
        y += row['deck_C'] * 0.05
        y += row['deck_D'] * 0.05
        y += row['deck_E'] * 0.05

        # Normalize the output to be between 0 and 1
        y = min(max(y, 0.0), 1.0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)