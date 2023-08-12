import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is that we are giving more weightage to the features which are more likely to result in survival.
        # For example, if the passenger is a female (sex_female=1), is in first class (class_First=1), and embarked from Cherbourg (embark_town_Cherbourg=1), 
        # then the chances of survival are high. Similarly, if the passenger is a male (sex_male=1), is in third class (class_Third=1), and embarked from Southampton (embark_town_Southampton=1), 
        # then the chances of survival are low.
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.2*row['embark_town_Cherbourg'] - 0.3*row['sex_male'] - 0.3*row['class_Third'] - 0.2*row['embark_town_Southampton']
        
        # The output is then passed through a sigmoid function to convert it into a probability between 0 and 1.
        y = 1 / (1 + np.exp(-y))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)