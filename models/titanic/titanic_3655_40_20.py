import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, if the passenger is female, in first class, and embarked from Cherbourg, they have a higher chance of survival.
        # Similarly, if the passenger is a child, they also have a higher chance of survival.
        # The fare is also considered, assuming that passengers who paid more had a higher chance of survival.
        # The weights for these features are determined based on their perceived importance.
        
        y = 0.3*row['sex_female'] + 0.3*row['class_First'] + 0.2*row['embark_town_Cherbourg'] + 0.1*row['who_child'] + 0.1*(row['fare']/df['fare'].max())
        
        # The result is then normalized to be between 0 and 1.
        y = (y - df.min()) / (df.max() - df.min())
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)