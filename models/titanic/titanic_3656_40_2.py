import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # The logic here is to give higher probability for the passengers who are female, belong to first class, 
        # embarked from Cherbourg, and have deck B, C, D, or E. These are generally the passengers who had higher 
        # survival rate in the Titanic disaster.
        
        prob = 0.5  # start with a base probability of 0.5
        
        # increase the probability if the passenger is female
        if row['sex_female'] == 1.0:
            prob += 0.2
            
        # increase the probability if the passenger belongs to first class
        if row['class_First'] == 1.0:
            prob += 0.1
            
        # increase the probability if the passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            prob += 0.05
            
        # increase the probability if the passenger has deck B, C, D, or E
        if row['deck_B'] == 1.0 or row['deck_C'] == 1.0 or row['deck_D'] == 1.0 or row['deck_E'] == 1.0:
            prob += 0.05
            
        # decrease the probability if the passenger is male and belongs to third class
        if row['sex_male'] == 1.0 and row['class_Third'] == 1.0:
            prob -= 0.2
            
        # decrease the probability if the passenger embarked from Southampton
        if row['embark_town_Southampton'] == 1.0:
            prob -= 0.05
            
        # make sure the probability is within [0, 1]
        prob = max(0, min(1, prob))
        
        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)