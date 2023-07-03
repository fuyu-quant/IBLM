import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.
        
        # Initialize the score to 0
        score = 0
        
        # Increase the score if the passenger is female
        if row['sex_female'] == 1.0:
            score += 0.3
            
        # Increase the score if the passenger is in first class
        if row['pclass'] == 1.0:
            score += 0.2
            
        # Increase the score if the passenger embarked from Cherbourg
        if row['embark_town_Cherbourg'] == 1.0:
            score += 0.1
            
        # Decrease the score if the passenger is alone
        if row['alone_True'] == 1.0:
            score -= 0.1
            
        # Decrease the score if the passenger is an adult male
        if row['adult_male_True'] == 1.0:
            score -= 0.2
            
        # Decrease the score if the passenger is in third class
        if row['pclass'] == 3.0:
            score -= 0.2
            
        # Convert the score to a probability using the sigmoid function
        y = 1 / (1 + np.exp(-score))
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)