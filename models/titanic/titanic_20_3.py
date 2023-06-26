import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Prediction logic
        score = 0
        
        # Higher class (1st class) increases the probability of survival
        if row['pclass'] == 1:
            score += 0.3
        
        # Female passengers have a higher probability of survival
        if row['sex_female'] == 1:
            score += 0.5
        
        # Passengers with lower age have a higher probability of survival
        if row['age'] <= 10:
            score += 0.3
        elif row['age'] <= 30:
            score += 0.2
        
        # Passengers with fewer siblings/spouses have a higher probability of survival
        if row['sibsp'] == 0:
            score += 0.1
        
        # Passengers with fewer parents/children have a higher probability of survival
        if row['parch'] == 0:
            score += 0.1
        
        # Normalize the score to get the probability of survival (target = 1)
        y = min(score, 1)
        
        # Do not change the code after this point.
        output.append(y)
    return np.array(output)