import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        
        # Calculate the probability based on the given data
        age_factor = row['age'] / 100
        education_num_factor = row['education-num'] / 20
        capital_gain_factor = row['capital-gain'] / (row['capital-gain'] + 1)
        capital_loss_factor = row['capital-loss'] / (row['capital-loss'] + 1)
        hours_per_week_factor = row['hours-per-week'] / 100

        # Combine the factors to calculate the probability
        probability = age_factor * education_num_factor * capital_gain_factor * capital_loss_factor * hours_per_week_factor

        # Normalize the probability to be between 0 and 1
        y = probability / (1 + probability)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)