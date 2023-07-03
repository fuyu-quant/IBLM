import numpy as np
import pandas as pd

# Define the data
data = [[1.342,-0.412,1.0],
[2.029,0.302,1.0],
[0.532,-0.396,1.0],
[0.021,0.333,1.0],
[1.731,-0.241,1.0],
[0.753,-0.613,1.0],
[1.957,0.304,1.0],
[1.209,-0.53,1.0],
[1.689,-0.229,1.0],
[0.06,0.525,1.0],
[0.731,-0.436,1.0],
[0.256,-0.106,1.0],
[1.516,-0.398,1.0],
[0.749,-0.513,1.0],
[0.084,0.316,1.0],
[1.643,-0.379,1.0],
[1.28,-0.498,1.0],
[1.998,0.356,1.0],
[0.986,-0.413,1.0],
[0.095,0.099,1.0]]

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2', 'target'])

# Calculate the mean and standard deviation of the features for each target value
mean_1 = df[df['target'] == 1.0][['Feature_1', 'Feature_2']].mean()
std_1 = df[df['target'] == 1.0][['Feature_1', 'Feature_2']].std()

# Define the prediction function
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Calculate the probability of the target being 1.0 using the Gaussian distribution formula
        prob_1 = 1 / (np.sqrt(2 * np.pi * std_1**2)) * np.exp(- (row[['Feature_1', 'Feature_2']] - mean_1)**2 / (2 * std_1**2))
        # The probability of the target being 1.0 is the product of the probabilities for each feature
        y = prob_1['Feature_1'] * prob_1['Feature_2']
        output.append(y)
    return np.array(output)