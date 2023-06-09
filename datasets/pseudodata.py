import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        A, B, C, D = row['A'], row['B'], row['C'], row['D']
        
        # Based on the given data, we can observe that when A and B are positive and C and D are negative, the target is more likely to be 1.
        # Similarly, when A and B are negative and C and D are positive, the target is more likely to be 0.
        # We can use this logic to predict the probability of the target being 1.
        
        if A > 0 and B > 0 and C < 0 and D < 0:
            y = 0.9
        elif A < 0 and B < 0 and C > 0 and D > 0:
            y = 0.1
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)