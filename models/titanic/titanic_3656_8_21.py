import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give higher probability for those who are female, in first class, and younger.
        # These are based on the known survival facts from the Titanic disaster.
        # This is a very simplistic model and in a real-world scenario, more sophisticated machine learning models should be used.

        prob = 0.5  # start with a base probability

        # increase probability if passenger is female
        if row['sex_female'] == 1.0:
            prob += 0.3

        # increase probability if passenger is in first class
        if row['pclass'] == 1.0:
            prob += 0.1

        # increase probability if passenger is a child
        if row['who_child'] == 1.0:
            prob += 0.1

        # decrease probability if passenger is an adult male
        if row['adult_male_True'] == 1.0:
            prob -= 0.2

        # decrease probability if passenger is in third class
        if row['pclass'] == 3.0:
            prob -= 0.1

        # make sure probability stays within [0,1]
        prob = max(0, min(prob, 1))

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)