import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # We will use a simple rule-based system to predict the target.
        # The rules are based on the data provided and general knowledge about the Titanic disaster.
        # For example, we know that women and children were more likely to survive, and that first class passengers had a higher survival rate.

        # Initialize the probability to 0.5 (neutral)
        prob = 0.5

        # Increase probability if the passenger is a woman or a child
        if row['who_woman'] == 1.0 or row['who_child'] == 1.0:
            prob += 0.2

        # Increase probability if the passenger is in first class
        if row['class_First'] == 1.0:
            prob += 0.1

        # Decrease probability if the passenger is a man in third class
        if row['who_man'] == 1.0 and row['class_Third'] == 1.0:
            prob -= 0.2

        # Make sure the probability stays within the [0, 1] range
        prob = max(0.0, min(1.0, prob))

        y = prob

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)