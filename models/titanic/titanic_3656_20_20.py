import numpy as np

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Here we are using a simple rule-based approach to predict the target.
        # We are assuming that the target is more likely to be 1 if the passenger is female, is in first class, and is alone.
        # This is based on the historical fact that in the Titanic disaster, women, children, and first-class passengers were given priority for lifeboats.
        # We are also assuming that the target is more likely to be 0 if the passenger is male, is in third class, and is not alone.
        # This is based on the historical fact that men, third-class passengers, and families were less likely to survive.
        # The actual prediction is a weighted sum of these factors, with weights chosen to reflect their relative importance.
        # The weights were chosen arbitrarily and could be optimized for better performance.

        y = 0.0
        if row['sex_female'] == 1.0:
            y += 0.3
        if row['pclass'] == 1.0:
            y += 0.2
        if row['alone_True'] == 1.0:
            y += 0.1
        if row['sex_male'] == 1.0:
            y -= 0.3
        if row['pclass'] == 3.0:
            y -= 0.2
        if row['alone_False'] == 1.0:
            y -= 0.1

        # Ensure the prediction is within the range [0, 1]
        y = max(0.0, min(1.0, y))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)