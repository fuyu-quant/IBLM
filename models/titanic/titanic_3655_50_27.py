import numpy as np
def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # The logic here is to give more weightage to the features that are more likely to result in survival.
        # For example, 'pclass' is negatively correlated with survival (i.e., lower class means lower survival rate).
        # On the other hand, 'fare' and 'sex_female' are positively correlated with survival.
        # 'age' is slightly negatively correlated with survival, but the correlation is not very strong, so we give it less weight.
        # The other features are not very strongly correlated with survival, so we ignore them for simplicity.
        y = 0.3 * (3 - row['pclass']) + 0.3 * row['fare'] / 100 + 0.3 * row['sex_female'] - 0.1 * row['age'] / 80

        # Normalize the output to the range [0, 1]
        y = (y - 0) / (1 - 0)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)