import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        words = re.findall(r'\w+', text.lower())
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'favorite', 'enjoy', 'beautiful']
        negative_words = ['bad', 'worst', 'terrible', 'awful', 'boring', 'hate', 'disappointing', 'poor', 'annoying', 'ugly']

        positive_count = sum([1 for word in words if word in positive_words])
        negative_count = sum([1 for word in words if word in negative_words])

        if positive_count + negative_count == 0:
            y = 0.5
        else:
            y = positive_count / (positive_count + negative_count)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)