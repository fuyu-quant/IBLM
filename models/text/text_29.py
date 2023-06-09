import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'good', 'love', 'best', 'favorite', 'enjoy', 'beautiful']
        negative_words = ['awful', 'terrible', 'bad', 'worst', 'disappointing', 'hate', 'boring', 'poor', 'waste', 'ridiculous']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        total_count = positive_count + negative_count

        if total_count == 0:
            y = 0.5
        else:
            y = positive_count / total_count

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)