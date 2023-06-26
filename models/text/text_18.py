import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'like', 'enjoy', 'best']
        negative_words = ['awful', 'terrible', 'horrible', 'worst', 'bad', 'hate', 'dislike', 'disappoint', 'boring']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        # Calculate the probability of target being 1
        y = (positive_count + 1) / (positive_count + negative_count + 2)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)