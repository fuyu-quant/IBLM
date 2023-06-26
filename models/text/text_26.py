import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'astonishing', 'amazing', 'wonderful', 'great', 'good', 'love', 'enjoy', 'best', 'favorite']
        negative_words = ['awful', 'terrible', 'horrible', 'worst', 'boring', 'disappointing', 'hate', 'annoying', 'ridiculous', 'stupid']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        # Calculate the probability of the target being 1 based on the counts of positive and negative words
        if positive_count + negative_count == 0:
            y = 0.5
        else:
            y = positive_count / (positive_count + negative_count)

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)