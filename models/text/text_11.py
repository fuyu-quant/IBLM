import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'beautiful', 'enjoy', 'favorite']
        negative_words = ['bad', 'worst', 'awful', 'terrible', 'hate', 'boring', 'disappointing', 'poor', 'annoying', 'waste']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        sentiment_score = positive_count - negative_count

        if sentiment_score > 0:
            y = 1
        elif sentiment_score < 0:
            y = 0
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)