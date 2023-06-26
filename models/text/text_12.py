import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'amazing', 'great', 'good', 'love', 'best', 'wonderful', 'enjoy', 'favorite', 'beautiful']
        negative_words = ['awful', 'bad', 'terrible', 'worst', 'boring', 'disappointing', 'poor', 'waste', 'dislike', 'ugly']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        sentiment_score = positive_count - negative_count

        y = 1 / (1 + np.exp(-sentiment_score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)