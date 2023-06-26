import numpy as np
import re

def predict(x):
    df = x.copy()
    output = []
    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        text = row['text']
        positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'best', 'love', 'good', 'favorite', 'enjoy', 'beautiful']
        negative_words = ['awful', 'terrible', 'worst', 'boring', 'disappointing', 'bad', 'hate', 'annoying', 'waste', 'ridiculous']

        positive_count = sum([1 for word in positive_words if word in text.lower()])
        negative_count = sum([1 for word in negative_words if word in text.lower()])

        # Calculate the sentiment score
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count + 1)

        # Normalize the sentiment score to a probability value between 0 and 1
        y = (sentiment_score + 1) / 2

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)