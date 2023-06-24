import numpy as np
import re
from collections import Counter

def predict(x):
    df = x.copy()
    output = []

    # Calculate the average sentiment score for each word in the dataset
    word_sentiments = Counter()
    total_words = 0
    for index, row in df.iterrows():
        words = re.findall(r'\w+', row['text'].lower())
        sentiment = row['target']
        for word in words:
            word_sentiments[word] += sentiment
            total_words += 1

    # Calculate the average sentiment score for each word
    for word in word_sentiments:
        word_sentiments[word] /= total_words

    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the sentiment score for the given text
        words = re.findall(r'\w+', row['text'].lower())
        sentiment_score = 0
        for word in words:
            sentiment_score += word_sentiments.get(word, 0)

        # Normalize the sentiment score
        sentiment_score /= len(words)

        # Convert the sentiment score to a probability value
        y = 1 / (1 + np.exp(-sentiment_score))

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)