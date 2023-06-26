import numpy as np
import re
from collections import Counter

def predict(x):
    df = x.copy()
    output = []

    # Calculate the average target value for each word in the dataset
    word_target = {}
    for index, row in df.iterrows():
        words = re.findall(r'\w+', row['text'].lower())
        for word in words:
            if word not in word_target:
                word_target[word] = [row['target'], 1]
            else:
                word_target[word][0] += row['target']
                word_target[word][1] += 1

    for key in word_target:
        word_target[key] = word_target[key][0] / word_target[key][1]

    # Predict the target value for each text based on the average target values of its words
    for index, row in df.iterrows():
        words = re.findall(r'\w+', row['text'].lower())
        word_counts = Counter(words)
        total_weight = sum(word_counts.values())
        y = sum(word_target[word] * count for word, count in word_counts.items()) / total_weight
        output.append(y)

    return np.array(output)