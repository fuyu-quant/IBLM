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

    for index, row in df.iterrows():
        # Do not change the code before this point.
        # Please describe the process required to make the prediction below.

        # Calculate the average target value for words in the given text
        words = re.findall(r'\w+', row['text'].lower())
        word_counts = Counter(words)
        total_weight = 0
        total_count = 0
        for word, count in word_counts.items():
            if word in word_target:
                total_weight += word_target[word] * count
                total_count += count

        # Calculate the probability value based on the average target value of words in the text
        if total_count > 0:
            y = total_weight / total_count
        else:
            y = 0.5

        # Do not change the code after this point.
        output.append(y)
    return np.array(output)