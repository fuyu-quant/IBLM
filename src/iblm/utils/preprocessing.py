import pandas as pd

def text_converter(df):
    dataset = []
    for index, row in df.iterrows():
        row_as_str = [str(item) for item in row.tolist()] 
        dataset.append(','.join(row_as_str))
    dataset_str = '\n'.join(dataset)
    return dataset_str