import textwrap
import pandas as pd
import numpy as np


def make_prompt(prompt_template: str, **kwargs) -> str:
    prompt = prompt_template.format(**kwargs)
    return textwrap.dedent(prompt)[1:-1].encode("utf-8")


def _data_to_text(X: pd.DataFrame, y: np.array):
    df = X.copy()
    df['target'] = y

    data_type = ', '.join(df.dtypes.astype(str))

    dataset = []
    for index, row in df.iterrows():
        row_as_str = [str(item) for item in row.tolist()]
        dataset.append(','.join(row_as_str))
    dataset_str = '\n'.join(dataset)
    return dataset_str, data_type

