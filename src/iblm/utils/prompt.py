from __future__ import annotations

import textwrap

from typing import TYPE_CHECKING

import jinja2


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def make_prompt(prompt_template: str, **kwargs) -> str:
    template = jinja2.Template(prompt_template)
    prompt = template.render(**kwargs)
    return textwrap.dedent(prompt)


def data_to_text(X: pd.DataFrame, y: np.array):
    df = X.copy()
    df["target"] = y

    ", ".join(df.dtypes.astype(str))

    dataset = []
    for _index, row in df.iterrows():
        row_as_str = [str(item) for item in row.tolist()]
        dataset.append(",".join(row_as_str))
    dataset_str = "\n".join(dataset)
    return dataset_str
