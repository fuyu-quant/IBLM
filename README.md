# IBLM:Inductive-bias Learning Models
<div align="center">

[[ArXiv]](https://arxiv.org/abs/2308.09890)

</div>

- [What is IBL](#what-is-ibl)
- [Examples](#examples)
- [How to Use](#how-to-use)
- [Inductive-bias Learning Models](#inductive-bias-learning-models)
- [Supported LLMs](#supported-llms)
- [Contributor](#contributor)
- [Backstory](#backstory)



## What is IBL?
IBL (Inductive-bias Learning) is a new machine learning modeling method that uses LLM to infer the structure of the model itself from the data set and outputs it as Python code. The learned model (code model) can be used as a machine learning model to predict a new dataset.In this repository, you can try different learning methods with IBL.(Currently only binary classification with simple methods is available.)

![ibl](./images/ibl.png)

* Currently, only binary classification is supported.

## Examples
Use the link below to try it out immediately on Google colab.
- Binary classification
  - IBL
    - OpenAI:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblmodel/pseudodata_openai.ipynb)
    - Claude:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblmodel/pseudodata_claude.ipynb)

  - IBLbagging
    - OpenAI:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblbagging/pseudodata_openai.ipynb)
    - Claude:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblbagging/pseudodata_claude.ipynb)


## How to Use

-  Installation and Import
```python
pip install iblm

import iblm
```

- Setting
  - OpenAI
    ```python
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

    ibl = iblm.IBLModel(api_type="openai", model_name="gpt-4-0125-preview", objective="binary")
    ```

  - Azure OpenAI
    ```python
    os.environ["AZURE_OPENAI_KEY"] = "YOUR_API_KEY"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "xxx"
    os.environ["OPENAI_API_VERSION"] = "xxx"

    ibl = iblm.IBLModel(api_type="azure", model_name="gpt-4-0125-preview", objective="binary")
    ```

  - Google API
    ```python
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
    ibl = iblm.IBLModel(api_type="gemini", model_name="gemini-pro", objective="binary")
    ```

  - Anthropic API
    ```python
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_API_KEY"
    ibl = iblm.IBLModel(api_type="", model_name="", objective="binary")
    ```

-  Model Learning\
Currently, only small amounts of data can be executed.
    ```python
    code_model = ibl.fit(x_train, y_train)

    print(code_model)
    ```

-  Model Predictions
    ```python
    y_proba = ibl.predict(x_test)
    ```

## Inductive-bias Learning Models

- Inductive-bias Learning\
Normal Inductive-bias Learning
  ```python
  from iblm import IBLBaggingModel

  iblm = IBLModel(
      api_type="openai",
      model_name="gpt-4-0125-preview",
      objective="binary"
      )
  ```

- Inductive-bias Learning bagging\
Sampling data from a given dataset, we create multiple models, and the average of these models is used as the predicted value.
  ```python
  from iblm import IBLBaggingModel

  iblbagging = IBLBaggingModel(
      api_type="openai",
      model_name="gpt-4-0125-preview",
      objective="binary",
      num_model=20,  # Number of models to create
      max_sample = 2000,  # Maximum number of samples from the data set
      min_sample = 300,　　# Minimum number of samples from the data set
      )
  ```


## Supported LLMs
- OpenAI
  - gpt-4-0125-preview
  - gpt-3.5-turbo-0125
- Azure OpenAI
  - gpt-4-0125-preview
  - gpt-3.5-turbo-0125
- Google
  - gemini-pro
- Anthropic
  - claude-3-opus-20240229
  - claude-3-sonnet-20240229



## Contributor
- [@t-ymbys](https://github.com/t-ymbys)
- [@cn47](https://github.com/cn47)


## Cite
If you find this repo helpful, please cite the following papers:
```
@article{tanaka2023inductive,
  title={Inductive-bias Learning: Generating Code Models with Large Language Model},
  author={Tanaka, Toma and Emoto, Naofumi and Yumibayashi, Tsukasa},
  journal={arXiv preprint arXiv:2308.09890},
  year={2023}
}
```
