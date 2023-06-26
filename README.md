# IBLM:Inductive Bias Learning Models


- [What is IBLM](#what-is-iblm)
- [How to Use](#how-to-use)
    - [Setting](#setting)
    - [Binary classificatin](#binary-classification)
    - [Notebooks](#notebooks)
- [Supported Models](#supported-models)
- [Contributor](#contributor)
- [Backstory](#backstory)



## What is IBLM?
IBLM (Inductive Bias Learning) is a new machine learning modeling method that uses LLM to infer the structure of the model itself from the data set and outputs it as Python code. The learned model (code model) can be used as a machine learning model to predict a new dataset.


## How to Use

### Setting

* Installation
```python
pip install iblm
```
* OpenAI API key settings
```python
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```

### Binary classification

* Model Definition
```python
from iblm import IBLMClassifier

# Load LLM via LangChain. (GPT-4 recommended)
llm_model = OpenAI(temperature=0, model_name = 'gpt-4')

params = {'columns_name': True}

iblm = IBLMClassifier(llm_model = llm_model, params=params)
```

* Model Learning
```python
file_path = 'Specify the directory to output python files.'

model = iblm.fit(x_train, y_train, model_name = 'model_name', file_path=file_path)
```

* Model Predictions
```python
y_proba = iblm.predict(x_test)
```

### Notebooks
Use the link below to try it out immediately on Google colab.
- [Binary classification](https://github.com/fuyu-quant/IBLM/blob/release-maintenance/examples/iblmodel_halfmoon.ipynb)


## Supported Models
Currently, the recommended model is GPT-4


## Contributor
- [@t-ymbys](https://github.com/t-ymbys)
- [@cn47](https://github.com/cn47)


## Backstory
This idea is based on [langchain-tools](https://github.com/fuyu-quant/langchain-tools), which was created in an attempt to make LLM learn LightGBM.