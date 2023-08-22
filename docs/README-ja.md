# IBLM:Inductive-bias Learning Models
<div align="center">

[[ArXiv]](https://arxiv.org/abs/2308.09890)

</div>


- [What is IBL](#what-is-ibl)
- [How to Use](#how-to-use)
    - [Setting](#setting)
    - [Binary classificatin](#binary-classification)
    - [Notebooks](#notebooks)
- [Supported Models](#supported-models)
- [Contributor](#contributor)
- [Backstory](#backstory)



## IBLとは?
IBL（Inductive-bias Learning）は、LLMを用いてデータセットからモデル自体の構造を推測し、Pythonコードとして出力する新しい機械学習モデリング手法である。学習されたモデル（コードモデル）は、新たなデータセットを予測する機械学習モデルとして利用することができる。このリポジトリではIBLによるさまざまな学習方法を試すことができる．(現在は単純な方法での二値分類のみ利用可能)

![ibl](./images/ibl.png)


## 利用方法

### 設定

* インストール
```python
pip install iblm
```
* OpenAIのAPIkeyの設定 
```python
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
```

### 二値分類
現在は少量のデータでのみ実行が可能です．
同じseed値でも上手くcode modelを生成できる場合とできない場合があります．
* モデルの
```python
from iblm import IBLModel

# LangChain経由でLLMを読み込む(GPT-4の利用を推奨)
llm_model = OpenAI(temperature=0, model_name = 'gpt-4-0613')

params = {
    'columns_name': True,
    'objective': 'classification',
    }

iblm = IBLModel(llm_model = llm_model, params=params)
```

* モデルの学習
```python
model = iblm.fit(x_train, y_train)
```

* モデルの予測
```python
y_proba = iblm.predict(x_test)
```

### Notebooks
以下のリンクから、Google colabですぐに試すことができる。
- Binary classification:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblmodel/iblmodel_titanic.ipynb)


## 対応しているモデル
現在はGPT-4の利用を推奨としています．


## Contributor
- [@t-ymbys](https://github.com/t-ymbys)
- [@cn47](https://github.com/cn47)


## Backstory
この取り組みのアイデアは [langchain-tools](https://github.com/fuyu-quant/langchain-tools)に基づいています．このリポジトリはLLMにLightGBMを学習させるために作られました．