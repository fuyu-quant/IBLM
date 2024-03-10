# IBLM: Inductive-bias Learning Models

<div align="center">

[[ArXiv]](https://arxiv.org/abs/2308.09890)

</div>

- [IBLとは](#iblとは)
- [例](#例)
- [使用方法](#使用方法)
- [Inductive-bias Learning Models](#inductive-bias-learning-models)
- [対応しているLLM](#対応しているllm)
- [貢献者](#貢献者)
- [背景](#背景)

## IBLとは
IBL（Inductive-bias Learning）は、LLM（大規模言語モデル）を使用して、データセットからモデルの構造を推論し、Pythonコードとして出力する新しい機械学習モデリング手法です。学習したモデル（Code Model）は、新しいデータセットを予測するための予測モデルとして使用できます。このリポジトリでは、IBLを使用した様々な学習方法を試すことができます。

![ibl](./images/ibl.png)

* 現在、バイナリ分類のみがサポートされています。

## 例
以下のリンクを使用して、Google Colabで直ちに試すことができます。
- バイナリ分類
  - IBL
    - OpenAI:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblmodel/pseudodata_openai.ipynb)
    - Claude:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblmodel/pseudodata_claude.ipynb)

  - IBLbagging
    - OpenAI:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblbagging/pseudodata_openai.ipynb)
    - Claude:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/IBLM/blob/main/examples/iblbagging/pseudodata_claude.ipynb)

## 使用方法

- インストールとインポート
```python
pip install iblm

import iblm
```

- 設定
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

- モデルの学習\
    現在は、少量のデータのみ実行できます。
    ```python
    code_model = ibl.fit(x_train, y_train)

    print(code_model)
    ```

- モデルの予測
    ```python
    y_proba = ibl.predict(x_test)
    ```

## Inductive-bias Learning Model

- Inductive-bias Learning\
通常の意味論バイアス学習
  ```python
  from iblm import IBLBaggingModel

  iblm = IBLModel(
      api_type="openai",
      model_name="gpt-4-0125-preview",
      objective="binary"
      )
  ```

- Inductive-bias Learning Bagging\
与えられたデータセットからデータをサンプリングし、複数のモデルを作成し、これらのモデルの平均を予測値として使用します。
  ```python
  from iblm import IBLBaggingModel

  iblbagging = IBLBaggingModel(
      api_type="openai",
      model_name="gpt-4-0125-preview",
      objective="binary",
      num_model=20,  # 作成するモデルの数
      max_sample = 2000,  # データセットからのサンプル数の最大値
      min_sample = 300,　　# データセットからのサンプル数の最小値
      )
  ```

## 対応しているLLM
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

## 貢献者
- [@t-ymbys](https://github.com/t-ymbys)
- [@cn47](https://github.com/cn47)


## 引用
このリポジトリが役にたつと思ったら，以下の論文を引用してください．
```
@article{tanaka2023inductive,
  title={Inductive-bias Learning: Generating Code Models with Large Language Model},
  author={Tanaka, Toma and Emoto, Naofumi and Yumibayashi, Tsukasa},
  journal={arXiv preprint arXiv:2308.09890},
  year={2023}
}
```
