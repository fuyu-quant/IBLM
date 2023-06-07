from langchain.llms import OpenAI
import re


class Classification():
    def __init__(self, llm_model, params):
        self.llm_model = llm_model,
        self.columns_name = params['columns_name']
        pass

    def train(self, x, y, file_path=None):
        print("> Start of model training.")
        df = x.copy()

        # yのデータをtargetとしてxのdataframeに加える
        df['target'] = y

        # 二値分類か多値分類かを判定
        if len(df['target'].unique()) == 2:
            task_type = 'binary classification'
            output_code = 'y = 1 / (1 + np.exp(-y))'
        else:
            task_type = 'multi-class classification'

        # データ型の取得
        data_type = ', '.join(df.dtypes.astype(str))



        # 文字列のdatasetを作成
        dataset = []
        for index, row in df.iterrows():
            row_as_str = [str(item) for item in row.tolist()]  # 各要素を文字列に変換
            dataset.append(','.join(row_as_str))

        # リスト全体を改行文字で結合
        dataset_str = '\n'.join(dataset)


        # ハイパーパラメータの設定
        if self.columns_name:
            col_name = ', '.join(df.columns.astype(str))

        create_prompt = """
        Please create your code in compliance with all of the following conditions. Output should be code only. Do not enclose the output in ``python ``` or the like.
        ・Analyze the large amount of data below and create a {task_type_} code to accurately predict "target".
        ------------------
        {dataset_str_}
        ------------------
        ・Each data type is as follows. If necessary, you can change the data type.
        ・Create code that can make predictions about new data based on logic from large amounts of input data without using machine learning models.
        ・If input is available, the column names below should also be used to help make decisions when creating the predictive model. Column Name:{col_name_}
        ・Create a code like the following. Do not change the input or output format.
        ------------------
        import numpy as np

        def model(x):
            df = x.copy()

            output = []
            for index, row in df.iterrows():


                # Feature creation and data preprocessing


                {output_code_}
                output.append(y)
                
            return output
        """.format(
            task_type_ = task_type,
            dataset_str_ = dataset_str,
            col_name_ = col_name,
            output_code_ = output_code
            )

        code = self.llm_model(create_prompt)

        name = re.search(r'def "(.*?)"', code).group(1)

        # Save to File
        if file_path_ != None:
            with open(folder_path_ + f'{name}.py', mode='w') as file:
                file.write(code)

        # モデルの訓練処理
        return code



    def predict(self, x_test):
        y_pred = 
        return y_pred




    def interpret(self):


        interpretprompt = """

        """

        output = self.llm_model(interpretprompt)

        return output

