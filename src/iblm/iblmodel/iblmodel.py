from langchain.callbacks import get_openai_callback
import numpy as np
import pandas as pd
from importlib import resources

from ..utils import preprocessing

import warnings
warnings.filterwarnings('ignore')


class IBLModel():
    def __init__(
        self, 
        llm_model,
        params,
        mode,
        ):
        self.llm_model = llm_model
        self.columns_name = params['columns_name']
        self.mode = mode
        self.code_model = None


    def fit(self, x, y, prompt = None, model_name = None, file_path = None):
        df = x.copy()
        df['target'] = y

        # Obtaining data types
        data_type = ', '.join(df.dtypes.astype(str))

        # Create a string dataset
        dataset_str = preprocessing.text_converter(df)

        # column name
        if self.columns_name:
            col_name = ', '.join(df.columns.astype(str))
            col_option = ''
        else:
            # serial number
            df.columns = range(df.shape[1])
            col_name = ', '.join(df.columns.astype(str))
            col_option = 'df.columns = range(df.shape[1])'


        # create prompt
        if prompt == None:
            if self.mode == 'regression':
                with resources.open_text('iblm.iblmodel.prompt', 'regression.txt') as file:
                    prompt = file.read()
            elif self.mode == 'classification':
                with resources.open_text('iblm.iblmodel.prompt', 'classification.txt') as file:
                    prompt = file.read()

        create_prompt = prompt.format(
            dataset_str_ = dataset_str,
            data_type_ = data_type,
            col_name_ = col_name,
            col_option_ = col_option
            )

        code_model = self.llm_model(create_prompt)

        # fixing prompts
        modification_prompt = """
        Please extract and output only the Python code from the following.
        Do not put ```python ```, etc. into the output.
        -------------
        {code_model_}
        """.format(code_model_ = code_model)

        code_model = self.llm_model(modification_prompt)

        # Save to File
        if file_path != None:
            with open(file_path + f'{model_name}.py', mode='w') as file:
                file.write(code_model)

        self.code_model = code_model

        return code_model



    def predict(self, x):
        if self.code_model is None:
            raise Exception("You must train the model before predicting!")

        code = self.code_model
        exec(code, globals())
        y = predict(x)
        return y



    def interpret(self):
        if self.code_model is None:
            raise Exception("You must train the model before interpreting!")

        interpret_prompt = """
        Refer to the code below and explain how you are going to process the data and make predictions.
        The only part to explain is the part where the data is processed.
        Do not explain df = x.copy().
        Please output the data in bulleted form.
        Please tell us what you can say based on the whole process.
        ------------------
        {code_model_}
        """.format(
            code_model_ = self.code_model
        )

        with get_openai_callback() as cb:
            output = self.llm_model(interpret_prompt)
            print(cb)


        return output