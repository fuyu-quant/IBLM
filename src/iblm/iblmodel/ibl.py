import numpy as np
import pandas as pd
from openai import OpenAI
import os
from importlib import resources

from ..utils import preprocessing
from ..utils import openai_model

import warnings
warnings.filterwarnings('ignore')

client = OpenAI()

class IBLModel():
    def __init__(
        self,
        model_name = None,
        prompt_file = None,
        prompt = None,
        seed = None
        ):

        self.model_name = model_name
        self.prompt_file = prompt_file
        self.prompt = prompt
        self.seed = seed
        self.code_model = None


    def _classifier_train(self, X, y):
        dataset_str, data_type = preprocessing._data_to_text(X, y)

        # create prompt
        if self.prompt != None:
            create_prompt = self.prompt.format(
                dataset_str_ = dataset_str,
                data_type_ = data_type
                )

        else:
            if self.prompt_file == 'classification_2.txt':
                with resources.open_text('iblm.iblmodel.prompt', self.prompt_file) as file:
                    self.prompt = file.read()
            elif self.prompt_file == 'classification_3.txt':
                with resources.open_text('iblm.iblmodel.prompt', self.prompt_file) as file:
                    self.prompt = file.read()

            create_prompt = self.prompt.format(
                dataset_str_ = dataset_str,
                data_type_ = data_type
                )

        code_model = openai_model._openai_model(self.model_name, create_prompt, self.seed)

        # prompt modification
        #modification_prompt = preprocessing._prompt_modification(code_model)

        #code_model = openai_model._openai_model(self.model_name, modification_prompt, self.seed)
        self.code_model = code_model

        return self.code_model

    def _predict(self, code, X):
        if self.code_model is None:
            raise Exception("You must train the model before predicting!")

        #code = self.code_model
        exec(code, globals())
        y = predict(X)
        return y



    def _interpret(self):
        if self.code_model is None:
            raise Exception("You must train the model before interpreting!")

        interpret_prompt = preprocessing._interpret_codemodel(self.code_model)
        return openai_model._openai_model(self.model_name, interpret_prompt, self.seed)

    def _generate_python_script(self, file_path, file_name):
        preprocessing._save_codemodel(file_path, file_name, self.code_model)
        return
