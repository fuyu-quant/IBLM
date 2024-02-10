from tqdm import tqdm
import numpy as np
from importlib import resources

from ..utils import preprocessing

import warnings
warnings.filterwarnings('ignore')


class ICLModel():
    def __init__(
        self, 
        llm_model, 
        params
        ):
        self.llm_model = llm_model
        self.columns_name = params['columns_name']
        self.objective = params['objective']
        self.icl_prompt = None


    def fit(self, x, y):
        df = x.copy()
        df['target'] = y

        # Create a string dataset
        dataset_str = preprocessing.text_converter(df)

        # create prompt
        if prompt == None:
            if self.objective == 'regression':
                with resources.open_text('iblm.iclmodel.prompt', 'regression.txt') as file:
                    prompt = file.read()
            elif self.objective == 'classification':
                with resources.open_text('iblm.iclmodel.prompt', 'classification.txt') as file:
                    prompt = file.read()

        icl_prompt = prompt.format(dataset_str_ = dataset_str)
            
        self.icl_prompt = icl_prompt

        num_prompt = len(icl_prompt)
        print(f'Number of input tokens:{num_prompt}')

        return icl_prompt



    def predict(self, x):
        if self.icl_prompt is None:
            raise Exception("You must train the model before predicting!")

        output = []
        for _, row in tqdm(x.iterrows(), total=x.shape[0]):
            str_row = ','.join([str(elm) for elm in row.to_list()])
            prompt = self.icl_prompt + str_row

            while True:
                try:
                    y = self.llm_model(prompt)
                    y = float(y)
                    output.append(y)
                    break  
                except ValueError:
                    pass

        return np.array(output)



