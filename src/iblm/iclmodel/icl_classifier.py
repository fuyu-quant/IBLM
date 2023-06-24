from langchain.llms import OpenAI
from tqdm import tqdm
import re

import numpy as np

import warnings
warnings.filterwarnings('ignore')


class ICLClassifier():
    def __init__(
        self, 
        llm_model_name, 
        ):
        self.llm_model_name = llm_model_name
        self.llm_model = OpenAI(temperature=0, model_name = self.llm_model_name)

        #self.llm_model = llm_model,
        self.icl_prompt = None


    def fit(self, x, y):
        print("> Start of code creating.")
        df = x.copy()

        df['target'] = y


        # Create a string dataset
        dataset = []
        for index, row in df.iterrows():
            row_as_str = [str(item) for item in row.tolist()] 
            dataset.append(','.join(row_as_str))

        dataset_str = '\n'.join(dataset)


        icl_prompt = """
        Predict the 'target' according to the following conditions.
        ・Please make your predictions as accurate as possible.
        ・No machine learning algorithms are used.
        ・Predict 'target' based on the following data only.The rightmost column with a value of 0 or 1 is 'target'.
        ------------------
        {dataset_str_}
        ------------------
        ・Predict the probability that 'target' is 1 for the following data. The output should be a probability value only.
        ------------------
        """.format(
            dataset_str_ = dataset_str
            )

        self.icl_prompt = icl_prompt

        print(len(icl_prompt))

        return icl_prompt



    def predict(self, x):
        if self.icl_prompt is None:
            raise Exception("You must train the model before predicting!")

        output = []
        for _, row in tqdm(x.iterrows(), total=x.shape[0]):
        #for _, row in x.iterrows():
            str_row = ','.join([str(elm) for elm in row.to_list()])
            prompt = self.icl_prompt + str_row

            y = self.llm_model(prompt)
            print(y)
            y = float(y)
            
            output.append(y)

        return np.array(output)


