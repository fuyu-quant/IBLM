from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class ICLClassifier():
    def __init__(
        self, 
        llm_model, 
        ):
        self.llm_model = llm_model
        self.icl_prompt = None


    def fit(self, x, y):
        df = x.copy()

        df['target'] = y


        # Create a string dataset
        dataset = []
        for index, row in df.iterrows():
            row_as_str = [str(item) for item in row.tolist()] 
            dataset.append(','.join(row_as_str))

        dataset_str = '\n'.join(dataset)


        icl_prompt = """
        Output the values according to all of the following conditions.
        ・The output should be numeric only.
        ・Do not output any text.
        ・Non-numeric output will result in an error.
        ・Predict the probability value as accurately as possible. Please be as detailed as possible.
        ・The rightmost column with a value of 0 or 1 is 'target'.
        ------------------
        {dataset_str_}
        ------------------
        ・Predict the probability that 'target' is 1 for the following data. 
        ------------------
        """.format(
            dataset_str_ = dataset_str
            )

        self.icl_prompt = icl_prompt

        print('Number of input tokens:' + len(icl_prompt))

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



