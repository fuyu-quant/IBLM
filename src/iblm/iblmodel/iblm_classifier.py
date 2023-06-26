from langchain.callbacks import get_openai_callback
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class IBLMClassifier():
    def __init__(
        self, 
        llm_model,
        params
        ):
        self.llm_model = llm_model
        self.columns_name = params['columns_name']
        self.model_code = None


    def fit(self, x, y, model_name, file_path=None):
        df = x.copy()
        df['target'] = y


        # Obtaining data types
        data_type = ', '.join(df.dtypes.astype(str))


        # Create a string dataset
        dataset = []
        for index, row in df.iterrows():
            row_as_str = [str(item) for item in row.tolist()] 
            dataset.append(','.join(row_as_str))
        dataset_str = '\n'.join(dataset)


        # column name
        if self.columns_name:
            col_name = ', '.join(df.columns.astype(str))
            col_option = ''

        else:
            # serial number
            df.columns = range(df.shape[1])
            col_name = ', '.join(df.columns.astype(str))
            col_option = 'df.columns = range(df.shape[1])'



        create_prompt = """
        Please create your code in compliance with all of the following conditions. Output should be code only. Do not enclose the output in ``python ``` or the like.
        ・The following data are for "target" of 0 and 1, respectively. Analyze these data and create a python code to predict the probability that the "target" of the unknown data is 1.
        ------------------
        {dataset_str_}
        ------------------
        ・Create a code that predicts a high probability value when "target" is 1 and a low probability value when "target" is 0 for the data given above.
        ・Each data type is as follows. If necessary, you can change the data type.
        ------------------
        {data_type_}
        ------------------
        ・The column names, in order, are as follows {col_name_}
        ・Think and code the logic to predict probability values based on the data without using a existing machine learning model.
        ・Please make your predictions as accurate as possible.
        ・Predicting probability values as finely as possible increases overall accuracy.
        ・If {col_option_} is not blank, add it after 'df = x.copy()'.
        ・You do not need to provide examples.
        ・Create a code like the following.
        ------------------
        import numpy as np
        def predict(x):
            df = x.copy()
            output = []
            for index, row in df.iterrows():
                # Do not change the code before this point.
                # Please describe the process required to make the prediction below.


                # Do not change the code after this point.
                output.append(y)
            return np.array(output)
        """.format(
            dataset_str_ = dataset_str,
            data_type_ = data_type,
            col_name_ = col_name,
            col_option_ = col_option
            )

        with get_openai_callback() as cb:
            model_code = self.llm_model(create_prompt)
            print(cb)


        # Save to File
        if file_path != None:
            with open(file_path + f'{model_name}.py', mode='w') as file:
                file.write(model_code)


        self.model_code = model_code

        return model_code



    def predict(self, x):
        if self.model_code is None:
            raise Exception("You must train the model before predicting!")

        code = self.model_code

        exec(code, globals())

        y = predict(x)
        return y




    def interpret(self):
        if self.model_code is None:
            raise Exception("You must train the model before interpreting!")

        interpret_prompt = """
        Refer to the code below and explain how you are going to process the data and make predictions.
        The only part to explain is the part where the data is processed.
        Do not explain df = x.copy().
        Please output the data in bulleted form.
        Please tell us what you can say based on the whole process.
        ------------------
        {model_code_}
        """.format(
            model_code_ = self.model_code
        )

        with get_openai_callback() as cb:
            output = self.llm_model(interpret_prompt)
            print(cb)


        return output
