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