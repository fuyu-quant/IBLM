import pandas as pd

def text_converter(df):
    dataset = []
    for index, row in df.iterrows():
        row_as_str = [str(item) for item in row.tolist()] 
        dataset.append(','.join(row_as_str))
    dataset_str = '\n'.join(dataset)
    return dataset_str


def columns_name(columns_name, df):
    if columns_name:
        col_name = ', '.join(df.columns.astype(str))
        col_option = ''
    else:
        # serial number
        df.columns = range(df.shape[1])
        col_name = ', '.join(df.columns.astype(str))
        col_option = 'df.columns = range(df.shape[1])'
    return col_name, col_option


def prompt_modification(code_model):
    modification_prompt = """
        Please extract and output only the Python code from the following.
        Do not put ```python ```, etc. into the output.
        -------------
        {code_model_}
        """.format(code_model_ = code_model)
    return modification_prompt


def save_codemodel(file_path, model_name, code_model):
    if file_path != None:
        with open(file_path + f'{model_name}.py', mode='w') as file:
            file.write(code_model)
    return


def interpret_codemodel(code_model):
    interpret_prompt = """
    Refer to the code below and explain how you are going to process the data and make predictions.
    The only part to explain is the part where the data is processed.
    Do not explain df = x.copy().
    Please output the data in bulleted form.
    Please tell us what you can say based on the whole process.
    ------------------
    {code_model_}
    """.format(code_model_ = code_model)

    return interpret_prompt