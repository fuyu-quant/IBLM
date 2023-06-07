from langchain.llms import OpenAI

from inductivebiaslearning import InductiveBiasLearning



class IBL:
    def __init__(self, llm_model):
        self.model = None
        self.code = None
        self.llm_model = llm_model

    def train(self, x_train, y_train):
        self.model = Classification(self.llm_model)
        self.code = self.model.fit(x_train, y_train)
        return self.code

    def predict(self, x_test):
        if self.model is None:
            raise Exception("You must train the model before predicting!")
        return self.model.predict(x_test)

    def interpret(self):
        if self.code is None:
            raise Exception("You must train the model before interpreting!")
        return self.code.interpret()
