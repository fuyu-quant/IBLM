from langchain.llms import OpenAI

from inductivebiaslearning import InductiveBiasLearning



class IBL:
    def __init__(self, llm_model):
        self.model = None
        self.llm_model = llm_model

    def train(self, x_train, y_train):
        self.model = InductiveBiasLearning(self.llm_model)
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        if self.model is None:
            raise Exception("You must train the model before predicting!")
        return self.model.predict(x_test)
