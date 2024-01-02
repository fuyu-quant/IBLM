from sklearn.base import BaseEstimator, ClassifierMixin
from .ibl import IBLModel


class IBLClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_name = 'gpt-4',
        prompt_file = 'classification_3.txt',
        prompt = None,
        seed = None
        ):
        self.model_name = model_name
        self.prompt_file = prompt_file
        self.prompt = prompt
        self.seed = seed
        self.ibl = IBLModel(self.model_name, self.prompt_file, self.prompt, self.seed)
        self.code_model = None

    def fit(self, X, y):
        self.code_model = self.ibl._classifier_train(X, y)
        return self

    def predict(self, X):
        return self.ibl._predict(self.code_model, X)

    def interpret(self):
        return self.ibl._interpret(self.code_model)

    def generate_python_script(self):
        return self.ibl._generate_python_script(self.code_model)


