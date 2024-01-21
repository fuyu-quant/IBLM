import pandas as pd
import numpy as np

from openai import OpenAI

import logging

import prompt

import os

os.environ["OPENAI_API_KEY"] = "ABCDG"


logger = logging.getLogger(__name__)
# logging.basicConfig(format="%(asctime)s [%(name)s][%(levelname)s] (%(module)s:%(filename)s")
logging.basicConfig(format="%(asctime)s [%(name)s][%(levelname)s] (%(module)s:%(filename)s:%(funcName)s:%(lineno)d)")
logger.setLevel(logging.INFO)


client = OpenAI()


class IBLModel:
    IBL_OBJECTIVES = ("regression", "binary", "multiclass")

    def __init__(self, model_name: str, objective: str):
        self.model_name = model_name
        # self._objective = objective
        self.objective = objective

        self.load_prompt_templates(objective)

        self.code_model = None
        self.interpret_result = None

    def load_prompt_templates(self, objective: str) -> None:
        # fit_prompt_templates
        if self.objective == "regression":
            with open("prompt_templates/ibl/regression.txt", "r") as file:
                self._default_fit_prompt_template = file.read()
        elif self.objective == "binary":
            with open("prompt_templates/ibl/classification.txt", "r") as file:
                self._default_fit_prompt_template = file.read()
        elif self.objective == "multiclass":
            with open("prompt_templates/ibl/classification.txt", "r") as file:
                self._default_fit_prompt_template = file.read()

        # interpret_prompt_templates
        with open("prompt_templates/interpret.txt", "r") as file:
            self._default_interpret_prompt_template = file.read()

    @property
    def objective(self) -> str:
        return self._objective

    @objective.setter
    def objective(self, objective) -> None:
        if objective in self.IBL_OBJECTIVES:
            self._objective = objective
        else:
            raise Exception(f"specify the objective from {self.IBL_OBJECTIVES}")

        self.load_prompt_templates(objective)

    @property
    def default_fit_prompt_template(self) -> str:
        return self._default_fit_prompt_template

    @property
    def default_interpret_prompt_template(self) -> str:
        return self._default_interpret_prompt_template

    def _run_prompt(self, prompt: str, seed: int, temperature: float = 0) -> str:
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        seed: int,
        temperature: float = 0,
        prompt_template: str = None,
    ) -> None:

        if prompt_template is None:
            prompt_template = self.default_fit_prompt_template

        dataset_str, data_type = prompt._data_to_text(X, y)

        prompt_args = dict(dataset_str=dataset_str, data_type=data_type)

        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.code_model = self._run_prompt(prompt=prompt_, seed=seed, temperature=temperature)

    def predict(self, X: pd.DataFrame) -> None:
        if self.code_model is None:
            raise Exception("You must load or train the model before predict!")

        _code_space = {}

        try:
            exec(self.code_model, globals(), _code_space)
        except Exception:
            logger.exception("Failed to `exec import` code_model")
            raise

        try:
            y = _code_space["predict"](X)
        except Exception:
            logger.exception("Failed to execute `predict` function in code_model")
            raise

        return y

    def load_code_model(self, file_path: str) -> None:
        with open(file_path, mode="r") as file:
            self.code_model = file.read()

    def save_code_model(self, file_path: str) -> None:
        if self.code_model is None:
            raise Exception("You must train the model before interpreting!")

        with open(file_path, mode="w") as file:
            file.write(self.code_model)

    def evaluate(y: np.array) -> dict:
        pass

    def interpret(self, seed: int, temperature: float = 0, prompt_template: str = None) -> None:
        if self.code_model is None:
            raise Exception("You must train the model before interpreting!")

        if prompt_template is None:
            prompt_template = self.default_interpret_prompt_template

        prompt_args = dict(code_model=self.code_model)

        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.interpret_result = self._run_prompt(prompt=prompt_, seed=seed, temperature=temperature)


