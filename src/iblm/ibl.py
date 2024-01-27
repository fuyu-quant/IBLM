from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import prompt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from exceptions import InvalidCodeModelError, InvalidModelObjectiveError, UndefinedCodeModelError
from llm_client import get_client, run_prompt


logger = logging.getLogger(__name__)
# logging.basicConfig(format="%(asctime)s [%(name)s][%(levelname)s] (%(module)s:%(filename)s")
logging.basicConfig(format="%(asctime)s [%(name)s][%(levelname)s] (%(module)s:%(filename)s:%(funcName)s:%(lineno)d)")
logger.setLevel(logging.INFO)


class IBLModel:
    IBL_OBJECTIVES = ("regression", "binary", "multiclass")

    def __init__(
        self,
        model_name: str,
        objective: str,
        # common
        api_type: str = "openai",
        # openai & azure
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: int = 120,
        organization: str | None = None,
        # azure
        api_version: str | None = None,
        azure_endpoint: str | None = None,
    ) -> None:

        self.model_name = model_name
        self.objective = objective
        self.client = get_client(
            # common
            api_type=api_type,
            # openai & azure
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            organization=organization,
            # azure
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

        self.load_prompt_templates(objective)

        self.code_model = None
        self.interpret_result = None

        self.fit_params = None

    def load_prompt_templates(self, objective: str) -> None:
        # fit_prompt_templates
        if self.objective == "regression":
            with open("prompt_templates/ibl/regression.j2") as file:
                self._default_fit_prompt_template = file.read()
        elif self.objective == "binary":
            with open("prompt_templates/ibl/classification_3.j2") as file:
                self._default_fit_prompt_template = file.read()
        elif self.objective == "multiclass":
            with open("prompt_templates/ibl/classification_3.j2") as file:
                self._default_fit_prompt_template = file.read()

        # interpret_prompt_templates
        with open("prompt_templates/interpret.j2") as file:
            self._default_interpret_prompt_template = file.read()

    @property
    def objective(self) -> str:
        return self._objective

    @objective.setter
    def objective(self, objective) -> None:
        if objective in self.IBL_OBJECTIVES:
            self._objective = objective
        else:
            raise InvalidModelObjectiveError(f"specify the objective from {self.IBL_OBJECTIVES}")

        self.load_prompt_templates(objective)

    @property
    def default_fit_prompt_template(self) -> str:
        return self._default_fit_prompt_template

    @property
    def default_interpret_prompt_template(self) -> str:
        return self._default_interpret_prompt_template

    def _run_prompt(self, prompt: str, temperature: float = 0, seed: int | None = None) -> str:
        return run_prompt(
            client=self.client,
            model_name=self.model_name,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        temperature: float = 0,
        seed: int | None = None,
        prompt_template: str = None,
    ) -> None:

        if prompt_template is None:
            prompt_template = self.default_fit_prompt_template

        dataset_str, data_type = prompt._data_to_text(X, y)
        prompt_args = dict(dataset_str=dataset_str, data_type=data_type)
        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.code_model = self._run_prompt(prompt=prompt_, seed=seed, temperature=temperature)

        self.fit_params = dict(temperature=temperature, seed=seed, prompt_template=prompt_template)

    def predict(self, X: pd.DataFrame) -> None:
        if self.code_model is None:
            raise UndefinedCodeModelError("You must load or train the model before predict!")

        _code_space = {}

        try:
            exec(self.code_model, globals(), _code_space)
        except Exception as err:
            raise InvalidCodeModelError("Failed to execute `exec code_model`") from err

        try:
            y = _code_space["predict"](X)
        except Exception as err:
            raise InvalidCodeModelError("Failed to execute `predict` function in code_model") from err

        return y

    def load_code_model(self, file_path: str) -> None:
        with open(file_path) as file:
            self.code_model = file.read()

    def save_code_model(self, file_path: str) -> None:
        if self.code_model is None:
            raise UndefinedCodeModelError("You must train the model before saving!")

        with open(file_path, mode="w") as file:
            file.write(self.code_model)

    def evaluate(self, y_true: np.array, y_pred: np.array) -> dict:
        if self.objective == "regression":
            metric_dict = {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
            }
        elif self.objective == "binary":
            y_prob = y_pred
            y_pred = np.where(y_prob > 0.5, 1, 0)
            metric_dict = {
                "roc_auc": roc_auc_score(y_true, y_prob),
                "pr_auc": average_precision_score(y_true, y_prob),
                "accuracy": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
            }
        elif self.objective == "multiclass":
            # TODO: add metrics
            metric_dict = {}

        return metric_dict

    def interpret(
        self,
        temperature: float = 0,
        seed: int | None = None,
        prompt_template: str = None,
    ) -> None:
        if self.code_model is None:
            raise UndefinedCodeModelError("You must train the model before interpreting!")

        if prompt_template is None:
            prompt_template = self.default_interpret_prompt_template

        prompt_args = dict(code_model=self.code_model)
        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.interpret_result = self._run_prompt(prompt=prompt_, temperature=temperature, seed=seed)
