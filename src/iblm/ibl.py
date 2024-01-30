from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import metrics
import prompt

from exceptions import InvalidCodeModelError, InvalidModelObjectiveError, UndefinedCodeModelError
from llm_client import get_client, run_prompt


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


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
        task_prompt_mapping = dict(
            regression="prompt_templates/ibl/regression.j2",
            binary="prompt_templates/ibl/classification_3.j2",
            multiclass="prompt_templates/ibl/classification_3.j2",
            interpret="prompt_templates/interpret.j2",
        )

        # fit_prompt_templates
        with open(task_prompt_mapping.get(self.objective)) as file:
            self._default_fit_prompt_template = file.read()

        # interpret_prompt_templates
        with open(task_prompt_mapping.get("interpret")) as file:
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
        prompt_template: str | None = None,
        prompt_args: dict | None = None,
        try_code: bool = True,
    ) -> None:

        if prompt_template is None:
            prompt_template = self.default_fit_prompt_template

        if prompt_args is None:
            dataset_str, data_type = prompt.data_to_text(X, y)
            # ??? {{col_option}} ???
            prompt_args = dict(dataset_str=dataset_str, data_type=data_type)

        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.code_model = self._run_prompt(prompt=prompt_, seed=seed, temperature=temperature)
        self.fit_params = dict(temperature=temperature, seed=seed, prompt_template=prompt_template)

        if try_code:
            try:
                self.predict(X.head(1))
                logger.info("Valid code_model successfully created!")
            except InvalidCodeModelError:
                raise

    def predict(self, X: pd.DataFrame) -> None:
        if self.code_model is None:
            raise UndefinedCodeModelError("You must load or train the model before predict!")

        _code_space = {}

        try:
            exec(self.code_model, globals(), _code_space)
        except Exception as err:
            logger.exception("Error has occured while prediction")
            raise InvalidCodeModelError("Failed to execute `exec code_model`") from err

        try:
            y = _code_space["predict"](X)
        except Exception as err:
            logger.exception("Error has occured while prediction")
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
        return metrics.evaluate(y_true, y_pred, self.objective)

    def interpret(
        self,
        temperature: float = 0,
        seed: int | None = None,
        prompt_template: str | None = None,
        prompt_args: dict | None = None,
    ) -> None:
        if self.code_model is None:
            raise UndefinedCodeModelError("You must train the model before interpreting!")

        if prompt_template is None:
            prompt_template = self.default_interpret_prompt_template

        if prompt_args is None:
            prompt_args = dict(code_model=self.code_model)

        prompt_ = prompt.make_prompt(prompt_template=prompt_template, **prompt_args)

        self.interpret_result = self._run_prompt(prompt=prompt_, temperature=temperature, seed=seed)
