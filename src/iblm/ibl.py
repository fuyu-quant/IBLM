from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import re

from typing import TYPE_CHECKING

from iblm.exceptions import InvalidCodeModelError, InvalidModelObjectiveError, UndefinedCodeModelError
from iblm.utils.llm_client import get_client, run_prompt
from iblm.utils.metrics import evaluate
from iblm.utils.prompt import data_to_text, make_prompt


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


logger = logging.getLogger(__name__)
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
        model_prompt_mapping = {
            "gpt-4-0125-preview": "iblm.prompt_templates.gpt-4-0125-preview.ibl",
            "gpt-3.5-turbo-0125": "iblm.prompt_templates.gpt-35-turbo-0125.ibl",
            "gemini-pro": "iblm.prompt_templates.gemini-pro.ibl",
            "claude-3-opus-20240229": "iblm.prompt_templates.claude-3-opus-20240229.ibl",
            "claude-3-sonnet-20240229": "iblm.prompt_templates.claude-3-sonnet-20240229.ibl",
        }

        task_prompt_mapping = {
            "regression": "regression.j2",
            "binary": "binary.j2",
            "multiclass": "binary.j2",  # TODO: change after multiclass.j2
            "interpret": "interpret.j2",
        }

        # fit_prompt_templates
        with pkg_resources.open_text(
            model_prompt_mapping[self.model_name], task_prompt_mapping[self.objective]
        ) as file:
            self._default_fit_prompt_template = file.read()

        # interpret_prompt_templates
        with pkg_resources.open_text(model_prompt_mapping[self.model_name], task_prompt_mapping["interpret"]) as file:
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

    def _extract_code_model(self, response: str) -> str:
        response = re.sub(r"^```python\n|\n```$", "", response, flags=re.MULTILINE)
        response = re.sub(r"^```\n|\n```$", "", response, flags=re.MULTILINE)

        keyword = "#########"

        start_index = response.find(keyword)
        end_index = response.rfind(keyword)
        if start_index != -1 and end_index != -1 and start_index < end_index:
            start_index += len(keyword)
            code_model = response[start_index:end_index].strip()
            return code_model
        else:
            code_model = response
            return code_model

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

        dataset_str = data_to_text(X, y)
        column_list = [str(element) for element in X.columns.tolist()] + ["label"]
        columns_name = ",".join(column_list)

        if prompt_args:
            prompt_args = {"dataset_str": dataset_str, **prompt_args}
        else:
            prompt_args = {"dataset_str": dataset_str, "columns_name": columns_name}

        prompt_ = make_prompt(prompt_template=prompt_template, **prompt_args)
        print(prompt_)
        code_model = self._run_prompt(prompt=prompt_, seed=seed, temperature=temperature)
        code_model = self._extract_code_model(code_model)
        self.code_model = code_model

        if try_code:
            try:
                self.predict(X.head(1))
                logger.info("Valid code_model successfully created!")
                return code_model
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
        return evaluate(y_true, y_pred, self.objective)

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
            prompt_args = {"code_model": self.code_model}

        prompt_ = make_prompt(prompt_template=prompt_template, **prompt_args)

        self.interpret_result = self._run_prompt(prompt=prompt_, temperature=temperature, seed=seed)
