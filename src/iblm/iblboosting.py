from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ibl import IBLModel

import metrics

from exceptions import UndefinedCodeModelError


class IBLBagging:
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

        self.ibl_model_config = dict(
            model_name=model_name,
            objective=objective,
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

        self.models = []

    @property
    def model_name(self):
        return self.ibl_model_config["model_name"]

    @property
    def objective(self):
        return self.ibl_model_config["objective"]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        n_estimators: int,
        temperature: float = 0,
        seeds: list[int] | None = None,
        prompt_template: str = None,
    ) -> None:

        self.models = []  # clear models

        if self.ibl_model_config["api_type"] == "gemini":
            seeds = [None] * n_estimators
        elif seeds is None:
            seeds = np.random.choice(range(10_000), size=n_estimators, replace=False)

        for seed in seeds:
            iblm = IBLModel(**self.ibl_model_config)
            iblm.fit(X, y, temperature, seed, prompt_template)

            self.models.append(iblm)

    def predict(self, X: pd.DataFrame) -> None:
        if self.models == []:
            raise UndefinedCodeModelError("You must load or train the model before predict!")

        y_preds = [iblm.predict(X) for iblm in self.models]

        return np.mean(y_preds, axis=0)

    def load_model(self, file_path: str) -> None:
        with open(file_path, "rb") as file:
            self.models = pickle.load(file)

    def save_model(self, file_path: str) -> None:
        if self.models == []:
            raise UndefinedCodeModelError("You must train the model before saving!")

        with open(file_path, "wb") as file:
            pickle.dump(self.models, file)

    def evaluate(self, y_true: np.array, y_pred: np.array) -> dict:
        return metrics.evaluate(y_true, y_pred, self.objective)
