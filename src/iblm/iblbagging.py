from __future__ import annotations

import logging
import pickle

from typing import TYPE_CHECKING

import numpy as np

from ibl import IBLModel

import metrics

from exceptions import UndefinedCodeModelError


if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s [%(name)s][%(levelname)s] (%(module)s:%(filename)s:%(funcName)s:%(lineno)d)")
logger.setLevel(logging.INFO)


class IBLBagging:

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
    def model_name(self) -> str:
        return self.ibl_model_config["model_name"]

    @property
    def objective(self) -> str:
        return self.ibl_model_config["objective"]

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        n_estimators: int,
        temperature: float = 0,
        seeds: list[int] | None = None,
        prompt_template: str | None = None,
        prompt_args: dict | None = None,
        try_code: bool = True,
    ) -> None:

        _models = []

        if self.ibl_model_config["api_type"] == "gemini":
            seeds = [None] * n_estimators
        elif seeds is None:
            seeds = np.random.choice(range(10_000), size=n_estimators, replace=False)
            logger.info(f"seeds wll be used: {seeds}")

        for i, seed in enumerate(seeds, start=1):
            iblm = IBLModel(**self.ibl_model_config)
            iblm.fit(
                X=X,
                y=y,
                temperature=temperature,
                seed=seed,
                prompt_template=prompt_template,
                prompt_args=prompt_args,
                try_code=try_code,
            )
            _models.append(iblm)
            logger.info(f"Fitting task finished: {i} / {len(seeds)}")

        self.models = _models

        logger.info("Fitting completed")

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
