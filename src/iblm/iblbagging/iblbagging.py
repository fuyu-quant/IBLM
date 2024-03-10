from __future__ import annotations

import sys

from typing import TYPE_CHECKING

import numpy as np

from iblm.ibl import IBLModel


sys.path.append("..")

if TYPE_CHECKING:
    import pandas as pd


class IBLBaggingModel(IBLModel):
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
        num_model: int = 10,
        max_sample: int = 600,
        min_sample: int = 400,
    ) -> None:
        super().__init__(
            model_name,
            objective,
            # common
            api_type,
            # openai & azure
            api_key,
            max_retries,
            timeout,
            organization,
            # azure
            api_version,
            azure_endpoint,
        )

        # bagging specific
        self.num_model = num_model
        self.max_sample = max_sample
        self.min_sample = min_sample
        self.bagging_code_model = None
        self.code_models = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.array,
        temperature: float = 0,
        seed: int | None = None,
        prompt_template: str | None = None,
        prompt_args: dict | None = None,
        try_code: bool = True,
    ):
        # sampling data
        np.random.seed(0)
        X.reset_index(drop=True, inplace=True)

        index_patterns = []
        for _ in range(self.num_model):
            random_sample_size = np.random.randint(self.min_sample, self.max_sample)
            sampled_indices = np.random.choice(X.index, random_sample_size, replace=False)
            index_patterns.append(sampled_indices)

        for i, indices in enumerate(index_patterns, start=1):
            X_sampled = X.loc[indices]
            y_sampled = y[indices]
            key = f"model_{i}"
            try:
                bagging_model = super().fit(
                    X_sampled, y_sampled, temperature, seed, prompt_template, prompt_args, try_code
                )
                self.code_model = bagging_model
                y_pred = super().predict(X_sampled)
                metric_dict = super().evaluate(y_sampled, y_pred)
                self.code_models[key] = {"code_model": bagging_model, "metric_dict": metric_dict}
            except Exception:
                continue

        self.code_models = sorted(self.code_models.items(), key=lambda x: x[1]["metric_dict"]["roc_auc"], reverse=True)
        return self.code_models

    def predict_(self, X: pd.DataFrame, top_model: int | None = None) -> np.array:
        y_preds = []
        if top_model is None:
            for _, model_info in self.code_models:
                self.code_model = model_info["code_model"]
                y_pred = super().predict(X)
                y_preds.append(y_pred)
        else:
            for _, model_info in self.code_models[:top_model]:
                self.code_model = model_info["code_model"]
                y_pred = super().predict(X)
                y_preds.append(y_pred)
        return np.mean(y_preds, axis=0)
