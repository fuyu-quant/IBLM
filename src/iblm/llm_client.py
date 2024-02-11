from __future__ import annotations

import os

import google.generativeai as genai

from openai import AzureOpenAI, OpenAI

from .exceptions import InvalidAPIOption, InvalidAPIType


API_TYPES = ("openai", "azure", "gemini")


class Gemini:
    genai = genai


def get_client(
    # common
    api_type: str,
    # openai & azure
    api_key: str | None = None,
    max_retries: int = 5,
    timeout: int = 120,
    organization: str | None = None,
    # azure
    api_version: str | None = None,
    azure_endpoint: str | None = None,
):
    if api_type == "openai":
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", api_key),
            max_retries=max_retries,
            timeout=timeout,
            organization=organization,
        )

    elif api_type == "azure":
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY", api_key),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", azure_endpoint),
            api_version=os.getenv("OPENAI_API_VERSION", api_version),
            max_retries=max_retries,
            timeout=timeout,
            organization=organization,
        )
    elif api_type == "gemini":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", api_key))
        return Gemini()
    else:
        raise InvalidAPIType(f"specify the api_type from {API_TYPES}")


def run_prompt(client, model_name: str, prompt: str, temperature: float = 0, seed: int | None = None) -> str:
    if isinstance(client, (OpenAI, AzureOpenAI)):
        # ref: https://platform.openai.com/docs/api-reference/completions/create
        response = client.chat.completions.create(
            model=model_name,  # if azure, specify `deployment_name`
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            # top_p=1,
            # n=1,
            # stop=None
            # max_tokens=16,
        )
        return response.choices[0].message.content
    elif isinstance(client, Gemini):
        # ref: https://ai.google.dev/tutorials/python_quickstart
        if seed:
            raise InvalidAPIOption("Gemini does not support `seed` option")
        model = client.genai.GenerativeModel(
            model_name,
            generation_config=dict(
                temperature=temperature,
                # top_p=None,
                # candidate_count=None,
                # stop_sequences=None,
                # max_output_tokens=None,
                # top_k=None,
            ),
        )
        response = model.generate_content(prompt)
        return response.text
