"""
This file contains the managers for the OpenAI and Ollama APIs.
"""
import os
from typing import Optional
import requests
from openai import OpenAI
from IPython.display import Markdown, display


class OpenAIManager:
    """Manager class for OpenAI API interactions"""

    def __init__(
        self, model: str = "gpt-4o-mini", api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)

    def display_response(self, question: str):
        """
        Ask a question and display the response as Markdown in Jupyter.
        Does not return the answer string to avoid raw output in Jupyter.
        """
        system_prompt = (
            "You are a helpful technical assistant. Provide clear, "
            "detailed explanations for technical questions."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                stream=False,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
        display(Markdown(answer))
        return None


class OllamaManager:
    """Manager class for Ollama API interactions (non-streaming only)"""

    SYSTEM_PROMPT = (
        "You are a helpful technical assistant. Provide clear, "
        "detailed explanations for technical questions."
    )

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api"

    def _check_ollama_connection(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def display_response(self, question: str):
        """
        Ask a question and display the response as Markdown in Jupyter.
        Does not return the answer string to avoid raw output in Jupyter.
        """
        if not self._check_ollama_connection():
            response = (
                "Error: Cannot connect to Ollama. Make sure Ollama is running "
                "on localhost:11434"
            )
        else:
            try:
                prompt = f"{self.SYSTEM_PROMPT}\n\nQuestion: {question}"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                }
                api_response = requests.post(
                    f"{self.api_url}/generate", json=payload, timeout=30
                )
                api_response.raise_for_status()
                result = api_response.json()
                response = result.get("response", "No response received")
            except Exception as e:
                response = f"Error: {str(e)}"
        display(Markdown(response))
        return None
