"""LLM answer generation, abstracted behind a small interface so the provider
can be swapped later without touching the rest of the app."""
import logging
from abc import ABC, abstractmethod

from groq import Groq

logger = logging.getLogger("llm")

SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы строго на основе предоставленных "
    "фрагментов документов (контекста). Отвечай на русском языке, точно и по "
    "существу. Если в контексте нет информации, достаточной для ответа, "
    "честно скажи, что не нашёл ответа в документах — не придумывай факты. "
    "Не ссылайся на 'фрагмент 1/2/3' — источники будут показаны пользователю отдельно."
)


class LLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> None:
        """Raises an exception if the provider is not reachable/configured."""
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_answer(self, question: str, context: str) -> str:
        user_prompt = f"Контекст из документов:\n\n{context}\n\nВопрос: {question}"
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return completion.choices[0].message.content or ""

    def health_check(self) -> None:
        # Cheap way to confirm the API key/model are valid without burning a
        # full completion: list available models.
        self.client.models.list()
