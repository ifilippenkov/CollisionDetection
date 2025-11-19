from typing import Optional
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import aiohttp

class YandexCloudLLM:
    """
    Сервис для работы с LLM через Yandex Cloud API,
    используя интерфейс OpenAI.
    """
    
    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model_url: str = None,
        model_uri: str = "yandexgpt-lite", #llama-lite
        model_type: str = "gpt",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 30
    ):
        """
        Инициализация клиента Yandex Cloud LLM.
        
        Args:
            api_key: API ключ Yandex Cloud
            folder_id: Идентификатор каталога
            model_uri: URI модели (например, "llama-lite")
            temperature: Температура генерации (0-1)
            max_tokens: Максимальное количество токенов
            timeout: Таймаут запроса в секундах
        """
        self.api_key=api_key
        if model_url:
            self.model_url = model_url
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://llm.api.cloud.yandex.net/v1"
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://llm.api.cloud.yandex.net/v1"
        )

        self.model = f"{model_type}://{folder_id}/{model_uri}/latest"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_consistency(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Анализ согласованности текста с помощью LLM.
        
        Args:
            prompt: Промпт для модели
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
        
        Returns:
            Структурированный ответ модели
        
        Raises:
            ValueError: При ошибке парсинга ответа
            Exception: При других ошибках API
        """
        try:
            # Создаем запрос к API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                timeout=self.timeout
            )
            
            # Получаем ответ
            result = response.choices[0].message.content

            return result
            
        except Exception as e:
            print(f"Ошибка при запросе к LLM API: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def request_gpt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        messages = []
        if system_prompt:
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        ),
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout
        )

        result = response.choices[0].message.content

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def request_emb(
        self,
        text: str = None
    ) -> str:
        return (
        (self.client.embeddings.create(
                input=text,
                model=self.model,
                encoding_format="float",
            )
        )
        .data[0]
        .embedding
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def request_gpt_async(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            ),
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                timeout=self.timeout
            )

            result = response.choices[0].message.content

            return result

        except Exception as e:
            print(f"Ошибка при запросе к LLM API: {str(e)}")
            raise
