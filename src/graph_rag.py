from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from yandex_cloud_ml_sdk import YCloudML
from .llm_service import YandexCloudLLM
import tokens
from llama_index.core.llms.callbacks import (
    llm_completion_callback,
)

EMB_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
GPT_URL="https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

emb_model = YandexCloudLLM(tokens.AUTH_TOKEN, tokens.FOLDER_ID, EMB_URL, "text-search-doc", "emb")
rag_model = YandexCloudLLM(tokens.AUTH_TOKEN, tokens.FOLDER_ID, GPT_URL, "yandexgpt")

# Глобальный кэш для эмбеддингов в памяти
_embeddings_cache = {}

class CustomEmbeddingModel(BaseEmbedding):
    def __init__(self):
        super().__init__(model_name="custom_embedder")

    def _get_query_embedding(self, query: str) -> list[float]:
        # Проверяем кэш
        if query in _embeddings_cache:
            return _embeddings_cache[query]
        
        # Если нет в кэше, запрашиваем новый
        embedding = emb_model.request_emb(query)
        _embeddings_cache[query] = embedding
        return embedding

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        # Проверяем кэш
        if text in _embeddings_cache:
            return _embeddings_cache[text]
        
        # Если нет в кэше, запрашиваем новый
        embedding = emb_model.request_emb(text)
        _embeddings_cache[text] = embedding
        return embedding

class CustomLLMAPI(CustomLLM):
    def __init__(self):
        super().__init__()

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        ans = rag_model.request_gpt(prompt)
        return CompletionResponse(text=ans)
    
    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        res = await rag_model.request_gpt_async(prompt)
        return CompletionResponse(text=res)

    @property
    def metadata(self):
        return LLMMetadata()

    def stream_complete(self, prompt, formatted, **kwargs):
        ans = rag_model.request_gpt(prompt)
        yield CompletionResponse(text=ans)

custom_embedder = CustomEmbeddingModel()
custom_llm = CustomLLMAPI()
