"""
FastAPI сервер для системы проверки фактологических противоречий
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import tempfile
from pathlib import Path
import logging
import nest_asyncio

# Разрешаем вложенные event loops для совместимости с LlamaIndex
nest_asyncio.apply()

# Импорты из существующей системы
from src.graph_rag import custom_embedder, custom_llm
from src.chunk_getter import Data
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from src.fact_checker import checker
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.retrievers import VectorContextRetriever, LLMSynonymRetriever
from llama_index.core import QueryBundle
import tokens

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="Collision Detection API",
    description="API для поиска фактологических противоречий в текстах",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное состояние
class AppState:
    def __init__(self):
        self.graph_index = None
        self.graph_store = None
        self.temp_dir = None
        
app_state = AppState()

# Pydantic модели для запросов/ответов
class CheckContradictionsRequest(BaseModel):
    text: str
    vector_top_k: int = 30
    reranker_top_n: int = 5
    with_reranker: bool = True
    language: str = "en"  # Язык системного промпта: "en" или "ru"

class StatusResponse(BaseModel):
    database_connected: bool
    index_exists: bool
    nodes_count: Optional[int] = None

class BuildIndexResponse(BaseModel):
    status: str
    message: str
    nodes_count: int

class ContradictionCheckResponse(BaseModel):
    has_conflicts: bool
    has_supporting_facts: bool
    inconsistencies: List[dict]
    supporting_facts: List[dict]
    confidence: float
    explanation: str
    relevant_facts_count: int


# Вспомогательные функции
def get_graph_store():
    """Получение подключения к Neo4j"""
    try:
        graph_store = Neo4jPGStore(
            url=tokens.NEO4J_URL,
            username=tokens.NEO4J_USERNAME,
            password=tokens.NEO4J_PASSWORD
        )
        return graph_store
    except Exception as e:
        logger.error(f"Ошибка подключения к Neo4j: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Не удалось подключиться к Neo4j: {str(e)}")


def get_retrieved_nodes(index, query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False):
    """Получение релевантных узлов из графа (копия из main.py)"""
    query_bundle = QueryBundle(query_str)

    syn = LLMSynonymRetriever(
        graph_store=index.property_graph_store,
        llm=custom_llm,
        include_text=True,
        max_keywords=8,
        path_depth=5
    )

    vec = VectorContextRetriever(
        graph_store=index.property_graph_store,
        vector_store=index.vector_store,
        embed_model=custom_embedder,
        include_text=True,
        similarity_top_k=vector_top_k,
        path_depth=5
    )

    retriever = index.as_retriever(sub_retrievers=[syn, vec], include_text=True)

    retrieved_nodes = retriever.retrieve(query_bundle)
    
    if with_reranker:
        prompt_str = (
            "A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided. \n"
            "Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score. The relevance score is a number from 1-10 based on how relevant you think the document is to the question.\n"
            "Prioritize documents based on their relevance to the question, regardless of whether they support or contradict the query. Both confirming and contradicting facts are considered equally relevant if they provide significant information, context, or arguments related to the question.\n"
            "Assign relevance scores in a balanced way to fairly represent differing viewpoints or data, ensuring that conflicting evidence is not overshadowed by other documents.\n"
            "Always include at least one document in the response, selecting the most relevant documents even if the relevance is low.\n"
            "Do not include documents that are irrelevant to the question.\n"
            "Example format: \n"
            "Document 1:\n<summary of document 1>\n\n"
            "Document 2:\n<summary of document 2>\n\n"
            "...\n\n"
            "Document 10:\n<summary of document 10>\n\n"
            "Question: <question>\n"
            "Answer:\n"
            "Doc: 9, Relevance: 7\n"
            "Doc: 3, Relevance: 4\n"
            "Doc: 7, Relevance: 3\n\n"
            "Let's try this now: \n\n"
            "{context_str}\n"
            "Question: {query_str}\n"
            "Answer:\n"
        )

        custom_choice_template = PromptTemplate(template=prompt_str)

        reranker = LLMRerank(
            llm=custom_llm,
            choice_batch_size=5,
            top_n=reranker_top_n,
            choice_select_prompt=custom_choice_template
        )

        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
    
    return retrieved_nodes


# API Endpoints
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Collision Detection API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "build_index": "/api/build_index",
            "check_contradictions": "/api/check_contradictions",
            "clear_index": "/api/clear_index"
        }
    }


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Проверка статуса системы"""
    try:
        # Проверяем подключение к Neo4j
        graph_store = get_graph_store()
        database_connected = True
        
        # Проверяем наличие индекса
        index_exists = app_state.graph_index is not None
        
        nodes_count = None
        if index_exists:
            try:
                # Попытка получить количество узлов
                nodes_count = len(app_state.graph_index.property_graph_store.get_triplets())
            except:
                pass
        
        return StatusResponse(
            database_connected=database_connected,
            index_exists=index_exists,
            nodes_count=nodes_count
        )
    except Exception as e:
        logger.error(f"Ошибка при проверке статуса: {str(e)}")
        return StatusResponse(
            database_connected=False,
            index_exists=False,
            nodes_count=None
        )


@app.post("/api/build_index", response_model=BuildIndexResponse)
async def build_index(
    files: List[UploadFile] = File(...),
    chunker: str = Form("basic")
):
    """
    Построение индекса из загруженных файлов
    
    Args:
        files: Список файлов для обработки
        chunker: Тип чанкера ("basic" или "LLM")
    """
    logger.info(f"Получен запрос на построение индекса. Файлов: {len(files)}, chunker: {chunker}")
    
    try:
        # Создаем временную директорию для файлов
        if app_state.temp_dir:
            shutil.rmtree(app_state.temp_dir, ignore_errors=True)
        
        app_state.temp_dir = tempfile.mkdtemp()
        logger.info(f"Создана временная директория: {app_state.temp_dir}")
        
        # Сохраняем загруженные файлы
        for file in files:
            file_path = os.path.join(app_state.temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info(f"Сохранен файл: {file.filename}")
        
        # Получаем graph store
        if not app_state.graph_store:
            app_state.graph_store = get_graph_store()
        
        # Создаем Data объект и получаем chunks
        data_processor = Data(
            input_promt="",  # Не используется при построении индекса
            data_path=app_state.temp_dir,
            format="text",
            chunker=chunker
        )
        
        logger.info("Начало обработки документов и создания chunks...")
        nodes = data_processor.node_getter()
        logger.info(f"Создано {len(nodes)} узлов")
        
        # Строим PropertyGraphIndex
        logger.info("Начало построения PropertyGraphIndex...")
        app_state.graph_index = PropertyGraphIndex(
            nodes=nodes,
            llm=custom_llm,
            property_graph_store=app_state.graph_store,
            embed_model=custom_embedder,
            include_embeddings=True,
        )
        logger.info("PropertyGraphIndex успешно построен")
        
        return BuildIndexResponse(
            status="success",
            message=f"Индекс успешно построен из {len(files)} файлов",
            nodes_count=len(nodes)
        )
        
    except Exception as e:
        logger.error(f"Ошибка при построении индекса: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при построении индекса: {str(e)}")


@app.post("/api/check_contradictions", response_model=ContradictionCheckResponse)
async def check_contradictions(request: CheckContradictionsRequest):
    """
    Проверка текста на противоречия с базой знаний
    
    Args:
        request: Запрос с текстом для проверки и параметрами поиска
    """
    logger.info(f"Получен запрос на проверку противоречий для текста: {request.text[:100]}...")
    
    # Проверяем наличие индекса
    if not app_state.graph_index:
        raise HTTPException(
            status_code=400,
            detail="Индекс не построен. Сначала загрузите файлы через /api/build_index"
        )
    
    try:
        # Получаем релевантные узлы
        logger.info("Поиск релевантных узлов...")
        retrieved_nodes = get_retrieved_nodes(
            app_state.graph_index,
            request.text,
            vector_top_k=request.vector_top_k,
            reranker_top_n=request.reranker_top_n,
            with_reranker=request.with_reranker
        )
        logger.info(f"Найдено {len(retrieved_nodes)} релевантных узлов")
        
        # Извлекаем факты из узлов
        facts = []
        for node in retrieved_nodes:
            text = str(node.node.get_text())
            facts.append(text)
        
        # Проверяем противоречия с выбранным языком
        logger.info(f"Проверка фактов через LLM (язык: {request.language})...")
        
        # Создаем checker с выбранным языком
        from src.llm_service import YandexCloudLLM
        from src.fact_checker import FactConsistencyChecker
        llm_service = YandexCloudLLM(
            api_key=tokens.AUTH_TOKEN,
            folder_id=tokens.FOLDER_ID,
            model_uri="yandexgpt-lite",
            temperature=0.1
        )
        language_checker = FactConsistencyChecker(llm_service=llm_service, language=request.language)
        
        result = language_checker.check_facts(request.text, facts)
        logger.info("Проверка завершена")
        
        return ContradictionCheckResponse(
            has_conflicts=result.has_conflicts,
            has_supporting_facts=result.has_supporting_facts,
            inconsistencies=result.inconsistencies,
            supporting_facts=result.supporting_facts,
            confidence=result.confidence,
            explanation=result.explanation,
            relevant_facts_count=len(facts)
        )
        
    except Exception as e:
        logger.error(f"Ошибка при проверке противоречий: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при проверке противоречий: {str(e)}")


@app.delete("/api/clear_index")
async def clear_index():
    """Очистка индекса и временных файлов"""
    try:
        # Очищаем индекс
        app_state.graph_index = None
        
        # Удаляем временные файлы
        if app_state.temp_dir:
            shutil.rmtree(app_state.temp_dir, ignore_errors=True)
            app_state.temp_dir = None
        
        logger.info("Индекс очищен")
        
        return {"status": "success", "message": "Индекс успешно очищен"}
    except Exception as e:
        logger.error(f"Ошибка при очистке индекса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при очистке индекса: {str(e)}")


# Обработчик исключений
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

