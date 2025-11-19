#  АнтиКоллизия: поиск противоречий в тексте с помощью LLM.

RAG-система для автоматического поиска внутренних противоречий и подтверждений в текстах с использованием векторного поиска и LLM.

---

## 1. Предварительные требования

### 1.1 Yandex Cloud API Token

1. Перейдите в [Yandex Cloud Console](https://console.cloud.yandex.ru/)
2. Создайте сервисный аккаунт
3. Создайте API ключ
4. Скопируйте:
   - `FOLDER_ID` - идентификатор каталога
   - `AUTH_TOKEN` - API ключ

### 1.2 Neo4j Database
1.
```bash
pip install llama-index llama-index-graph-stores-neo4j
```
2.
```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    neo4j:latest
```
3. Переходим на http://localhost:7474/browser/, регистрируемся и запоминаем логин с паролем

### 1.3 Системные требования

- Python 3.9+
- Docker (для Neo4j)

---

## 2. Установка

### 2.1 Клонирование репозитория

```bash
git clone https://github.com/ifilippenkov/CollisionDetection.git
cd CollisionDetection
```

### 2.2 Создание виртуального окружения

```bash
python3 -m venv .env
source .env/bin/activate  
```

### 2.3 Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2.4 Настройка credentials

Создайте файл `tokens.py`:

```python
# Yandex Cloud
FOLDER_ID = "ваш_folder_id"
AUTH_TOKEN = "ваш_api_token"

# Neo4j
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "ваш логин"
NEO4J_PASSWORD = "ваш_пароль"
```

---

##  3. Использование

### 3.1 Через консоль (main.py)

#### Шаг 1: Подготовьте данные

Поместите текстовые файлы в папку `input_data/`

#### Шаг 2: Создайте конфигурацию

Файл `input.json`:
```json
{
  "conflict": "Текст для проверки",
  "data_path": "input_data",
  "chunker": "basic"
}
```

#### Шаг 3: Запустите проверку

```bash
# Первый запуск (построение графа)
python main.py --input_json "Путь до input.json" --language en

# Последующие запуски (использование существующего графа)
python main.py --input_json "Путь до input.json" --has_graph True --language ru
```

**Параметры:**
- `--input_json` - путь к JSON конфигурации
- `--has_graph` - использовать существующий граф (True/False)
- `--language` - язык промпта: `en` (английский) или `ru` (русский)

### 3.2 Через Web-интерфейс

#### Шаг 1: Запустите API сервер

```bash
python api.py
```

API будет доступен на `http://localhost:8000`

#### Шаг 2: Запустите Frontend

В новом терминале:

```bash
cd frontend
python -m http.server 8080
```

#### Шаг 3: Откройте в браузере

Перейдите на `http://localhost:8080`

**Использование интерфейса:**

1. **Вкладка "Загрузка базы знаний":**
   - Перетащите файлы или выберите через кнопку
   - Выберите метод разбиения (basic/LLM)
   - Нажмите "Построить базу знаний"

2. **Вкладка "Проверка текста":**
   - Введите текст для проверки
   - Выберите язык системного промпта
   - Настройте параметры поиска (опционально)
   - Нажмите "Найти противоречия"

3. **Результаты:**
   -  **Противоречия** - факты, которые противоречат тексту
   -  **Подтверждения** - факты, которые подтверждают текст
   - Показывается уверенность модели и детальное объяснение

---

## Структура проекта

```
CollisionDetection/
├── src/                      # Backend модули
│   ├── chunk_getter.py       # Разбиение документов
│   ├── fact_checker.py       # Проверка фактов
│   ├── graph_rag.py          # RAG система
│   └── llm_service.py        # Yandex Cloud LLM
├── frontend/                 # Web-интерфейс
│   ├── index.html
│   ├── app.js
│   └── style.css
├── prompts/                  # Системные промпты
│   ├── system_prompt_eng.txt
│   └── system_prompt_ru.txt
├── api.py                    # FastAPI сервер
├── main.py                   # CLI интерфейс
├── tokens.py                 # Credentials
└── requirements.txt          # Зависимости
```