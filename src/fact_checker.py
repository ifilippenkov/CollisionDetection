from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from .llm_service import YandexCloudLLM
import tokens

@dataclass
class FactCheckResult:
    """Результат проверки фактов"""
    has_conflicts: bool  # Есть ли противоречия
    has_supporting_facts: bool  # Есть ли подтверждающие факты
    inconsistencies: List[Dict[str, str]]  # Список найденных противоречий
    supporting_facts: List[Dict[str, str]]  # Список подтверждающих фактов
    confidence: float  # Уверенность в результате (0-1)
    relevant_facts: List[str]  # Использованные для проверки факты
    explanation: str  # Объяснение результата

class FactConsistencyChecker:
    """
    Основной класс для проверки фактологической согласованности текстов.
    Координирует работу компонентов RAG-системы: разбиение текста, 
    векторный поиск и взаимодействие с LLM.
    """
    
    def __init__(
        self,
        llm_service: Any,  # Сервис для работы с LLM
        config: List[Union[float, int]] = [0.1, 1000],
        language: str = "en"  # Язык промпта: "en" или "ru"
    ):
        """
        Инициализация системы проверки фактов.
        
        Args:
            llm_service: Сервис для работы с LLM
            config: Конфигурация системы
            language: Язык системного промпта ("en" или "ru")
        """
        self.llm_service = llm_service
        self.config = config
        self.language = language
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def check_facts(self, text: str, relevant_facts: List[str]) -> FactCheckResult:
        """
        Проверка фактологической согласованности текста.
        
        Args:
            text: Текст для проверки
        
        Returns:
            Результат проверки фактов
        """
        self.logger.info("Начало проверки фактов")
        try:            
            # Формирование промпта для LLM
            prompt = self._create_prompt(text, relevant_facts)

            # Формирование системного промпта
            system_prompt = self._get_system_prompt()

            
            # Получение и обработка ответа от LLM
            llm_response = self.llm_service.analyze_consistency(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.config[0],
                max_tokens=self.config[1]
            )

            # Парсим ответ LLM в JSON с повторными попытками исправления
            max_attempts = 5
            
            for attempt in range(max_attempts):
                try:
                    # Очищаем ответ от markdown разметки
                    cleaned_response = self._clean_json_response(llm_response)
                    parsed_result = json.loads(cleaned_response)
                    self._validate_response(parsed_result)
                    break
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(llm_response)
                    if attempt < max_attempts - 1:
                        # Если это не последняя попытка, отправляем запрос на исправление
                        self.logger.warning(f"Попытка {attempt + 1}: Ошибка парсинга JSON: {str(e)}")
                        self.logger.info("Отправляем запрос на исправление JSON формата...")
                        
                        correction_prompt = f"""Ваш предыдущий ответ имел некорректный JSON формат. 
Пожалуйста, исправьте его и верните ТОЛЬКО корректный JSON без markdown разметки, без дополнительного текста, без тройных кавычек.

Ваш предыдущий ответ:
{llm_response}

Ошибка: {str(e)}

Верните исправленный JSON в правильном формате (только JSON, без ```):"""
                        
                        llm_response = self.llm_service.analyze_consistency(
                            prompt=correction_prompt,
                            system_prompt="Вы - помощник для исправления JSON формата. Верните ТОЛЬКО корректный JSON без markdown разметки, без тройных кавычек, без дополнительного текста.",
                            temperature=0.1,
                            max_tokens=1500
                        )
                    else:
                        # Последняя попытка не удалась
                        self.logger.error(f"Не удалось получить корректный JSON после {max_attempts} попыток")
                        raise ValueError(f"Ошибка парсинга JSON ответа после {max_attempts} попыток: {str(e)}")

            
            # Парсим JSON в структуру FactCheckResult
            result = self._parse_llm_response(parsed_result, relevant_facts)
            
            self.logger.info(
                f"Проверка завершена. "
                f"Найдено противоречий: {len(result.inconsistencies)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при проверке фактов: {str(e)}")
            raise

    def _clean_json_response(self, response: str) -> str:
        """
        Очистка ответа от markdown разметки и других артефактов.
        
        Args:
            response: Сырой ответ от модели
            
        Returns:
            Очищенный JSON текст
        """
        # Убираем markdown блоки кода
        response = response.strip()
        
        # Удаляем тройные кавычки в начале и конце
        if response.startswith("```json"):
            response = response[7:]  # Убираем ```json
        elif response.startswith("```"):
            response = response[3:]  # Убираем ```
            
        if response.endswith("```"):
            response = response[:-3]  # Убираем ``` в конце
        
        # Убираем лишние пробелы
        response = response.strip()
        
        return response
    
    def _validate_response(self, response: Dict[str, Any]) -> None:
        """
        Проверка структуры ответа от модели.
        
        Args:
            response: Ответ от модели
        
        Raises:
            ValueError: Если структура ответа некорректна
        """
        required_fields = ['has_conflicts', 'inconsistencies', 'supporting_facts', 'confidence', 'explanation']
        
        # Проверяем наличие всех необходимых полей
        if not all(field in response for field in required_fields):
            raise ValueError("В ответе отсутствуют обязательные поля")
        
        # Проверяем типы данных
        if not isinstance(response['has_conflicts'], bool):
            raise ValueError("Поле 'has_conflicts' должно быть boolean")
        
        if not isinstance(response['has_supporting_facts'], bool):
            raise ValueError("Поле 'has_supporting_facts' должно быть boolean")
        
        if not isinstance(response['inconsistencies'], list):
            raise ValueError("Поле 'inconsistencies' должно быть списком")
            
        if not isinstance(response['supporting_facts'], list):
            raise ValueError("Поле 'supporting_facts' должно быть списком")
        
        if not isinstance(response['confidence'], (int, float)):
            raise ValueError("Поле 'confidence' должно быть числом")
        
        if not isinstance(response['explanation'], str):
            raise ValueError("Поле 'explanation' должно быть строкой")
        
        # Проверяем структуру inconsistencies
        for inc in response['inconsistencies']:
            if not all(k in inc for k in ['statement', 'fact', 'explanation']):
                raise ValueError("Некорректная структура элемента inconsistencies")
                
        # Проверяем структуру supporting_facts
        for sup in response['supporting_facts']:
            if not all(k in sup for k in ['statement', 'fact', 'explanation']):
                raise ValueError("Некорректная структура элемента supporting_facts")
        # Удаляем элементы с пустыми фактами из списка противоречий
        response['inconsistencies'] = [inc for inc in response['inconsistencies'] if inc.get('fact')]
        
        # Если после фильтрации список пустой, устанавливаем has_conflicts в False
        if not response['inconsistencies']:
            response['has_conflicts'] = False
            
        # Удаляем элементы с пустыми фактами из списка подтверждающих фактов
        response['supporting_facts'] = [sup for sup in response['supporting_facts'] if sup.get('fact')]
        
        # Если после фильтрации список пустой, устанавливаем has_supporting_facts в False
        if not response['supporting_facts']:
            response['has_supporting_facts'] = False

    def _create_prompt(self, text: str, facts: List[str]) -> str:
        """
        Формирование промпта для LLM.
        
        Args:
            text: Проверяемый текст
            facts: Релевантные факты из базы знаний
        
        Returns:
            Промпт для LLM
        """
        formatted_facts = "\n".join(f"- {fact}" for fact in facts)
        prompt_template = f"""Input text to check: {text}
        Known facts: {formatted_facts}"""
        
        return prompt_template

    def _get_system_prompt(self) -> str:
        """
        Получение системного промпта для модели.
        
        Returns:
            Системный промпт
        """
        import os
        
        # Определяем путь к файлу промпта в зависимости от языка
        prompt_filename = f"system_prompt_{'eng' if self.language == 'en' else 'ru'}.txt"
        
        # Ищем файл в папке prompts относительно корня проекта
        project_root = Path(__file__).resolve().parent.parent
        prompt_path = project_root / "prompts" / prompt_filename
        
        if not prompt_path.exists():
            # Fallback на старое расположение
            prompt_path = project_root / prompt_filename
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        relevant_facts: List[str]
    ) -> FactCheckResult:
        """
        Преобразование ответа LLM в структурированный результат.
        
        Args:
            response: Ответ от LLM
            relevant_facts: Использованные факты
        
        Returns:
            Структурированный результат проверки
        """
        return FactCheckResult(
            has_conflicts=response["has_conflicts"],
            has_supporting_facts=response["has_supporting_facts"],
            inconsistencies=response["inconsistencies"],
            supporting_facts=response["supporting_facts"],
            confidence=response["confidence"],
            relevant_facts=relevant_facts,
            explanation=response["explanation"]
        )
        
llm_service = YandexCloudLLM(
    api_key=tokens.AUTH_TOKEN,
    folder_id=tokens.FOLDER_ID,
    model_uri="yandexgpt-lite", #llama-lite
    temperature=0.1
)

# Создаем экземпляр checker с языком по умолчанию (английский)
# Можно переопределить через checker = FactConsistencyChecker(llm_service=llm_service, language="ru")
checker = FactConsistencyChecker(llm_service=llm_service, language="en")
