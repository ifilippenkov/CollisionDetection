// Конфигурация API
const API_BASE_URL = 'http://localhost:8000';

// Глобальное состояние
const state = {
    files: [],
    indexBuilt: false,
    isLoading: false
};

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Проверка статуса системы
    await checkSystemStatus();

    // Инициализация обработчиков событий
    initializeEventListeners();
}

// Проверка статуса системы
async function checkSystemStatus() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = statusIndicator.querySelector('.status-text');

    try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        const data = await response.json();

        if (data.database_connected) {
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Система готова к работе';

            if (data.index_exists) {
                state.indexBuilt = true;
                enableCheckTab();
            }
        } else {
            statusIndicator.classList.add('disconnected');
            statusText.textContent = 'Ошибка подключения к базе данных';
        }
    } catch (error) {
        statusIndicator.classList.add('disconnected');
        statusText.textContent = 'Не удалось подключиться к серверу';
        console.error('Ошибка проверки статуса:', error);
    }
}

// Инициализация обработчиков событий
function initializeEventListeners() {
    // Вкладки
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });

    // Upload Area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFileSelect({ target: { files: e.dataTransfer.files } });
    });

    // Build Index Button
    document.getElementById('buildIndexBtn').addEventListener('click', buildIndex);

    // Check Button
    document.getElementById('checkBtn').addEventListener('click', checkContradictions);
}

// Переключение вкладок
function switchTab(tabName) {
    // Деактивация всех вкладок
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    // Активация выбранной вкладки
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}Pane`).classList.add('active');
}

// Обработка выбора файлов
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        // Проверка, не добавлен ли файл уже
        if (!state.files.find(f => f.name === file.name)) {
            state.files.push(file);
        }
    });

    updateFileList();
    updateBuildButton();
}

// Обновление списка файлов
function updateFileList() {
    const fileList = document.getElementById('fileList');
    
    if (state.files.length === 0) {
        fileList.innerHTML = '';
        return;
    }

    fileList.innerHTML = state.files.map((file, index) => `
        <div class="file-item">
            <div class="file-info">
                <div class="file-icon">📄</div>
                <div class="file-details">
                    <h4>${file.name}</h4>
                    <p>${formatFileSize(file.size)}</p>
                </div>
            </div>
            <button class="file-remove" onclick="removeFile(${index})">✕</button>
        </div>
    `).join('');
}

// Удаление файла из списка
function removeFile(index) {
    state.files.splice(index, 1);
    updateFileList();
    updateBuildButton();
}

// Обновление кнопки построения индекса
function updateBuildButton() {
    const buildBtn = document.getElementById('buildIndexBtn');
    buildBtn.disabled = state.files.length === 0 || state.isLoading;
}

// Форматирование размера файла
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Построение индекса
async function buildIndex() {
    if (state.files.length === 0 || state.isLoading) return;

    state.isLoading = true;
    const buildBtn = document.getElementById('buildIndexBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressText = document.getElementById('progressText');
    const progressFill = document.getElementById('progressFill');

    buildBtn.disabled = true;
    progressContainer.style.display = 'block';
    progressText.textContent = 'Загрузка файлов...';
    progressFill.style.width = '33%';

    try {
        // Подготовка FormData
        const formData = new FormData();
        state.files.forEach(file => {
            formData.append('files', file);
        });

        const chunker = document.getElementById('chunkerSelect').value;
        formData.append('chunker', chunker);

        progressText.textContent = 'Обработка документов и построение индекса...';
        progressFill.style.width = '66%';

        // Отправка запроса
        const response = await fetch(`${API_BASE_URL}/api/build_index`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при построении индекса');
        }

        const result = await response.json();

        progressText.textContent = 'Индекс успешно построен!';
        progressFill.style.width = '100%';

        // Показываем успешное сообщение
        setTimeout(() => {
            alert(`✅ База знаний успешно построена!\n\nОбработано узлов: ${result.nodes_count}\n\nТеперь вы можете переключиться на вкладку "Проверка текста".`);
            
            state.indexBuilt = true;
            enableCheckTab();

            // Сброс прогресса
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
        }, 1000);

    } catch (error) {
        console.error('Ошибка при построении индекса:', error);
        alert(`❌ Ошибка при построении индекса:\n${error.message}`);
        progressContainer.style.display = 'none';
        progressFill.style.width = '0%';
    } finally {
        state.isLoading = false;
        buildBtn.disabled = false;
    }
}

// Активация вкладки проверки
function enableCheckTab() {
    const checkTab = document.getElementById('checkTab');
    checkTab.disabled = false;
}

// Проверка противоречий
async function checkContradictions() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();

    if (!text) {
        alert('⚠️ Пожалуйста, введите текст для проверки');
        return;
    }

    if (!state.indexBuilt) {
        alert('⚠️ Сначала постройте базу знаний на вкладке "Загрузка базы знаний"');
        return;
    }

    state.isLoading = true;
    const checkBtn = document.getElementById('checkBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');

    checkBtn.disabled = true;
    loadingIndicator.style.display = 'block';
    resultsContainer.style.display = 'none';

    try {
        const requestData = {
            text: text,
            vector_top_k: parseInt(document.getElementById('topKInput').value),
            reranker_top_n: parseInt(document.getElementById('rerankerInput').value),
            with_reranker: document.getElementById('rerankerCheckbox').checked,
            language: document.getElementById('languageSelect').value
        };

        const response = await fetch(`${API_BASE_URL}/api/check_contradictions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при проверке противоречий');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Ошибка при проверке противоречий:', error);
        alert(`❌ Ошибка при проверке противоречий:\n${error.message}`);
    } finally {
        state.isLoading = false;
        checkBtn.disabled = false;
        loadingIndicator.style.display = 'none';
    }
}

// Отображение результатов
function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const resultsSummary = document.getElementById('resultsSummary');
    const conflictCount = document.getElementById('conflictCount');
    const supportCount = document.getElementById('supportCount');
    const conflictsList = document.getElementById('conflictsList');
    const supportsList = document.getElementById('supportsList');
    const explanationContent = document.getElementById('explanationContent');

    // Показываем контейнер результатов
    resultsContainer.style.display = 'block';

    // Уверенность
    const confidence = result.confidence;
    let confidenceClass = 'low';
    let confidenceText = 'Низкая уверенность';

    if (confidence >= 0.9) {
        confidenceClass = 'high';
        confidenceText = 'Высокая уверенность';
    } else if (confidence >= 0.7) {
        confidenceClass = 'medium';
        confidenceText = 'Средняя уверенность';
    }

    confidenceBadge.className = `confidence-badge ${confidenceClass}`;
    confidenceBadge.textContent = `${confidenceText} (${(confidence * 100).toFixed(0)}%)`;

    // Сводка
    const hasConflicts = result.has_conflicts;
    const hasSupport = result.has_supporting_facts;

    let summaryIcon = '✅';
    let summaryText = 'Противоречий не обнаружено';
    let summaryClass = 'success';

    if (hasConflicts && hasSupport) {
        summaryIcon = '⚠️';
        summaryText = 'Обнаружены как противоречия, так и подтверждения';
        summaryClass = 'warning';
    } else if (hasConflicts) {
        summaryIcon = '❌';
        summaryText = 'Обнаружены противоречия с базой знаний';
        summaryClass = 'danger';
    } else if (hasSupport) {
        summaryIcon = '✅';
        summaryText = 'Текст подтверждается фактами из базы знаний';
        summaryClass = 'success';
    }

    resultsSummary.innerHTML = `
        <h3 style="font-size: 1.25rem; margin-bottom: 0.5rem;">
            ${summaryIcon} ${summaryText}
        </h3>
        <p style="color: var(--text-secondary);">
            Проанализировано фактов: ${result.relevant_facts_count}
        </p>
    `;

    // Противоречия
    conflictCount.textContent = result.inconsistencies.length;
    if (result.inconsistencies.length === 0) {
        conflictsList.innerHTML = '<p class="empty-message">Противоречий не обнаружено</p>';
    } else {
        conflictsList.innerHTML = result.inconsistencies.map(item => `
            <div class="fact-item">
                <div class="fact-statement">📝 ${escapeHtml(item.statement)}</div>
                <div class="fact-reference">⚡ Противоречит: ${escapeHtml(item.fact)}</div>
                <div class="fact-explanation">${escapeHtml(item.explanation)}</div>
            </div>
        `).join('');
    }

    // Подтверждения
    supportCount.textContent = result.supporting_facts.length;
    if (result.supporting_facts.length === 0) {
        supportsList.innerHTML = '<p class="empty-message">Подтверждений не обнаружено</p>';
    } else {
        supportsList.innerHTML = result.supporting_facts.map(item => `
            <div class="fact-item">
                <div class="fact-statement">📝 ${escapeHtml(item.statement)}</div>
                <div class="fact-reference">✓ Подтверждает: ${escapeHtml(item.fact)}</div>
                <div class="fact-explanation">${escapeHtml(item.explanation)}</div>
            </div>
        `).join('');
    }

    // Объяснение
    explanationContent.textContent = result.explanation;

    // Прокрутка к результатам
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Экранирование HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

