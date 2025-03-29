import os
import csv
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_actions.csv")
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
YANDEX_OAUTH_TOKEN = os.getenv('YANDEX_OAUTH_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion" 

def get_iam_token(oauth_token: str) -> str:
    """Получаем IAM-токен для Yandex Cloud API."""
    try:
        response = requests.post(
            'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            json={'yandexPassportOauthToken': oauth_token}
        )
        response.raise_for_status()
        return response.json()['iamToken']
    except Exception as e:
        logger.error(f"Ошибка получения IAM токена: {e}")
        raise

def init_log_file():
    """Создает файл лога с заголовками, если он не существует"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["user_id", "timestamp", "action"])

def log_user_action(user_id: int, action: str):
    """Логирует действие пользователя в CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = [user_id, timestamp, action]
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)
        logger.info(f"Записано в лог: {log_entry}") 
    except Exception as e:
        logger.error(f"Ошибка записи в лог: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start с логированием"""
    user = update.effective_user
    log_user_action(user.id, "/start command")
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help с логированием"""
    log_user_action(update.message.from_user.id, "/help command")
    await update.message.reply_text("Help!")

async def process_message(update: Update, context: CallbackContext) -> None:
    """Обработка текстовых сообщений с логированием"""
    user = update.effective_user
    user_text = update.message.text
    
    # Логируем вопрос пользователя
    log_user_action(user.id, f"User question: {user_text[:50]}")  # Логируем первые 50 символов
    
    try:
        iam_token = get_iam_token(YANDEX_OAUTH_TOKEN) 

        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt", 
            "completionOptions": {"temperature": 0.3, "maxTokens": 1000},
            "messages": [{"role": "user", "text": user_text}]
        }

        response = requests.post(
            YANDEX_API_URL, 
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {iam_token}"
            },
            json=data,
        ).json()

        answer = response.get('result', {})\
                         .get('alternatives', [{}])[0]\
                         .get('message', {})\
                         .get('text', "Не удалось получить ответ от модели.")
        
        log_user_action(user.id, "Bot response")
        
        await update.message.reply_text(answer)

    except Exception as e:
        log_user_action(user.id, f"Error: {str(e)}")
        logger.error(f"Ошибка обработки сообщения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке запроса.")

def main() -> None:
    """Запуск бота с инициализацией логов"""
    init_log_file()
    print(f"Файл логов будет создан/использован: {os.path.abspath(LOG_FILE)}")
    
    if not all([TELEGRAM_TOKEN, YANDEX_OAUTH_TOKEN, YANDEX_FOLDER_ID]):
        logger.error("Не все необходимые переменные окружения установлены!")
        raise ValueError("Проверьте .env файл")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()