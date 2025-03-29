import logging
import os
import requests
from dotenv import load_dotenv
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Константы из переменных окружения
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет {user.mention_html()}! Я бот с интеграцией Yandex GPT.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(
        "Просто отправьте мне любой текст, и я обработаю его с помощью Yandex GPT!"
    )

async def process_message(update: Update, context: CallbackContext) -> None:
    """Обработка текстовых сообщений с использованием Yandex GPT API."""
    try:
        user_text = update.message.text
        if not user_text:
            await update.message.reply_text("Пожалуйста, отправьте текст для обработки.")
            return

        # Получаем IAM токен
        iam_token = get_iam_token(YANDEX_OAUTH_TOKEN)

        # Формируем запрос к Yandex GPT API
        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt",
            "completionOptions": {
                "temperature": 0.3,
                "maxTokens": 1000
            },
            "messages": [{"role": "user", "text": user_text}]
        }

        response = requests.post(
            YANDEX_API_URL,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {iam_token}"
            },
            json=data,
            timeout=10
        ).json()

        # Извлекаем ответ
        answer = (response.get('result', {})
                         .get('alternatives', [{}])[0]
                         .get('message', {})
                         .get('text', "Не удалось получить ответ от модели."))

        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке запроса.")

def main() -> None:
    """Запуск бота."""
    if not all([TELEGRAM_TOKEN, YANDEX_OAUTH_TOKEN, YANDEX_FOLDER_ID]):
        logger.error("Не все необходимые переменные окружения установлены!")
        raise ValueError("Проверьте .env файл")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()