import os
import csv
import yadisk
import logging
import requests
import pandas as pd 
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
EXCEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_actions.xlsx")
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
YANDEX_OAUTH_TOKEN = os.getenv('YANDEX_OAUTH_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_DISK_TOKEN = os.getenv('YANDEX_DISK_TOKEN')

def convert_csv_to_xlsx():
    """Конвертирует CSV файл в XLSX формат"""
    try:
        if os.path.exists(EXCEL_FILE):
            os.remove(EXCEL_FILE)
            logger.info(f"Старый файл {EXCEL_FILE} удален.")
        
        df = pd.read_csv(LOG_FILE)
        df.to_excel(EXCEL_FILE, index=False)
        logger.info(f"Файл {LOG_FILE} успешно конвертирован в {EXCEL_FILE}")
        return True
    except Exception as e:
        logger.error(f"Ошибка конвертации CSV в XLSX: {e}")
        return False

def upload_to_yandex_disk():
    """Загружает файл на Яндекс.Диск с проверками"""
    try:
        y = yadisk.YaDisk(token=YANDEX_DISK_TOKEN)
        
        if not y.check_token():
            logger.error("Токен Яндекс.Диска недействителен или отсутствует")
            return False
        
        if not os.path.exists(EXCEL_FILE):
            logger.error(f"Локальный файл {EXCEL_FILE} не существует")
            return False
            
        remote_dir = "/bot_logs"
        remote_path = f"{remote_dir}/user_actions.xlsx"
        
        if not y.exists(remote_dir):
            logger.info(f"Создаю папку {remote_dir} на Яндекс.Диске")
            y.mkdir(remote_dir)
        
        logger.info(f"Начинаю загрузку {EXCEL_FILE} на Яндекс.Диск...")
        y.upload(EXCEL_FILE, remote_path, overwrite=True) 
        
        if y.exists(remote_path):
            file_info = y.get_meta(remote_path)
            logger.info(f"Файл успешно загружен! Размер на диске: {file_info['size']} байт")
            return True
            
        logger.error("Файл не появился на Яндекс.Диске после загрузки")
        return False
        
    except yadisk.exceptions.UnauthorizedError:
        logger.error("Ошибка авторизации на Яндекс.Диске. Проверьте токен.")
    except Exception as e:
        logger.error(f"Неизвестная ошибка при загрузке: {str(e)}")
    return False

def backup_logs():
    """Создает бэкап и загружает на Яндекс.Диск"""
    try:
        logger.info("Запуск процедуры бэкапа...")
        
        if not os.path.exists(LOG_FILE):
            logger.error("Файл логов не существует!")
            return False
            
        if os.path.getsize(LOG_FILE) == 0:
            logger.warning("Файл логов пуст, бэкап не требуется")
            return False
            
        if convert_csv_to_xlsx():
            logger.info(f"Файл {EXCEL_FILE} успешно создан")
            
            if upload_to_yandex_disk():
                logger.info("Бэкап успешно завершен")
                return True
                
        return False
    except Exception as e:
        logger.error(f"Критическая ошибка при бэкапе: {e}")
        return False

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
            writer.writerow([
                "user_id", 
                "timestamp", 
                "action_type", 
                "user_message_length",
                "bot_response_length",
                "processing_time_sec",
                "user_message",  
                "bot_response",  
                "full_action" 
            ])

def log_user_action(
    user_id: int, 
    action_type: str, 
    user_message: str = "", 
    bot_response: str = "", 
    processing_time: float = 0.0
):
    """Логирует действие пользователя с расширенной информацией"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = [
        user_id,
        timestamp,
        action_type,
        len(user_message),
        len(bot_response),
        round(processing_time, 2),
        user_message,
        bot_response,
        f"{action_type}: {user_message}"
    ]
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)
        
        logger.info(f"Записано в лог: user_id={user_id}, action={action_type}")
        
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip()) 
            
        if line_count >= 10 and line_count % 10 == 0: 
            logger.info(f"Набралось {line_count} записей - запускаю бэкап...")
            backup_logs()
            
    except Exception as e:
        logger.error(f"Ошибка записи в лог: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_user_action(
        user_id=user.id,
        action_type="command",
        user_message="/start"
    )
    await update.message.reply_html(rf"Hi {user.mention_html()}!")

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
    help_text = """Доступные команды:
                    /start - Начать работу с ботом
                    /help - Показать это сообщение
                    /joke - Рассказать анекдот"""
    
    log_user_action(
        user_id=update.message.from_user.id,
        action_type="command",
        user_message="/help"
    )
    await update.message.reply_text(help_text)
    
async def joke_command(update: Update, context: CallbackContext) -> None:
    """Обработчик команды /joke"""
    user = update.effective_user
    start_time = datetime.now()
    prompt = "Расскажи анекдот про Филиппа Киркорова"
    
    try:
        iam_token = get_iam_token(YANDEX_OAUTH_TOKEN)
        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt", 
            "completionOptions": {"temperature": 0.7, "maxTokens": 500}, 
            "messages": [{"role": "user", "text": prompt}]
        }
        response = requests.post(
            YANDEX_API_URL, 
            headers={"Authorization": f"Bearer {iam_token}"},
            json=data,
        ).json()

        answer = (
            response.get('result', {})
                .get('alternatives', [{}])[0]
                .get('message', {})
                .get('text', "Не удалось получить анекдот.")
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        log_user_action(
            user_id=user.id,
            action_type="command",
            user_message="/joke",
            bot_response=answer,
            processing_time=processing_time
        )
        await update.message.reply_text(answer)

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        log_user_action(
            user_id=user.id,
            action_type="error",
            user_message="/joke",
            bot_response=str(e),
            processing_time=processing_time
        )
        logger.error(f"Ошибка в команде /joke: {e}")
        await update.message.reply_text("Не удалось получить анекдот, попробуйте позже.")

async def process_message(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    user_text = update.message.text
    start_time = datetime.now()
    
    try:
        iam_token = get_iam_token(YANDEX_OAUTH_TOKEN)
        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt", 
            "completionOptions": {"temperature": 0.3, "maxTokens": 1000},
            "messages": [{"role": "user", "text": user_text}]
        }
        response = requests.post(
            YANDEX_API_URL, 
            headers={"Authorization": f"Bearer {iam_token}"},
            json=data,
        ).json()

        answer = (
            response.get('result', {})
               .get('alternatives', [{}])[0]
               .get('message', {})
               .get('text', "Не удалось получить ответ от модели.")
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        log_user_action(
            user_id=user.id,
            action_type="text",
            user_message=user_text,
            bot_response=answer,
            processing_time=processing_time
        )
        await update.message.reply_text(answer)

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        log_user_action(
            user_id=user.id,
            action_type="error",
            user_message=user_text,
            bot_response=str(e),
            processing_time=processing_time
        )
        logger.error(f"Ошибка обработки сообщения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке запроса.")

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_user_action(
        user_id=user.id,
        action_type="command",
        user_message="/backup"
    )
    if backup_logs():
        await update.message.reply_text("Логи успешно сохранены на Яндекс.Диск")
    else:
        await update.message.reply_text("Ошибка при создании бэкапа логов")

def main() -> None:
    """Запуск бота"""
    init_log_file()
    print(f"Файл логов будет создан/использован: {os.path.abspath(LOG_FILE)}")
    
    try:
        y = yadisk.YaDisk(token=YANDEX_DISK_TOKEN)
        if y.check_token():
            logger.info("Подключение к Яндекс.Диску успешно")
            if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
                logger.info("Делаю начальный бэкап существующих логов...")
                backup_logs()
        else:
            logger.warning("Не удалось подключиться к Яндекс.Диску")
    except Exception as e:
        logger.error(f"Ошибка проверки Яндекс.Диска: {e}")
    
    if not all([TELEGRAM_TOKEN, YANDEX_OAUTH_TOKEN, YANDEX_FOLDER_ID, YANDEX_DISK_TOKEN]):
        logger.error("Не все необходимые переменные окружения установлены!")
        raise ValueError("Проверьте .env файл")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("backup", backup_command)) 
    application.add_handler(CommandHandler("joke", joke_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()