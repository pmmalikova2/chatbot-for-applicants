import os
import logging
import sys
import platform
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
if VENV_PYTHON.exists() and platform.system() == "Darwin" and platform.machine() == "x86_64" and os.getenv("BOT_ARM64_REEXEC") != "1":
    env = os.environ.copy()
    env["BOT_ARM64_REEXEC"] = "1"
    os.execve("/usr/bin/arch", ["arch", "-arm64", str(VENV_PYTHON), *sys.argv], env)
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from src.config import load_config
from src.rag_pipeline import build_index, query_pipeline

logging.basicConfig(level=logging.INFO)
load_dotenv()

CONFIG_PATH = "configs/exp_agentic_rewrite_best.yaml"
PIPELINE = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я чат-бот для абитуриентов НИУ ВШЭ (Москва, бакалавриат, 2026). "
        "Задайте вопрос о поступлении: сроки, программы, баллы, олимпиады, скидки, стоимость."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await update.message.chat.send_action("typing")
    try:
        chunks, generation = query_pipeline(PIPELINE, question)
        answer = generation.answer
    except Exception:
        logging.exception("pipeline error")
        answer = "Произошла ошибка при обработке запроса. Попробуйте переформулировать вопрос."
    await update.message.reply_text(answer)


def main():
    global PIPELINE
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Не найден TELEGRAM_BOT_TOKEN. Добавьте его в .env или переменные окружения.")
        return
    cfg = load_config(CONFIG_PATH)
    print("[*] Building index, это займет 1-2 минуты...")
    PIPELINE = build_index(cfg)
    print("[*] Готово. Бот запускается.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
