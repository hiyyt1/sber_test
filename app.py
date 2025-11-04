import yaml, asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from src.route import predict

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

BOT_TOKEN = ""

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    res = predict(text)

    llm_note = ""
    if res.get("stage", "").endswith("+llm"):
        llm_note = " (с LLM-валидацией)"

    msg = (
        f"📩 *Запрос:* {text}\n"
        f"📌 *Класс:* `{res['label']}`{llm_note}\n"
        f"📈 *Уверенность:* {res['confidence']:.2f}\n"
        f"🧩 *Режим:* {res['stage']}\n\n"
    )
    if "examples" in res and res["examples"]:
        msg += "🧠 *Похожие запросы:*\n" + "\n".join(f"- {ex}" for ex in res["examples"])

    await update.message.reply_markdown(msg)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    print("🤖 Telegram-бот запущен.")
    asyncio.run(app.run_polling())
