import os
import textwrap
import google.generativeai as genai
from telegram import Bot
from telegram.constants import ParseMode

# Get env vars from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not GEMINI_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing required environment variables. Check GitHub secrets.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

def generate_instagram_idea():
    prompt = """
You are an expert at creating viral Instagram Reels for English learning.

Generate ONE complete reel concept with this structure in plain text:

[HOOK - max 8 words]
A super catchy hook line that grabs attention.

[ON-SCREEN SCRIPT - max 90 seconds]
Write what the creator should SAY on screen.
Short, punchy sentences. Simple English.
Format with short lines, easy to read.

[CAPTION]
A short caption for the post.

[HASHTAGS]
10–15 relevant hashtags in one line.

Target audience: people improving their English / exam prep.
Tone: motivating, clear, slightly funny but not cringe.
"""
    response = model.generate_content(prompt)
    text = response.text.strip()
    return text

def send_to_telegram(message: str):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    # Telegram has a 4096 char limit per message, so we may need to split
    chunks = textwrap.wrap(message, width=3900)
    first = True
    for chunk in chunks:
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=chunk,
            parse_mode=ParseMode.MARKDOWN
        )

def main():
    try:
        idea = generate_instagram_idea()
        final_message = (
            "🔥 *New Instagram Reel Idea Generated!*\n\n"
            + idea
            + "\n\n---\nIf this was useful, save this chat."
        )
        send_to_telegram(final_message)
        print("Message sent to Telegram successfully.")
    except Exception as e:
        # If something goes wrong, at least log it in GitHub Actions
        print("Error occurred:", e)
        raise

if __name__ == "__main__":
    main()
