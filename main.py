import os
import csv
import random
import datetime
import time
import textwrap
import asyncio

import requests
from huggingface_hub import InferenceClient
from moviepy.editor import (
    ColorClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip,
)
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import cloudinary
import cloudinary.uploader


# ====== CONFIG FROM ENV ======
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTA_ID = os.getenv("INSTAGRAM_BUSINESS_ID")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_FILE = "english_schedule.csv"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ====== BASIC VALIDATION ======
missing = []
for name, value in [
    ("INSTAGRAM_ACCESS_TOKEN", ACCESS_TOKEN),
    ("INSTAGRAM_BUSINESS_ID", INSTA_ID),
    ("HUGGINGFACE_TOKEN", HUGGINGFACE_TOKEN),
    ("HF_MODEL_ID", HF_MODEL_ID),
    ("CLOUDINARY_CLOUD_NAME", CLOUDINARY_CLOUD_NAME),
    ("CLOUDINARY_API_KEY", CLOUDINARY_API_KEY),
    ("CLOUDINARY_API_SECRET", CLOUDINARY_API_SECRET),
    ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
    ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
]:
    if not value:
        missing.append(name)

if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


# ====== CLOUDINARY CONFIG ======
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)


# ====== HUGGING FACE INFERENCE ======
hf_client = InferenceClient(model=HF_MODEL_ID, token=HUGGINGFACE_TOKEN)


# ====== TELEGRAM HELPER ======
def send_telegram(message: str):
    """Send a Telegram message; ignore failures."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.get(url, params={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4000]})
    except Exception as e:
        print("Telegram send failed:", e)


# ====== SCHEDULE / TOPIC PICKER ======
def pick_today_theme_and_topic():
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Monday=0..Sunday=6 -> 1..7
    utc_weekday = datetime.datetime.utcnow().weekday()  # 0..6
    today_day = utc_weekday + 1

    for r in rows:
        if int(r["Day"]) == today_day:
            theme = r["Theme"]
            subtopics = [x.strip() for x in r["SubTopics"].split(",") if x.strip()]
            sub_topic = random.choice(subtopics)
            return theme, sub_topic

    # Fallback if CSV missing matching day
    first = rows[0]
    theme = first["Theme"]
    subtopics = [x.strip() for x in first["SubTopics"].split(",") if x.strip()]
    sub_topic = random.choice(subtopics)
    return theme, sub_topic


# ====== TEXT GENERATION (HUGGING FACE) ======
def generate_piece(kind: str, theme: str, sub_topic: str) -> str:
    prompt = f"""
You are creating viral Instagram content for an English learning page.

Type: {kind}
Theme: {theme}
Sub-topic: {sub_topic}

Requirements:
- First line must be a strong HOOK that stops scrolling.
- Teach ONE clear point (definition, example, or common mistake).
- Include at least one quiz or challenge (like "Choose A/B/C" or "Fill in the blank").
- End with a clear CTA such as "Comment your score", "Tag a friend", or "Duet this".
- Use natural emojis (3–8 total).
- Add 3-5 relevant English-learning hashtags at the end.
Return ONLY the final text, no explanations, no markdown, no quotes, no labels.
"""

    try:
        response = hf_client.text_generation(
            prompt,
            max_new_tokens=220,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
        )
        text = response.strip()

        if len(text) < 60:
            raise ValueError("Generated text too short")

        return text
    except Exception as e:
        print("HF generation failed:", e)
        return (
            f"{kind} about {sub_topic}: What does it mean to you? "
            f"Comment your own example! 😎 #EnglishLearning #EnglishReels"
        )


# ====== TTS (EDGE-TTS) ======
async def _tts_to_file(text: str, filename: str, voice: str = "en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)


def create_tts(text: str, filename: str, voice: str = "en-US-AriaNeural"):
    try:
        # Remove hashtags from spoken text
        spoken = "\n".join(line for line in text.splitlines() if "#" not in line)
        asyncio.run(_tts_to_file(spoken, filename, voice))
        return filename
    except Exception as e:
        send_telegram(f"TTS failed: {e}")
        print("TTS error:", e)
        return None


# ====== VIDEO CREATION (MOVIEPY) ======
def split_script_into_chunks(text: str, max_words: int = 10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


def create_reel_video(script: str, audio_file: str | None, output_file: str) -> str | None:
    try:
        spoken_text = "\n".join(
            line for line in script.splitlines() if "#" not in line
        ).strip()
        if not spoken_text:
            spoken_text = script

        chunks = split_script_into_chunks(spoken_text, max_words=10)

        if not chunks:
            raise ValueError("No chunks for reel")

        clips = []
        for chunk in chunks:
            bg = ColorClip(size=(1080, 1920), color=(0, 90, 200)).set_duration(3.5)
            txt = TextClip(
                chunk,
                fontsize=70,
                color="white",
                font="DejaVu-Sans-Bold",
                method="caption",
                size=(950, None),
            ).set_duration(3.5)
            txt = txt.set_position(("center", "center"))
            clip = CompositeVideoClip([bg, txt]).fadein(0.5).fadeout(0.5)
            clips.append(clip)

        video = concatenate_videoclips(clips)

        if audio_file and os.path.exists(audio_file):
            audio = AudioFileClip(audio_file)
            final_duration = max(video.duration, audio.duration)
            video = video.set_duration(final_duration)
            audio = audio.set_duration(final_duration)
            video = video.set_audio(audio)

        video.write_videofile(
            output_file,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
        return output_file
    except Exception as e:
        send_telegram(f"Reel creation failed: {e}")
        print("Reel creation error:", e)
        return None


# ====== IMAGE CREATION (PILLOW) ======
def create_image(text: str, filename: str, title: str | None = None) -> str | None:
    try:
        img = Image.new("RGB", (1080, 1080), color=(0, 90, 200))
        draw = ImageDraw.Draw(img)
        font_title = ImageFont.truetype(FONT_PATH, 80)
        font_body = ImageFont.truetype(FONT_PATH, 56)

        # Border
        draw.rectangle((20, 20, 1060, 1060), outline="white", width=5)

        y = 60
        if title:
            draw.text((60, y), title, font=font_title, fill="white")
            y += 140

        wrapped = []
        for line in text.splitlines():
            wrapped.extend(textwrap.wrap(line, width=24) or [""])

        for line in wrapped:
            draw.text((60, y), line, font=font_body, fill="white")
            y += 70
            if y > 980:
                break

        img.save(filename)
        return filename
    except Exception as e:
        send_telegram(f"Image creation failed: {e}")
        print("Image creation error:", e)
        return None


# ====== CLOUDINARY UPLOAD ======
def upload_to_cloudinary(path: str) -> str | None:
    """
    Uploads image or video to Cloudinary (resource_type=auto).
    Returns secure URL or None.
    """
    try:
        result = cloudinary.uploader.upload(path, resource_type="auto")
        url = result.get("secure_url")
        if not url:
            raise RuntimeError(f"No secure_url in Cloudinary response: {result}")
        return url
    except Exception as e:
        send_telegram(f"Cloudinary upload failed for {path}: {e}")
        print("Cloudinary upload error:", e)
        return None


# ====== INSTAGRAM POSTING (GRAPH API) ======
GRAPH_BASE = "https://graph.facebook.com/v20.0"


def post_video_to_instagram(video_url: str, caption: str):
    try:
        create_url = f"{GRAPH_BASE}/{INSTA_ID}/media"
        payload = {
            "access_token": ACCESS_TOKEN,
            "media_type": "VIDEO",
            "video_url": video_url,
            "caption": caption,
        }
        r = requests.post(create_url, data=payload)
        data = r.json()
        if "id" not in data:
            raise RuntimeError(f"Create video failed: {data}")
        creation_id = data["id"]

        publish_url = f"{GRAPH_BASE}/{INSTA_ID}/media_publish"
        r2 = requests.post(
            publish_url,
            data={"access_token": ACCESS_TOKEN, "creation_id": creation_id},
        )
        if not r2.ok:
            raise RuntimeError(f"Publish video failed: {r2.text}")
    except Exception as e:
        send_telegram(f"Posting Reel error: {e}")
        print("Posting Reel error:", e)


def post_image_to_instagram(image_url: str, caption: str):
    try:
        create_url = f"{GRAPH_BASE}/{INSTA_ID}/media"
        r = requests.post(
            create_url,
            data={"access_token": ACCESS_TOKEN, "image_url": image_url, "caption": caption},
        )
        data = r.json()
        if "id" not in data:
            raise RuntimeError(f"Create image failed: {data}")
        creation_id = data["id"]

        publish_url = f"{GRAPH_BASE}/{INSTA_ID}/media_publish"
        r2 = requests.post(
            publish_url,
            data={"access_token": ACCESS_TOKEN, "creation_id": creation_id},
        )
        if not r2.ok:
            raise RuntimeError(f"Publish image failed: {r2.text}")
    except Exception as e:
        send_telegram(f"Posting image error: {e}")
        print("Posting image error:", e)


def post_carousel_to_instagram(image_urls: list[str], caption: str):
    try:
        child_ids = []
        create_url = f"{GRAPH_BASE}/{INSTA_ID}/media"

        for url in image_urls:
            r = requests.post(
                create_url,
                data={"access_token": ACCESS_TOKEN, "image_url": url},
            )
            data = r.json()
            if "id" not in data:
                raise RuntimeError(f"Create carousel child failed: {data}")
            child_ids.append(data["id"])

        parent_r = requests.post(
            create_url,
            data={
                "access_token": ACCESS_TOKEN,
                "caption": caption,
                "media_type": "CAROUSEL",
                "children": ",".join(child_ids),
            },
        )
        parent_data = parent_r.json()
        if "id" not in parent_data:
            raise RuntimeError(f"Create carousel parent failed: {parent_data}")
        creation_id = parent_data["id"]

        publish_url = f"{GRAPH_BASE}/{INSTA_ID}/media_publish"
        r2 = requests.post(
            publish_url,
            data={"access_token": ACCESS_TOKEN, "creation_id": creation_id},
        )
        if not r2.ok:
            raise RuntimeError(f"Publish carousel failed: {r2.text}")
    except Exception as e:
        send_telegram(f"Posting carousel error: {e}")
        print("Posting carousel error:", e)


# ====== MAIN FLOW ======
def main():
    send_telegram("🚀 Script started – generating today's content...")

    try:
        theme, sub_topic = pick_today_theme_and_topic()
        send_telegram(f"Today: Theme = {theme}, Topic = {sub_topic}")
    except Exception as e:
        send_telegram(f"CSV / schedule error: {e}")
        print("Schedule error:", e)
        return

    # Generate texts
    reel1_script = generate_piece("20-second Reel script", theme, sub_topic)
    reel2_script = generate_piece("another 20-second Reel script", theme, sub_topic)
    post_caption = generate_piece("Static post caption with poll-style CTA", theme, sub_topic)

    carousel_texts = [
        generate_piece(f"Carousel slide {i}", theme, sub_topic) for i in range(1, 5)
    ]

    # Create TTS
    reel1_audio = create_tts(reel1_script, "reel1_audio.mp3")
    reel2_audio = create_tts(reel2_script, "reel2_audio.mp3")

    # Create videos
    reel1_file = create_reel_video(reel1_script, reel1_audio, "reel1.mp4")
    reel2_file = create_reel_video(reel2_script, reel2_audio, "reel2.mp4")

    # Create images
    post_file = create_image(post_caption, "post.png", title=f"{theme}: {sub_topic}")

    carousel_files = []
    for i, text in enumerate(carousel_texts, start=1):
        f = create_image(text, f"carousel{i}.png", title=f"{theme} - Slide {i}")
        if f:
            carousel_files.append(f)

    # Upload & post
    # Reel 1
    if reel1_file:
        url = upload_to_cloudinary(reel1_file)
        if url:
            post_video_to_instagram(url, reel1_script)
            time.sleep(120)

    # Reel 2
    if reel2_file:
        url = upload_to_cloudinary(reel2_file)
        if url:
            post_video_to_instagram(url, reel2_script)
            time.sleep(120)

    # Static post
    if post_file:
        url = upload_to_cloudinary(post_file)
        if url:
            post_image_to_instagram(url, post_caption)
            time.sleep(120)

    # Carousel (need at least 4 slides)
    if len(carousel_files) >= 4:
        urls = []
        for f in carousel_files[:4]:
            u = upload_to_cloudinary(f)
            if u:
                urls.append(u)
        if len(urls) == 4:
            carousel_caption = (
                f"Today's {theme} carousel on {sub_topic}! Save this and comment your score 👇 "
                "#EnglishLearning #EnglishTips"
            )
            post_carousel_to_instagram(urls, carousel_caption)

    send_telegram("✅ Daily automation complete – posts should be live on Instagram.")
    print("Done.")


if __name__ == "__main__":
    main()
