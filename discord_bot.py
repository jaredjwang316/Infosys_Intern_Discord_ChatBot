import os
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# import psycopg2

from functions.query import query_data
from functions.summary import summarize_conversation
from functions.search import search_conversation

load_dotenv()
api_key       = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")
model_name = os.getenv("MODEL_NAME")

# Check keys
if not api_key or not discord_token:
    print("❌ Missing keys in .env")
    exit()

# gemini
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# ── Discord client setup ──────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory: {user_id: [(role, message)]}
user_chat_history = {}
total_chat_history = {}

def split_response(response):
    max_length = 1900
    lines = response.split('\n')
    parts = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            parts.append(current)
            current = line
        else:
            if current:
                current += '\n' + line
            else:
                current = line
    if current:
        parts.append(current)
    return parts

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id      = message.author.id
    user_name    = message.author.mention
    user_message = message.content.strip()
    channel_id   = message.channel.id

    # Initialize histories
    user_chat_history.setdefault(user_id, [])
    total_chat_history.setdefault(channel_id, [])

    # ---- Special commands -----------------------
    if user_message.lower() == "exit":
        user_chat_history[user_id] = []
        await message.channel.send("🧠 Memory cleared!")
        return

    if user_message.lower() == "summary":
        summary = summarize_conversation(user_chat_history[user_id])
        response = split_response(summary)
        await message.channel.send(f"📋 Summary:\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        return

    # ---- NEW /query handler ---------------------
    if user_message.lower().startswith("query: "):
        user_query = user_message[7:].strip()
        texts = query_data(user_query)
        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)
        return

    # ---- Existing /search handler ---------------
    if user_message.lower().startswith("search: "):
        terms = user_message[len("search: "):].strip()
        result = search_conversation(user_chat_history[user_id], terms)
        await message.channel.send(f"🔎 Search:\n{result}")
        return

    # ---- Testing utilities ----------------------
    if user_message.lower() == "test":
        await message.channel.send(f"Your ID: {user_id}")
        await message.channel.send(f"{user_name} just sent me a message")
        return

    if user_message.lower() == "show_history":
        await message.channel.send(total_chat_history[channel_id])
        return

    if user_message.lower() == "where_am_i":
        await message.channel.send(f"Channel: {message.channel.name} ({channel_id})")
        return

    # ---- Default chat behavior -----------------
    total_chat_history[channel_id].append((user_name, user_message))
    user_chat_history[user_id].append((user_name, user_message))

    # Build prompt
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so.\n"
    for role, msg in user_chat_history[user_id]:
        full_prompt += f"{role}: {msg}\n"

    try:
        response = model.invoke(full_prompt)
        bot_reply = response.content.strip()
        replies = split_response(bot_reply)
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        return

    # (!) (!) (!) THIS SECTION IS ONLY FOR DEMO PURPOSES (!) (!) (!)
    # Send and store bot reply
    for reply in replies:
        await message.channel.send(reply)
    total_chat_history[channel_id].append(("Bot", bot_reply))
    user_chat_history[user_id].append(("Bot", bot_reply))

client.run(discord_token)
