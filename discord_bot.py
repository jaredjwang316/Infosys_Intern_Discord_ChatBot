import os
import re
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
import mysql.connector
from mysql.connector import Error as  MySQLError
import re
# import psycopg2

from functions.query import query_data
from functions.summary import summarize_conversation
from functions.search import search_conversation

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")
model_name = os.getenv("MODEL_NAME")

with open("./database/schema.txt", "r") as f:
    SCHEMA_TEXT = f.read()

# DB config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "database": os.getenv("DB_NAME"),
}
conn = mysql.connector.connect(**DB_CONFIG)
cur =  conn.cursor()

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

# Bot intents
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
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = message.author.id
    user_name = message.author
    user_message = message.content.strip()
    channel_id = message.channel.id

    # Init chat history
    if user_id not in user_chat_history:
        user_chat_history[user_id] = []

    if channel_id not in total_chat_history:
        total_chat_history[channel_id] = []


    # Handle special commands
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
    
    if user_message.lower().startswith("query: "):
        user_query = user_message[7:].strip()
        texts = query_data(user_query)
        for text in texts:
            await message.channel.send(text)
        return

    if user_message.lower().startswith("search: "):
        search_query = user_message[8:].strip()
        search_result = search_conversation(user_chat_history[user_id], search_query)
        await message.channel.send(f"🔎 Search:\n{search_result}")
        return
    


    # TESTING SECTION START ---------------------------------------------------------------------------------------------------------
    # anything here will run when you say "test" to the bot in a discord chat
    if user_message.lower() == "test":
        await message.channel.send(user_id)
        await message.channel.send(f"{user_name.mention} just sent me a message")
        return
    
    # this will show the current history the chat bot has stored
    # later make it so that it will show histories for each channel it has stored seperatley
    if user_message.lower() == "show_history":
        await message.channel.send(total_chat_history[channel_id])
        return

    #
    if user_message.lower() == "where_am_i":
        await message.channel.send(message.channel.name)
        await message.channel.send(message.channel.id)
        return

    # TESTING SECTION END -----------------------------------------------------------------------------------------------------------

    # Update history
    total_chat_history[channel_id].append((f"{user_name}", user_message))
    user_chat_history[user_id].append((f"{user_name}", user_message))

    # Build prompt
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so.\n"
    for role, msg in user_chat_history[user_id]:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "Bot:"

    print(full_prompt) # just to test

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
