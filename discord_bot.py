import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env values
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")

# Check keys
if not api_key or not discord_token:
    print("❌ Missing keys in .env")
    exit()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Bot intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory: {user_id: [(role, message)]}
chat_histories = {}

def summarize_conversation(history):
    # prompt will only be applied when looking at the entire conversation

    # PROMT_1 : headers with bulletpoints
    # Summarize the following conversation in a nicely formatted paragraph. Make sure that it is readible with headers all the main points if there are more than one. Under each Header I want bulletpoints of the main points:
    
    # PROMPT_2 : 

    prompt = "Summarize the following conversation in a nicely formatted paragraph. Make sure that it is readible with headers all the main points if there are more than one. Under each Header I want bulletpoints of the main points:\n"
    for role, message in history:
        prompt += f"{role}: {message}\n"
    response = model.generate_content(prompt)
    return response.text.strip()

def query_data():

    prompt = "show me an SQL query for the following message:\n"
    response = model.generate_content(prompt)
    return response.text.strip()

def search_conversation(history, search_query):

    prompt = f"search for the following terms and bulletpoint anything of relevance to {search_query} for me to read:\n"
    for role, message in history:
        prompt += f"{role}: {message}\n"
    response = model.generate_content(prompt)
    return response.text.strip()

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = message.author.id
    user_message = message.content.strip()

    # Init chat history for user
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    # Handle special commands
    if user_message.lower() == "exit":
        chat_histories[user_id] = []
        await message.channel.send("🧠 Memory cleared!")
        return

    if user_message.lower() == "summary":
        summary = summarize_conversation(chat_histories[user_id])
        await message.channel.send(f"📋 Summary:\n{summary}")
        return
    
    if user_message.lower() == "query":
        query = query_data()
        # query = summarize_conversation(chat_histories[user_id])
        await message.channel.send(f"[] Query:\n{query}")
        return

    if user_message.lower().startswith("search: "):
        search_query = user_message[8:].strip()
        search_result = search_conversation(chat_histories[user_id], search_query)
        await message.channel.send(f"[] Search:\n{search_result}")
        return
    


    # Update history
    chat_histories[user_id].append(("User", user_message))

    # Build prompt
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so."
    for role, msg in chat_histories[user_id]:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "Bot:"

    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text.strip()
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        return

    # Send and store bot reply
    await message.channel.send(bot_reply)
    chat_histories[user_id].append(("Bot", bot_reply))

client.run(discord_token)