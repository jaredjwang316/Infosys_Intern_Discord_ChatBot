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
user_chat_history = {}
total_chat_history = {}

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

def query_data(sql_query):

    # change prompt to not be a hypothetical if correct. Validating will be the next step.
    prompt = f"show me an SQL query for the following message and assume that whatever tables/rows/column names you decide to use are correct. DO NOT GIVE ME ANY EXPLANATIONS, I ONLY WANT TO SEE THE SQL QUERIES. I DO NOT WANT ANY OPTIONS. JUST CHOOSE AN OPTION AND SEND IT TO ME: {sql_query}\n"
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
        await message.channel.send(f"📋 Summary:\n{summary}")
        return
    
    if user_message.lower().startswith("query: "):
        sql_query = user_message[7:].strip()
        query = query_data(sql_query)
        await message.channel.send(f"👁️‍🗨️ Query:\n{query}")
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
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so."
    for role, msg in user_chat_history[user_id]:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "Bot:"

    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text.strip()
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        return

    # (!) (!) (!) THIS SECTION IS ONLY FOR DEMO PURPOSES (!) (!) (!)
    # Send and store bot reply
    await message.channel.send(bot_reply)
    total_chat_history[channel_id].append(("Bot", bot_reply))
    user_chat_history[user_id].append(("Bot", bot_reply))

client.run(discord_token)
