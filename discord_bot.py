import os
import discord
from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import mysql.connector
from mysql.connector import Error as  MySQLError
import re

# Load .env values
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")

with open("schema.txt", "r") as f:
    SCHEMA_TEXT = f.read()

# DB config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "user": os.getenv("DB_USER", "test"),
    "password": os.getenv("DB_PASS", "1234"),
    "database": os.getenv("DB_NAME", "chatops"),
}
conn = mysql.connector.connect(**DB_CONFIG)
cur =  conn.cursor()

# Check keys
if not api_key or not discord_token:
    print("❌ Missing keys in .env")
    exit()

# Configure Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
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

def summarize_conversation(history):
    # prompt will only be applied when looking at the entire conversation

    # PROMT_1 : headers with bulletpoints
    # Summarize the following conversation in a nicely formatted paragraph. Make sure that it is readible with headers all the main points if there are more than one. Under each Header I want bulletpoints of the main points:
    
    # PROMPT_2 : 

    conversation_text = ""
    for role, message in history:
        if role.lower() == "bot":
            conversation_text += f"AI: {message}\n"
        else:
            conversation_text += f"User: {message}\n"
    
    conversation_text = conversation_text.strip()
    
    prompt_template = f"""
    You are an expert at summarizing conversations. Your task is to summarize the following conversation.

    ### INSTRUCTION ###
    Summarize the conversation provided below. Your summary must be nicely formatted.
    - Start with a short introductory paragraph.
    - Use descriptive headers for each main topic discussed.
    - Under each header, use bullet points to list the key details and conclusions.

    ### CONVERSATION LOG ###
    ---
    {conversation_text}
    ---

    ### SUMMARY ###
    """

    message = [
        HumanMessage(content=prompt_template)
    ]
    response =  model.invoke(message)
    return response.content.strip()

def query_data(sql_query):
    # change prompt to not be a hypothetical if correct. Validating will be the next step.

    db_schema = SCHEMA_TEXT

    prompt_template = f"""
    You are an expert at querying databases. Your task is to generate a SQL query based on the user's request.

    ### INSTRUCTION ###
    Given the database schema below, generate a SQL query that fulfills the user's request.
    - Ensure the SQL query is syntactically correct.
    - Use appropriate table and column names from the schema.
    - Do not use comments, markdown, or any other formatting in the SQL query.
    
    ### DATABASE SCHEMA ###
    {db_schema}

    ### USER REQUEST ###
    {sql_query}

    ### SQL QUERY ###
    """

    print(prompt_template)

    message = [
        HumanMessage(content=prompt_template)
    ]
    response = model.invoke(prompt_template).content.strip()

    def strip_query(query):
        return query.strip().strip('`').strip('"').strip("'").replace('sql', '').replace('SQL', '').strip()
    
    response = strip_query(response)

    count = 0
    while not is_valid_sql(response):
        print(response)
        count += 1
        if count > 3:
            return "❌ Unable to generate a valid SQL query after multiple attempts.", False
        
        reprompt_template = f"""
        The SQL query you provided is not valid. Please generate a correct SQL query based on the user's request and the database schema.
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query.

        ### DATABASE SCHEMA ###
        {db_schema}

        ### USER REQUEST ###
        {sql_query}

        ### PREVIOUS SQL QUERY ###
        {response}

        ### NEW SQL QUERY ###
        """
        response = model.invoke(reprompt_template).content.strip()
        response = strip_query(response)

    return response, True

def is_valid_sql(query):
    if not query or not isinstance(query, str):
        return False

    cleaned_text = query.upper().strip()
    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN"]
    allowed_table_names = ["EMPLOYEE", "DEPARTMENT", "PROJECT"]
    
    if not cleaned_text.startswith("SELECT"):
        return False
    if "FROM" not in cleaned_text:
        return False
    if not any(keyword in cleaned_text for keyword in sql_keywords):
        return False
    
    from_pattern = r"\bFROM\s+(`|\")?(\w+)(`|\")?"
    join_pattern = r"\bJOIN\s+(`|\")?(\w+)(`|\")?"

    from_tables = re.findall(from_pattern, cleaned_text)
    join_tables = re.findall(join_pattern, cleaned_text)

    extracted_tables = [table[1] for table in from_tables + join_tables]
    if not extracted_tables:
        return False

    is_valid = set(extracted_tables).issubset(set(allowed_table_names))

    return is_valid

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
        query, result = query_data(sql_query)

        print(query)

        if not result:
            await message.channel.send(query)
            return
        else:
            cur.execute(query)
            if not query:
                await message.channel.send(f"❌ Could not execute SQL even after retry. Final SQL was:\nsql\n{query}\n")
                return
            
            rows = cur.fetchall()

            if not rows:
                await message.channel.send("🔍 No results found.")
                return
            
            cols = [desc[0] for desc in cur.description]
            lines = [" | ".join(cols)]
            lines += [" | ".join(map(str, row)) for row in rows]
            table = "```" + "\n".join(lines) + "```"
            await message.channel.send(table)

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
        response = model.invoke(full_prompt)
        bot_reply = response.content.strip()
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        return

    # (!) (!) (!) THIS SECTION IS ONLY FOR DEMO PURPOSES (!) (!) (!)
    # Send and store bot reply
    await message.channel.send(bot_reply)
    total_chat_history[channel_id].append(("Bot", bot_reply))
    user_chat_history[user_id].append(("Bot", bot_reply))

client.run(discord_token)
