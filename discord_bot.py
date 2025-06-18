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

load_dotenv()
api_key       = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")

with open("schema.txt", "r") as f:
    SCHEMA_TEXT = f.read()

with open("schema_test.sql", "r") as f:
    raw_schema = f.read()

table_names = re.findall(
   r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([A-Za-z0-9_]+)`?",
   raw_schema,
   flags = re.IGNORECASE
)

allowed_tables = {name.upper() for name in table_names}



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
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# ── Discord client setup ──────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ── In-memory chat history ─────────────────────────────────────────────────────
user_chat_history  = {}  # per-user
total_chat_history = {}  # per-channel

# ── Existing summary helper ───────────────────────────────────────────────────
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
    blacklisted_sql_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    allowed_table_names = list(allowed_tables)
    
    if not cleaned_text.startswith("SELECT"):
        print("Does not start with SELECT")
        return False
    if "FROM" not in cleaned_text:
        print("No FROM clause")
        return False
    if any(keyword in cleaned_text for keyword in blacklisted_sql_keywords):
        print("Contains blacklisted keywords")
        return False
    
    from_pattern = r"\bFROM\s+(`|\")?(\w+)(`|\")?"
    join_pattern = r"\bJOIN\s+(`|\")?(\w+)(`|\")?"

    from_tables = re.findall(from_pattern, cleaned_text)
    join_tables = re.findall(join_pattern, cleaned_text)

    from_tables = [m[1] for m in re.findall(from_pattern, cleaned_text)]
    join_tables = [m[1] for m in re.findall(join_pattern, cleaned_text)]

    # ensure every referenced table is in our schema
    for tbl in from_tables + join_tables:
        if tbl.upper() not in allowed_tables:
            print(f"Table '{tbl}' not in schema")
            return False

    return True

# uses Chroma, FAISS is hard to install on macos - rochan
def search_conversation(history, search_query):
    # turn history into Documents
    docs = [
        Document(page_content=message, metadata={"role": role})
        for role, message in history
    ]

    # chunk into ~1000-char slices with 200-char overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # embed & index in Chroma (or FAISS)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectorstore = FAISS.from_documents(chunks, embeddings) # FAISS (difficult on mac)
    vectorstore = Chroma.from_documents(chunks, embeddings) # Chroma (works on mac)

    # build a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="map_reduce", # robust for aggregation
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # top-5 chunks
        return_source_documents=False
    )

    # run
    prompt = (
        f"Please gather *all* the information related to “{search_query}” "
        "from the conversation, and present it as concise bullet points."
    )

    # return qa.run(prompt) # deprecated
    return qa.invoke(prompt)

def split_response(response):
    max_length = 1900
    return [response[i:i+max_length] for i in range(0, len(response), max_length)]

def format_table(table):
    max_length = 2000
    lines = table.split("\n")
    formatted_lines = []
    current_chunk = ""

    curr_len = 0
    for line in lines:
        if curr_len + len(line) >= max_length:
            formatted_lines.append(current_chunk)
            current_chunk = line + "\n"
            curr_len = len(line) + 1
        else:
            current_chunk += line + "\n"
            curr_len += len(line) + 1
    if current_chunk:
        formatted_lines.append(current_chunk)
    return formatted_lines


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
        sql_query = user_message[7:].strip()
        query, result = query_data(sql_query)

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
            table = "\n".join(lines)
            tables = format_table(table)
            for table in tables:
                text = "```" + table + "```"
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
