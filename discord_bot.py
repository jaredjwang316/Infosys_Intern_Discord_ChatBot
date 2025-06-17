import os
import re
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS, Chroma
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.embeddings import SentenceTransformerEmbeddings
import re
import psycopg2

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

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
    
    # feed the schema manually to Gemini so it can generate a correct query, as it cannot access your actual database
    db_schema = """
    -- 1. Employees – Infosys staff
    CREATE TABLE employees (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        role VARCHAR(50) NOT NULL,
        joined_at DATE NOT NULL
    );

    -- 2. Clients – Represents the external clients Infosys serves.
    CREATE TABLE clients (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        industry VARCHAR(50) NOT NULL,
        location VARCHAR(100)
    );

    -- 3. Projects – client projects Infosys runs
    CREATE TABLE projects (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        client_id INTEGER REFERENCES clients(id) ON DELETE CASCADE,
        start_date DATE NOT NULL,
        end_date DATE,
        status VARCHAR(20) NOT NULL
    );

    -- 4. Employee Project Assignments – who is working on what
    CREATE TABLE employee_project_assignments (
        id SERIAL PRIMARY KEY,
        employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
        project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
        assigned_on DATE NOT NULL,
        role_on_project VARCHAR(50) NOT NULL
    );

    -- 5. Skills – skill catalog
    CREATE TABLE skills (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50) NOT NULL
    );

    -- 6. Employee Skills – each employee’s skills
    CREATE TABLE employee_skills (
        employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
        skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
        PRIMARY KEY (employee_id, skill_id)
    );
    """

    prompt_template = f"""
    You are an expert at querying databases. Your task is to generate a SQL query based on the user's request.

    ### INSTRUCTION ###
    Given the database schema below, generate a SQL query that fulfills the user's request.
    - Ensure the SQL query is syntactically correct.
    - Use appropriate table and column names from the schema.
    - Do not use comments, markdown, or any other formatting in the SQL query.
    - You are only allowed to query the Employee table.
    
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
            return "❌ Unable to generate a valid SQL query after multiple attempts."
        
        reprompt_template = f"""
        The SQL query you provided is not valid. Please generate a correct SQL query based on the user's request and the database schema.
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query.
        - You are only allowed to query the Employee table.

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

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()     # Create a cursor object that allows you to Execute SQL commands and fetch results from database
        cur.execute(response)   # Runs the SQL query 
        rows = cur.fetchall()   # Fetches all the rows returned by the executed query.
        colnames = [desc[0] for desc in cur.description]    # gets the names of the columns in the result set.

        cur.close()     # ends the query session
        conn.close()

        if not rows:    #query ran successfully but result is empty
            return "✅ Query executed, but no results found."

        # Format result
        result_lines = [" | ".join(colnames)]   #Joins all column names with | separator to create a readable header 
        result_lines.append("-" * 50)   #50-dash divider line
        for row in rows:
            result_lines.append(" | ".join(str(item) for item in row))

        return "\n".join(result_lines)
    except Exception as e:
        return f"❌ Error executing SQL query: {str(e)}"




def is_valid_sql(query):
    if not query or not isinstance(query, str):
        return False

    cleaned_text = query.upper().strip()
    sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN"]
    allowed_table_names = [    "EMPLOYEES", "CLIENTS", "PROJECTS", 
                           "EMPLOYEE_PROJECT_ASSIGNMENTS", "SKILLS", "EMPLOYEE_SKILLS"]
    
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
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so.\n"
    for role, msg in user_chat_history[user_id]:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "Bot:"

    print(full_prompt) # just to test

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
