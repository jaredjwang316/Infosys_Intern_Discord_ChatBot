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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
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

# gemini
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Memory: {user_id: [(role, message)]}
user_chat_history = {}
total_chat_history = {}

def generate_query(sql_query):
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
    response = model.invoke(message).content.strip()

    def strip_query(query):
        return query.strip().strip('`').strip('"').strip("'").replace('sql', '').replace('SQL', '').strip()
    
    response = strip_query(response)

    count = 0
    while not is_valid_sql(response):
        print(response)
        count += 1
        if count > 3:
            return None
        
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

    return response

def is_valid_sql(query):
    if not query or not isinstance(query, str):
        return False

    cleaned_text = query.upper().strip()
    blacklisted_sql_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
    allowed_table_names = ["EMPLOYEES", "CLIENTS", "PROJECTS", 
                           "EMPLOYEE_PROJECT_ASSIGNMENTS", "SKILLS", "EMPLOYEE_SKILLS", "DEPARTMENT", "PROJECT", "EMPLOYEE"]
    # allowed_table_names = ["EMPLOYEE", "DEPARTMENT", "PROJECT"]
    
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

    extracted_tables = {table[1].upper() for table in from_tables + join_tables}
    if not extracted_tables.issubset(set(allowed_table_names)):
        print("References disallowed tables")
        return False
    
    return True

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

def query_data(user_query):
    sql_query = generate_query(user_query)
    if not sql_query:
        return "❌ Unable to generate a valid SQL query after multiple attempts."
    
    cur.execute(sql_query)
    rows = cur.fetchall()
    if not rows:
        return "🔍 No results found."
    
    cols = [desc[0] for desc in cur.description]
    lines = [" | ".join(cols)]
    lines += [" | ".join(map(str, row)) for row in rows]
    table = "\n".join(lines)
    tables = format_table(table)
    texts = list()
    for table in tables:
        texts.append("```" + table + "```")
    
    return texts