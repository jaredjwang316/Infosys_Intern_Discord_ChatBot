import matplotlib.pyplot as plt
import io
import base64
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import psycopg2
from psycopg2 import OperationalError
#from query import *
from prompts import *

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

try:
    with open("./database/schema.txt", "r") as f:
        SCHEMA_TEXT = f.read()
except FileNotFoundError:
    print("Schema file not found. Please ensure the schema.txt file exists in the database directory.")
    raise

try:
    with open("./database/Schema_test.sql", "r") as f:
        raw_schema = f.read()
except FileNotFoundError:
    print("Schema test file not found. Please ensure the Schema_test.sql file exists in the database directory.")
    raise

table_names = re.findall(
   r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([A-Za-z0-9_]+)`?",
   raw_schema,
   flags = re.IGNORECASE
)

allowed_tables = {name.upper() for name in table_names}

# DB config
PG_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT", 5432),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname":   os.getenv("DB_NAME"),
}

print("Connecting to Postgres...")

try:
    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = True
    cur  = conn.cursor()
except OperationalError as e:
    print("Could not connect to Postgres:", e)
    raise

print("Connected to Postgres successfully!")

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

tips = str()
db_schema = SCHEMA_TEXT

class Query:
    def __init__(self, user_id, user_query, db_schema, session_history=None):
        self.user_query = user_query
        self.user_id = user_id
        self.session_history = session_history
        self.sql_query = ""
    
    def generate_query(self):
        global tips
        tips = "PostgreSQL does not support strftime(). Use TO_CHAR(date_column, 'YYYY-MM') instead to format dates."
        if tips:
            query_tips = f"\n### TIPS ###\n{tips}\n"
        else:
            query_tips = ""

        prompt_template = prompt_instructions + f"""
        ### DATABASE SCHEMA ###
        {db_schema}
        {query_tips}
        ### USER REQUEST ###
        {self.user_query}

        ### SQL QUERY ###
        """

        message = [
            HumanMessage(content=prompt_template)
        ]
        response = model.invoke(message).content.strip()
        self.sql_query = self.strip_query(response)

        print(self.sql_query)

        count = 0
        while not is_valid_sql(self.sql_query):
            count += 1
            if count > 3:
                return None
            
            self.retry_invalid(query_tips)

        return response
    
    def retry_invalid(self, query_tips, information=None):
        reprompt_template = not_valid_instructions + f"""

        ### DATABASE SCHEMA ###
        {db_schema}
        {query_tips}
        ### USER REQUEST ###
        {self.user_query}

        ### PREVIOUS SQL QUERY ###
        {self.sql_query}

        ### NEW SQL QUERY ###
        """
        if information != None:
            reprompt_template += f"""
            ### FOUND INFORMATION ###
            {information}"""
        response = model.invoke(reprompt_template).content.strip()
        self.sql_query = strip_query(response)

    def build_context(self):
    # Take all previous queries except the current one
        previous_queries = self.session_history[:-1]
    
    # Manually build the context block line by line
        context_lines = []
        count = 1
        for q in previous_queries:
            context_lines.append(f"{count}. {q}")
            count += 1

        context_block = ""
        for line in context_lines:
            context_block += line + "\n"
        
        return context_block

    
    def query_data(self):
        global tips
        tips = "PostgreSQL does not support strftime(). Use TO_CHAR(date_column, 'YYYY-MM') instead to format dates."
        if tips:
            query_tips = f"\n### TIPS ###\n{tips}\n"
        else:
            query_tips = ""

        if self.session_history != None and len(self.session_history) > 1:
            context_block = self.build_context()

            # Add context into the prompt
            contextualized_query = (
                "Here is the context of this conversation session:\n"
                + context_block +
                "\nNow answer the follow-up question:\n"
                + self.user_query
            )

        sql_query = self.generate_query()
        if not sql_query:
            return ["❌ Unable to generate a valid SQL query after multiple attempts."]
        
        cur.execute(sql_query)
        rows = cur.fetchall()
        if not rows:
            tables = retry_query(query_tips)
            if not tables:
                return ["❌ No results found for the query. Please refine your request or try a different query."]
        else:
            # Save short-term memory
            user_chat_history[self.id] = {
                "last_user_query": self.query,
                "last_sql_query": sql_query
            }

            cols = [desc[0] for desc in cur.description]
            lines = [" | ".join(cols)]
            lines += [" | ".join(map(str, row)) for row in rows]
            table = "\n".join(lines)

            if is_visualization_query(self.query):
                chart_type = extract_chart_type(self.query)
                chart_file = generate_chart_file(rows, cols, chart_type)
                if chart_file:
                    return [{"type": "image", "file": chart_file, "filename": "chart.png"}]
            tables = format_table(table)
        texts = list()
        for table in tables:
            texts.append("```" + table + "```")
        
        return texts
    
    def retry_no_result(self, query_tips, information=None):

        if not information:
            information = "No additional information found yet."

        count = 0
        
        rows = None
        while not rows and count < 3:
            reprompt_template = no_result_instructions + f"""

            ### DATABASE SCHEMA ###
            {db_schema}
            {query_tips}
            ### FOUND INFORMATION ###
            {information}

            ### USER REQUEST ###
            {self.user_query}

            ### PREVIOUS SQL QUERY ###
            {self.sql_query}

            ### NEW SQL QUERY ###
            """
            if information == None:
                reprompt_template += "Based on the new information retrieved, please refine your SQL query to ensure it fulfills the user's request and retrieves relevant data."
            response = model.invoke(reprompt_template).content.strip()
            self.sql_query = strip_query(response)

    def retry_query(self, query_tips, information=None):
        
        self.retry_no_result(query_tips)

        while not is_valid_sql(self.sql_query):
            self.retry_invalid(query_tips)

            cur.execute(self.sql_query)
            rows = cur.fetchall()

            if not rows:
                information = "No results found."
            cols = [desc[0] for desc in cur.description]
            lines = [" | ".join(cols)]
            lines += [" | ".join(map(str, row)) for row in rows]
            table = "\n".join(lines)
            information = format_table(table)

            self.retry_no_result(query_tips, information)

            while not is_valid_sql(response):
                self.retry_invalid(query_tips, information)

            
            cur.execute(response)
            rows = cur.fetchall()
            count += 1

        if not rows:
            return None
        cols = [desc[0] for desc in cur.description]
        lines = [" | ".join(cols)]
        lines += [" | ".join(map(str, row)) for row in rows]
        table = "\n".join(lines)
        tables = format_table(table)

        return tables