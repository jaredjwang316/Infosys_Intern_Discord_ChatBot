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



# Determine if a query asks for visualization
def is_visualization_query(user_query):
    keywords = ["visualize", "chart", "plot", "bar chart", "pie chart", "line chart", "trend"]
    return any(k in user_query.lower() for k in keywords)

# Extract preferred chart type
def extract_chart_type(user_query):
    if "pie" in user_query.lower():
        return "pie"
    elif "line" in user_query.lower() or "trend" in user_query.lower():
        return "line"
    elif "bar" in user_query.lower():
        return "bar"
    else:
        return "bar"  # default

def generate_chart_file(rows, columns, chart_type="bar"):
    import matplotlib.pyplot as plt
    import io

    if len(columns) < 2 or not rows:
        return None

    try:
        x_vals = [str(row[0]) for row in rows]
        y_vals = [float(row[1]) for row in rows]
    except (ValueError, IndexError):
        return None
    
    # Dynamically adjust figure width based on number of x-values
    fig_width = max(10, len(x_vals) * 0.4)  # Scale up for large x_vals
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type == "bar":
        ax.bar(x_vals, y_vals)
    elif chart_type == "line":
        ax.plot(x_vals, y_vals, marker="o")
    elif chart_type == "pie":
        ax.pie(y_vals, labels=x_vals, autopct='%1.1f%%')
    else:
        return None

    ax.set_title(f"{columns[1]} over {columns[0]}")
    if chart_type != "pie":
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])

        # Rotate labels and align to the right
        ax.tick_params(axis='x', labelrotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

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

def strip_query(query):
    # Remove common code fences and leading/trailing whitespace, but not quotes inside
    query = query.strip()
    # Remove markdown code block fences if present (```sql or ``` etc.)
    if query.startswith("```") and query.endswith("```"):
        query = "\n".join(query.split("\n")[1:-1]).strip()
    # Remove single line backticks (`)
    query = query.strip('`').strip()
    # Remove any leading 'sql' or 'SQL' on a separate line
    query = re.sub(r"^(sql|SQL)\s*", "", query, flags=re.IGNORECASE)
    return query

def generate_query(sql_query):
    # change prompt to not be a hypothetical if correct. Validating will be the next step.

    db_schema = SCHEMA_TEXT
    global tips
    tips = "PostgreSQL does not support strftime(). Use TO_CHAR(date_column, 'YYYY-MM') instead to format dates."
    if tips:
        query_tips = f"\n### TIPS ###\n{tips}\n"
    else:
        query_tips = ""

    prompt_template = f"""
    You are an expert at querying databases. Your task is to generate a SQL query based on the user's request.

    ### INSTRUCTION ###
    Given the database schema below, generate a SQL query that fulfills the user's request.
    - Ensure the SQL query is syntactically correct.
    - Use appropriate table and column names from the schema.
    - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
    - DO NOT SHOW ID COLUMNS UNLESS SPECIFICALLY REQUESTED.
    - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.
    
    ### DATABASE SCHEMA ###
    {db_schema}
    {query_tips}
    ### USER REQUEST ###
    {sql_query}

    ### SQL QUERY ###
    """

    message = [
        HumanMessage(content=prompt_template)
    ]
    response = model.invoke(message).content.strip()
    response = strip_query(response)

    print(response)

    count = 0
    while not is_valid_sql(response):
        count += 1
        if count > 3:
            return None
        
        reprompt_template = f"""
        The SQL query you provided is not valid. Please generate a correct SQL query based on the user's request and the database schema.
        
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - DO NOT SHOW ID COLUMNS UNLESS SPECIFICALLY REQUESTED.
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.

        ### DATABASE SCHEMA ###
        {db_schema}
        {query_tips}
        ### USER REQUEST ###
        {sql_query}

        ### PREVIOUS SQL QUERY ###
        {response}

        ### NEW SQL QUERY ###
        """
        response = model.invoke(reprompt_template).content.strip()
        response = strip_query(response)

        print(response)


    return response

def retry_query(sql_query, information=None):
    global tips
    if tips:
        query_tips = f"\n### TIPS ###\n{tips}\n"
    else:
        query_tips = ""

    if not information:
        information = "No additional information found yet."
    
    count = 0
    
    rows = None
    while not rows and count < 3:
        reprompt_template = f"""
        The SQL query you provided did not return any results. This may be due to an error in the query or a lack of matching data in the database. Please generate a correct SQL query based on the user's request and the database schema to find more information about the data to help you refine your query.
        
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that retrieves more information about the data.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.

        ### DATABASE SCHEMA ###
        {SCHEMA_TEXT}
        {query_tips}
        ### FOUND INFORMATION ###
        {information}

        ### USER REQUEST ###
        {sql_query}

        ### PREVIOUS SQL QUERY ###
        {sql_query}

        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Find more information about the data to help you refine your query.

        ### NEW SQL QUERY ###
        """
        response = model.invoke(reprompt_template).content.strip()
        response = strip_query(response)

        print(response)


        while not is_valid_sql(response):
            reprompt_template = f"""
            The SQL query you provided is not valid. Please generate a correct SQL query based on the user's request and the database schema to find more information about the data to help you refine your query.
            
            ### INSTRUCTION ###
            Given the database schema below, generate a SQL query that retrieves more information about the data.
            - Ensure the SQL query is syntactically correct.
            - Use appropriate table and column names from the schema.
            - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
            - DO NOT SHOW ID COLUMNS UNLESS SPECIFICALLY REQUESTED.
            - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.

            ### DATABASE SCHEMA ###
            {SCHEMA_TEXT}
            {query_tips}

            ### FOUND INFORMATION ###
            {information}

            ### USER REQUEST ###
            {sql_query}

            ### PREVIOUS SQL QUERY ###
            {response}

            Remember to:
            - Ensure the SQL query is syntactically correct.
            - Find more information about the data to help you refine your query.

            ### NEW SQL QUERY ###
            """
            response = model.invoke(reprompt_template).content.strip()
            response = strip_query(response)

            print(response)


        cur.execute(response)
        rows = cur.fetchall()

        if not rows:
            information = "No results found."
        cols = [desc[0] for desc in cur.description]
        lines = [" | ".join(cols)]
        lines += [" | ".join(map(str, row)) for row in rows]
        table = "\n".join(lines)
        information = format_table(table)

        retry_template = f"""
        You previously generated a SQL query that did not return any results. Based on the new information retrieved, please refine your SQL query to ensure it fulfills the user's request and retrieves relevant data.
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.

        ### DATABASE SCHEMA ###
        {SCHEMA_TEXT}
        {query_tips}
        ### FOUND INFORMATION ###
        {information}

        ### PREVIOUS SQL QUERY ###
        {response}

        ### USER REQUEST ###
        {sql_query}

        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Find more information about the data to help you refine your query.

        ### NEW SQL QUERY ###
        """
        response = model.invoke(retry_template).content.strip()
        response = strip_query(response)

        print(response)


        while not is_valid_sql(response):
            reprompt_template = f"""
            The SQL query you provided is not valid. Please generate a correct SQL query that fulfills the user's request.

            ### INSTRUCTION ###
            Given the database schema below, generate a SQL query that retrieves more information about the data.
            - Ensure the SQL query is syntactically correct.
            - Use appropriate table and column names from the schema.
            - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
            - If asked for a certain type of email (.com, .org, etc.), search the end of the email adress.

            ### DATABASE SCHEMA ###
            {SCHEMA_TEXT}
            {query_tips}

            ### FOUND INFORMATION ###
            {information}

            ### PREVIOUS SQL QUERY ###
            {response}

            ### USER REQUEST ###
            {sql_query}

            Remember to:
            - Ensure the SQL query is syntactically correct.
            - Fulfill the user's request.

            ### NEW SQL QUERY ###
            """
            response = model.invoke(reprompt_template).content.strip()
            response = strip_query(response)

            print(response)

        
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

    from_tables = [m[1] for m in re.findall(from_pattern, cleaned_text)]
    join_tables = [m[1] for m in re.findall(join_pattern, cleaned_text)]

    for tbl in from_tables + join_tables:
        if tbl.upper() not in allowed_table_names:
            print(f"Table '{tbl}' not in schema")
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

def query_data(user_id, user_query, session_history=None):
    # Create a contextual prompt using the session history (previous queries in the session)
    contextualized_query = user_query

    if session_history != None and len(session_history) > 1:
        # Take all previous queries except the current one
        previous_queries = session_history[:-1]
        
        # Manually build the context block line by line
        context_lines = []
        count = 1
        for q in previous_queries:
            context_lines.append(f"{count}. {q}")
            count += 1

        context_block = ""
        for line in context_lines:
            context_block += line + "\n"

        # Add context into the prompt
        contextualized_query = (
            "Here is the context of this conversation session:\n"
            + context_block +
            "\nNow answer the follow-up question:\n"
            + user_query
        )


    sql_query = generate_query(contextualized_query)
    if not sql_query:
        return ["❌ Unable to generate a valid SQL query after multiple attempts."]
    
    cur.execute(sql_query)
    rows = cur.fetchall()
    if not rows:
        tables = retry_query(sql_query)
        if not tables:
            return ["❌ No results found for the query. Please refine your request or try a different query."]
    else:
        # Save short-term memory
        user_chat_history[user_id] = {
            "last_user_query": user_query,
            "last_sql_query": sql_query
        }

        cols = [desc[0] for desc in cur.description]
        lines = [" | ".join(cols)]
        lines += [" | ".join(map(str, row)) for row in rows]
        table = "\n".join(lines)

        if is_visualization_query(user_query):
            chart_type = extract_chart_type(user_query)
            chart_file = generate_chart_file(rows, cols, chart_type)
            if chart_file:
                return [{"type": "image", "file": chart_file, "filename": "chart.png"}]
        tables = format_table(table)
    texts = list()
    for table in tables:
        texts.append("```" + table + "```")
    
    return texts