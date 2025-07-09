"""
query.py

Purpose:
--------
This module handles natural language query interpretation, SQL query generation, validation, execution, 
and optional visualization for a PostgreSQL database used in an AI-powered chatbot system.

It bridges the gap between plain English user input and executable, secure SQL queries by leveraging 
Google's Gemini model through LangChain, with database schema awareness and retry logic.

Key Responsibilities:
---------------------
- Interprets natural language user queries and generates corresponding SQL queries using Gemini.
- Validates SQL queries against the loaded database schema to prevent unauthorized operations.
- Executes SQL queries against a live PostgreSQL database and formats the results for Discord-compatible output.
- Generates charts (bar, pie, or line) when the user request implies visualization.
- Handles retries with context if queries return no results or is invalid.
"""
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import psycopg2
from psycopg2 import OperationalError



# Determine if a query asks for visualization
def is_visualization_query(user_query):
    """
    Determine if the user's query requests a visualization/chart.

    Args:
        user_query (str): The raw user query in natural language.

    Returns:
        bool: True if the query contains visualization-related keywords; False otherwise.

    Keywords checked include "visualize", "chart", "plot", "bar chart", "pie chart", "line chart", "trend".
    """
    keywords = ["visualize", "chart", "plot", "bar chart", "pie chart", "line chart", "trend"]
    return any(k in user_query.lower() for k in keywords)

# Extract preferred chart type
def extract_chart_type(user_query):
    """
    Extract the preferred chart type from the user's query.

    Args:
        user_query (str): The raw user query.

    Returns:
        str: One of "pie", "line", or "bar" representing the chart type.
             Defaults to "bar" if no specific type is found.
    """
    if "pie" in user_query.lower():
        return "pie"
    elif "line" in user_query.lower() or "trend" in user_query.lower():
        return "line"
    elif "bar" in user_query.lower():
        return "bar"
    else:
        return "bar"  # default

def generate_chart_title(user_query, columns):
    """
    Generate a human-readable and descriptive chart title using the user query and database columns.

    Args:
        user_query (str): The user's original query requesting the chart.
        columns (List[str]): A list of two column names [X-axis, Y-axis] from the query result.

    Returns:
        str: A natural language chart title suitable for dashboards or reports.

    Notes:
        - Avoids raw column names and technical jargon.
        - Ensures the title is understandable by non-technical users.
        - Uses the Gemini model to generate the title based on instructions.
    """
    prompt = f"""
    Generate a clear, human-readable chart title based on the user's request and the two database columns 
    used for the chart.

    ### INSTRUCTIONS ###
    - Do NOT use raw column names like 'count', 'id', 'role', 'hire_month' as-is.
    - Replace technical terms and abbreviations with descriptive, natural phrases.
    - Avoid generic phrases like "count over role", "total over type", or "value by category".
    - DO NOT use words like "count", "total", "id", "data", or "chart" in the title unless absolutely necessary.
    - DO NOT repeat column names exactly as they appear.
    - The title must make sense to someone with no knowledge of SQL or databases.
    - Make it sound like a real chart you'd see in a report or dashboard.
    - Use proper capitalization and spacing.


    ### USER QUERY ###
    {user_query}

    ### COLUMNS ###
    X-axis: {columns[0]}
    Y-axis: {columns[1]}

    ### TITLE ###
    Only return the generated chart title as a single line of text.
    """
    message = [HumanMessage(content=prompt)]
    title = model.invoke(message).content.strip()
    return title

def generate_chart_file(rows, columns, chart_type="bar", user_query=None):
    """
    Generate a chart image (PNG) from query results using matplotlib.

    Args:
        rows (List[Tuple]): Query result rows with at least two columns.
        columns (List[str]): Corresponding column names for X and Y axes.
        chart_type (str): The type of chart to generate ("bar", "line", or "pie").
        user_query (str, optional): The original user query for context (used to generate the chart title).

    Returns:
        io.BytesIO or None: A bytes buffer containing the PNG image data of the chart, or None if generation fails.

    Behavior:
        - Supports bar, line, and pie charts.
        - Automatically formats axis labels and titles.
        - Returns None if input data is insufficient or invalid.
    """
    import matplotlib.pyplot as plt
    import io

    if len(columns) < 2 or not rows:
        return None

    try:
        x_vals = [str(row[0]) for row in rows]
        y_vals = [float(row[1]) for row in rows]
    except (ValueError, IndexError):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type == "bar":
        ax.bar(x_vals, y_vals)
    elif chart_type == "line":
        ax.plot(x_vals, y_vals, marker="o")
    elif chart_type == "pie":
        ax.pie(y_vals, labels=x_vals, autopct='%1.1f%%')
    else:
        return None

    title = generate_chart_title(user_query, columns)
    ax.set_title(title)
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
    """
    Clean SQL query text by removing common formatting artifacts.

    Args:
        query (str): The raw SQL query string possibly containing markdown code fences or prefixes.

    Returns:
        str: The cleaned SQL query suitable for execution.
    
    Actions performed:
        - Strips leading/trailing whitespace.
        - Removes triple backtick code fences (```...```) if present.
        - Removes single backticks (`).
        - Removes leading 'sql' or 'SQL' tokens.
    """
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
    """
    Generate a validated SQL SELECT query from a natural language request using Gemini.

    Args:
        sql_query (str): The natural language user query or request.

    Returns:
        str or None: A syntactically valid SQL SELECT query that matches the user's intent,
                     or None if a valid query cannot be generated after retries.

    Process:
        - Uses a detailed prompt including the database schema to guide Gemini in SQL generation.
        - Validates the query syntax and allowed table names.
        - Retries up to 3 times if the query is invalid.
        - Ensures the query only selects data (no modification commands).
    """

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
    """
    Retry and refine a SQL query if it returns no results, leveraging additional information.

    Args:
        sql_query (str): The last generated SQL query that returned empty results.
        information (str, optional): Additional context or feedback from previous attempts.

    Returns:
        List[str] or None: Formatted query result strings after refinement, or None if no data found.

    Behavior:
        - Invokes Gemini with instructions to generate a better query to retrieve relevant data.
        - Validates SQL syntax for each generated query.
        - Executes queries against the database.
        - Retries up to 3 times before giving up.
        - Returns formatted text tables or None if no results.
    """
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

        
        cur.execute(response)   #executes the query against the live database.
        rows = cur.fetchall()   #retrieves all the rows returned by the executed query.
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
    """
    Check if a SQL query is syntactically valid and safe to execute.

    Args:
        query (str): The SQL query string to validate.

    Returns:
        bool: True if the query is a safe SELECT statement using only allowed tables; False otherwise.

    Validation Rules:
        - Must start with SELECT.
        - Must contain FROM clause.
        - Must not contain blacklisted keywords like INSERT, UPDATE, DELETE, DROP, ALTER, CREATE.
        - Table names in FROM and JOIN clauses must be in the allowed schema.
    """
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
    """
    Split a large textual table into chunks to avoid message length limits.

    Args:
        table (str): A string representing tabular data (headers and rows separated by newlines).

    Returns:
        List[str]: A list of smaller string chunks each under the maximum allowed length (e.g., 2000 chars).

    Purpose:
        - Facilitates sending large results over Discord or similar platforms without truncation.
    """
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
    """
    Main entry point to process a user's natural language query into data response(s).

    Args:
        user_id (str): Unique identifier for the user (used for session memory).
        user_query (str): The natural language query string from the user.
        session_history (List[str], optional): Prior queries in this session to provide conversational context.

    Returns:
        List[str] or List[Dict]: A list of formatted text responses, or
                                a list containing an image dict (type, file, filename) for charts.

    Workflow:
        - Incorporates session context into the prompt.
        - Generates and validates SQL query using Gemini.
        - Executes the query on PostgreSQL.
        - If no results, retries with refined queries.
        - Detects if the query requests visualization and generates charts accordingly.
        - Formats and returns query results as markdown code blocks or images.
        - Stores last user and SQL queries for short-term memory.
    """

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

        if is_visualization_query(user_query):
            chart_type = extract_chart_type(user_query)
            chart_file = generate_chart_file(rows, cols, chart_type, user_query=user_query)
            if chart_file:
                return [{"type": "image", "file": chart_file, "filename": "chart.png"}]

        lines = [" | ".join(cols)]
        lines += [" | ".join(map(str, row)) for row in rows]
        table = "\n".join(lines)
        tables = format_table(table)

    texts = list()
    for table in tables:
        texts.append("```" + table + "```")
    
    return texts