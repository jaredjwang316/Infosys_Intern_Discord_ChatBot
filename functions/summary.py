import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import psycopg2
from psycopg2 import OperationalError
import datetime

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

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

def summarize_conversation(history):
    # Accepts history as list of (role, message, timestamp) or (role, message, timestamp, ...)
    conversation_text = ""
    for entry in history:
        if len(entry) >= 3:
            role, message, timestamp = entry[:3]
            time_str = timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
            if str(role).lower() == "bot":
                conversation_text += f"AI [{time_str}]: {message}\n"
            else:
                conversation_text += f"User [{time_str}]: {message}\n"
        elif len(entry) == 2:
            role, message = entry
            if str(role).lower() == "bot":
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

# def summarize_conversation_by_time(channel_id, start_time, end_time=datetime.datetime.now().utcnow()):
#     """
#     Summarize conversation history for a specific channel within a time range.
#     """
#     query = """
#         SELECT sender, content, timestamp
#         FROM messages
#         WHERE channel_id = %s AND timestamp >= %s AND timestamp <= %s
#         ORDER BY timestamp ASC;
#     """
#     cur.execute(query, (channel_id, start_time, end_time))
#     history = cur.fetchall()
    
#     if not history:
#         return "No conversation history found for the specified time range."
    
#     summary = summarize_conversation(history)
#     return summary