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
    """
    Summarize a conversation history.
    This function takes a list of conversation entries and generates a summary using the Google Generative AI model.
    Each entry in the history should be a tuple of (role, message, timestamp) or (role, message).
    The role can be "user" or "bot", and the timestamp should be a datetime object or a string.
    """
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

def summarize_conversation_by_time(channel_id, start_time, end_time=datetime.datetime.now()):
    """
    Summarize conversation history for a specific channel within a time range.
    Handles both old format (role only) and new format (role, timestamp, channel_id) metadata.
    """
    # Query for messages with new format (has timestamp and channel_id)
    new_format_query = """
        SELECT document, cmetadata, 
               (cmetadata->>'timestamp')::timestamp as parsed_timestamp
        FROM langchain_pg_embedding 
        WHERE cmetadata->>'channel_id' = %s 
        AND cmetadata->>'timestamp' IS NOT NULL
        AND (cmetadata->>'timestamp')::timestamp >= %s 
        AND (cmetadata->>'timestamp')::timestamp <= %s
        ORDER BY (cmetadata->>'timestamp')::timestamp ASC;
    """
    
    try:
        formatted_history = []
        
        # Get new format messages (with timestamp and channel filtering)
        cur.execute(new_format_query, (str(channel_id), start_time, end_time))
        new_messages = cur.fetchall()
        
        print(f"🔍 Found {len(new_messages)} messages matching all criteria")
        
        # If no messages in time range, return none found message
        if not new_messages:
            print("No conversation history found for this channel in the specified time range.")
            return "No conversation history found for this channel in the specified time range."
        
        for document, metadata, timestamp in new_messages:
            role = metadata.get('role', 'Unknown')
            
            formatted_history.append((role, document, timestamp))
        
        # Sort all messages by timestamp
        formatted_history.sort(key=lambda x: x[2])
        
        if not formatted_history:
            return "No conversation history found for this channel."
        
        # Pass the formatted history to the existing summarize_conversation function
        print(f"📊 Summarizing {len(formatted_history)} messages...")
        summary = summarize_conversation(formatted_history)
        return summary
        
    except Exception as e:
        print(f"❌ Error querying conversation history: {e}")
        return f"Error retrieving conversation history: {str(e)}"