import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import psycopg2
from psycopg2 import OperationalError
import datetime

from memory_storage import memory_storage

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

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
    
    history = memory_storage.search_by_time(channel_id, start_time, end_time)
    if not history:
        return "No messages found in the specified time range."
    
    formatted_history = _format_memory_history(history)
    if not formatted_history:
        return "No valid messages found in the specified time range."
    
    return summarize_conversation(formatted_history)
    
def _format_memory_history(history):
    """
    Format the memory history for summarization.
    Converts the history into a list of tuples with (role, message, timestamp).
    """
    formatted_history = []
    for entry in history:
        role = entry.get('sender', 'Unknown')
        message = entry.get('content', '')
        timestamp = entry.get('timestamp', datetime.datetime.now())
        bot_message = entry.get('bot_message', '')

        formatted_history.append((role, message, timestamp))
        if bot_message:
            formatted_history.append(('bot', bot_message, timestamp))
        
    return formatted_history