import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

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