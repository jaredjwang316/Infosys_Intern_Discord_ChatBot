import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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


def get_event_details(event_details):

    prompt = f"""
    The following input (labeled as 'Event Details') contains information about an event to be scheduled.
    From this sentence, identify and return the following event details in the syntax of a Python dictionary with
    the keys/values as follows:

    - key: 'title', value: the name of the meeting as a string with proper capitalization
        - If a proper title is not provided, generate an appropriate one based on the information provided.
            If no information is provided, use 'Meeting' as a default.
    - key: start_dt, value: start date and time of the meeting in the format of a datetime object
    - key: end_dt, value: end date and time of the meeting in the format of a datetime object

    Ignore any tagged users in the message.

    Only return the completed Python dictionary. Do not return any other text, including any indications of assent.

    ### EVENT DETAILS ###
    {event_details}
    """

    try:
        response = model.invoke(prompt)
        response = eval(response.strip())
        return response

    except Exception as e:
        print(f"Error generating event details: {e}")
        



