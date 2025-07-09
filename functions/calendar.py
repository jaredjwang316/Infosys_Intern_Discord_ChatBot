import os
import pickle
import ast
from tzlocal import get_localzone
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
#from google.ouath2 import service_account
#from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import datetime

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

SCOPES = ['https://www.googleapis.com/auth/calendar.events']



def get_event_details(event_details):

    prompt = f"""
You are a helpful assistant extracting event details from natural language.

From the following text, return ONLY a Python dictionary in this exact format:
{{
  "title": "Team Sync",
  "start_dt": "2025-07-09T14:00:00",
  "end_dt": "2025-07-09T14:30:00"
}}

Use ISO 8601 strings for the times. Do not include code fences, markdown formatting, or any other text.

### EVENT DETAILS ###
{event_details}
"""

    try:
        print("Calling Gemini model...")
        response = model.invoke([HumanMessage(content=prompt)])
        print("Gemini raw output:", response.content)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        
        parsed = ast.literal_eval(raw)

        # message = [HumanMessage(content=prompt)]
        # response = model.invoke(message)
        # event_dict = ast.literal_eval(response.content.strip())

        local_tz = get_localzone()

        parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
        parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])

        parsed['start_dt'] = parsed['start_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)
        parsed['end_dt'] = parsed['end_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}") 

def get_calendar_service():
    creds = None
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=8)
        
        with open('token.pkl', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('calendar', 'v3', credentials=creds)
    return service

def create_gcal_event(event_dict):
    service = get_calendar_service()

    event = {
        'summary': event_dict['title'],
        'start': {'dateTime': event_dict['start_dt'].isoformat()},
        'end': {'dateTime': event_dict['end_dt'].isoformat()}
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return created_event.get('htmlLink')


# def create_gcal_event(event_dict):
#     SERVICE_ACCOUNT_FILE = 'credentials.json'

#     credentials = service_account.Credentials.from_service_account_file(
#         SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
#     service = build('calendar', 'v3', credentials=credentials)
    
#     event = {
#         'name': event_dict['title'],
#         'start': {'dateTime': event_dict['start_dt'].isoformat()},
#         'end': {'dateTime': event_dict['end_dt'].isoformat()}
#     }

#     created_event = service.events().insert(calendarId='primary', body=event).execute()
#     return created_event.get('htmlLink')