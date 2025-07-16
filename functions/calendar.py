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
from email.mime.text import MIMEText
import base64
import datetime
import time
import json

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

SCOPES = ['https://www.googleapis.com/auth/calendar.events', 'https://www.googleapis.com/auth/gmail.send']


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
        
        # Remove code blocks if present
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("python"):
                raw = raw[6:].strip()
        
        # Handle cases where model returns JSON format instead of Python dict
        if raw.startswith('{') and raw.endswith('}'):
            # Try to parse as JSON first, then convert to dict
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Fall back to ast.literal_eval for Python dict format
                parsed = ast.literal_eval(raw)
        else:
            # Handle other formats like "python" prefix
            if raw.startswith('python'):
                raw = raw[6:].strip()
            parsed = ast.literal_eval(raw)

        local_tz = get_localzone()

        # Convert string timestamps to datetime objects first
        parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
        parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])
        
        if parsed['end_dt'] <= parsed['start_dt']:
            print("Warning: End time is before or equal to start time. Setting end time to 1 hour after start time.")
            parsed['end_dt'] = parsed['start_dt'] + datetime.timedelta(hours=1)

        parsed['start_dt'] = parsed['start_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)
        parsed['end_dt'] = parsed['end_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}") 
        return None

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

def create_gcal_event(event_dict, user_id):
    service = get_calendar_service()

    event = {
        'summary': event_dict['title'],
        'start': {'dateTime': event_dict['start_dt'].isoformat()},
        'end': {'dateTime': event_dict['end_dt'].isoformat()},
        'conferenceData': {
            'createRequest': {
                'requestId': f"{user_id}-{event_dict['title']}",
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        }
    }

    created_event = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1).execute()
    event_id = created_event['id']

    return created_event.get('hangoutLink'), event_id

def edit_event_details(event_details, event_list):

    prompt = f"""
    You are a helpful assistant that updates calendar events based on natural language instructions.

    Given:
    - A message requesting a change
    - A list of existing events (formatted as key-value pairs)

    Your task:
    1. Select the event that most closely matches the request. If the message is vague, choose the most recent event in the list.
    2. Identify which details need to change based on the message (e.g., title, start time, end time).
    3. Return a new version of the selected dictionary with the the changes applied only to the key-value pair that needed to be modified.
    4. Use ISO 8601 format for the values mapped to 'start_dt' and 'end_dt'.
    5. Keep the formatting and datatypes for the values mapped to 'title' the same as the original dictionary.
    6. DO NOT CHANGE THE VALUES MAPPED TO 'gcal_event_id' OR 'gcal_link'.

    ### EXISTING EVENTS ###
    {event_list}

    ### MESSAGE ###
    {event_details}

    Return ONLY the modified dictionary and nothing else.
    MAKE SURE THE LENGTH OF THE MEETING (end time - start time) STAYS THE SAME UNLESS EXPLICITLY CHANGED.
    Do not include any code blocks, extra formatting, or explanation.
    """
    try:
        print("Calling Gemini model...")
        response = model.invoke([HumanMessage(content=prompt)])
        print("Gemini raw output:", response.content)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        
        parsed = ast.literal_eval(raw)

        local_tz = get_localzone()

        parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
        parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])

        parsed['start_dt'] = parsed['start_dt'].astimezone(local_tz)
        parsed['end_dt'] = parsed['end_dt'].astimezone(local_tz)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}")
        return None

def edit_gcal_event(updated_event_dict):
    try:
        service = get_calendar_service()

        updated_event = {
            'summary': updated_event_dict['title'],
            'start': {'dateTime': updated_event_dict['start_dt'].isoformat()},
            'end': {'dateTime': updated_event_dict['end_dt'].isoformat()}
        }

        result = service.events().patch(calendarId='primary', eventId=updated_event_dict['gcal_event_id'], body=updated_event).execute()
        print(f"✅ Event '{result['summary']}' updated.")
        return result.get('htmlLink')
    
    except Exception as e:
        print(f"❌ Error editing Google Calendar event: {e}")
        return None

def delete_event_details(event_details, event_list):

    prompt = f"""
    You are a helpful assistant that selects calendar events based on natural language instructions.

    Given:
    - A message requesting the deletion of an event
    - A list of existing events (formatted as key-value pairs)

    Your task:
    1. Select the event that most closely matches the request. If the message is vague, choose the most recent event in the list.
    2. Return the dictionary of the selected event with the following changes:
        - Convert the values mapped to 'start_dt' and 'end_dt' to ISO 8601 format.
        - Keep the formatting and datatypes for the values mapped to 'title' the same as the original dictionary.

    ### EXISTING EVENTS ###
    {event_list}

    ### MESSAGE ###
    {event_details}

    Do not include any code blocks, extra formatting, or explanation.
    """
    try:
        print("Calling Gemini model...")
        response = model.invoke([HumanMessage(content=prompt)])
        print("Gemini raw output:", response.content)

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
        
        parsed = ast.literal_eval(raw)

        local_tz = get_localzone()

        parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
        parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])

        parsed['start_dt'] = parsed['start_dt'].astimezone(local_tz)
        parsed['end_dt'] = parsed['end_dt'].astimezone(local_tz)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}")

def delete_gcal_event(event_id):
    try:
        service = get_calendar_service()
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        print(f"🗑️ Google Calendar event {event_id} deleted successfully.")
        return True
    except Exception as e:
        print(f"❌ Error deleting Google Calendar event: {e}")
        return False

def cal_handler(event_details):
    prompt = f"""
    You are a helpful assistant that evaluates user messages in natural language.
    Your job is to determine whether the user wants to CREATE a new event or to EDIT an existing one.

    Return your answer as the following:

    - If the user wants to CREATE an event, return 'CREATE'
    - If the user wants to EDIT an existing event, return 'EDIT'
    - If the user wants to DELETE an existing event, return 'DELETE'

    Return ONLY 'True' or 'False' in the given formatting (with the given capitalization).
    Do not include quotes or punctuation of any kind. Do not include any extra formatting or explanation.

    ### USER MESSAGE ###
    {event_details}
    """

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        return decision

    except Exception as e:
        print(f"Error in confirming calendar tool action: {e}")