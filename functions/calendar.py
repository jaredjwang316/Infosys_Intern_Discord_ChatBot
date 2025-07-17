import os
import pickle
import ast
from tzlocal import get_localzone
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import datetime
import pytz
import dateparser
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

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly', 'https://www.googleapis.com/auth/calendar.freebusy', 'https://www.googleapis.com/auth/calendar.events', 'https://www.googleapis.com/auth/gmail.send']

def get_google_service(api_name, version):
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
    
    service = build(api_name, version, credentials=creds)
    return service, creds

def cal_handler(event_details):
    prompt = f"""
    You are a helpful assistant that evaluates user messages in natural language.
    Your job is to determine whether the user wants to CREATE a new event or to EDIT an existing one.

    Return your answer as the following:

    - If the user wants to CREATE an event, return 'CREATE'
    - If the user wants to EDIT an existing event, return 'EDIT'
    - If the user wants to DELETE an existing event, return 'DELETE'

    Return ONLY 'Create', 'Edit', or 'Delete' in the given formatting (with the given capitalization).
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

def get_event_details(event_details):
        tz = get_localzone()
        now = datetime.datetime.now(tz).isoformat()

        prompt = f"""
        You are a helpful assistant extracting event details from natural language.

        From the following text, return ONLY a Python dictionary in this exact format:
        {{
            "title": "Team Sync",
            "start_dt": "2025-07-09T14:00:00",
            "end_dt": "2025-07-09T15:00:00",
            "duration": 60,
            "search_start": "2025-07-09T08:00:00",
            "search_end": "2025-07-12T17:00:00"
        }}

        Rules:
        - All dates/times must be at least 1 hour AFTER the given 'Current Date/Time'. Use the 'Current Date/Time' when parsing relational phrases like 'today', 'tomorrow', 'next week', etc.
        - Use ISO 8601 format for 'start_dt', 'end_dt', 'search_start', and 'search_end' values.
        - The value for 'duration' should be the duration of the meeting in minutes as an integer.
        - If 'start_dt' and 'end_dt' are both missing, enter 'None' as a string for those fields and try to extract a search range from the message:
            - Look for date/time phrases indicating a search window (e.g. "after July 19th", "between July 9th and July 12th").
            - Populate 'search_start' and 'search_end' accordingly. If you find only one boundary, fill that one and leave the other as 'None' as a string.
            - Choose a date/time at least 1 hour AFTER the given 'Current Date/Time'.
        - If 'start_dt' is given and 'end_dt' is missing:
            - Look for a duration and calculate 'end_dt' by adding it to 'start_dt'.
            - If no duration is found, default 'duration' to 60 and set 'end_dt' as 'start_dt' + 60 minutes.
        - If both 'start_dt' and 'end_dt' are given, use them to calculate 'duration'.
        - If no duration info can be inferred, default 'duration' to 60 minutes.
        - Leave fields empty with the string 'None' if the information cannot be determined.

        Respond ONLY with the Python dictionary described above. Do not add any explanatory text, markdown, or formatting.

        ### EVENT DETAILS ###
        {event_details}

        ### CURRENT DATE/TIME ###
        {now}
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
            if parsed['search_start'] != 'None':
                parsed['search_start'] = datetime.datetime.fromisoformat(parsed['search_start'])
                parsed['search_start'] = parsed['search_start'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)

            if parsed['search_end'] != 'None':
                parsed['search_end'] = datetime.datetime.fromisoformat(parsed['search_end'])
                parsed['search_end'] = parsed['search_end'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)

            if parsed['start_dt'] != 'None' and parsed['end_dt'] != 'None':
                parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
                parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])

                parsed['start_dt'] = parsed['start_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)
                parsed['end_dt'] = parsed['end_dt'].replace(tzinfo=local_tz).astimezone(datetime.timezone.utc)

            return parsed

        except Exception as e:
            print(f"Error generating event details: {e}")

def edit_event_details(event_details, event_list):
    tz = get_localzone()
    now = datetime.datetime.now(tz).isoformat()

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

    Rules:
    - All dates/times must be at least 1 hour AFTER the given 'Current Date/Time'. Use the 'Current Date/Time' when parsing relational phrases like 'today', 'tomorrow', 'next week', etc.
    - Use ISO 8601 format for 'start_dt', 'end_dt', 'search_start', and 'search_end' values.
    - The value for 'duration' should be the duration of the meeting in minutes as an integer.
    - If 'start_dt' and 'end_dt' are both missing, enter 'None' as a string for those fields and try to extract a search range from the message:
        - Look for date/time phrases indicating a search window (e.g. "after July 19th", "between July 9th and July 12th").
        - Populate 'search_start' and 'search_end' accordingly. If you find only one boundary, fill that one and leave the other as 'None' as a string.
        - Choose a date/time at least 1 hour AFTER the given 'Current Date/Time'.
    - If 'start_dt' is given and 'end_dt' is missing:
        - Look for a duration and calculate 'end_dt' by adding it to 'start_dt'.
        - If no duration is found, default 'duration' to 60 and set 'end_dt' as 'start_dt' + 60 minutes.
    - If both 'start_dt' and 'end_dt' are given, use them to calculate 'duration'.
    - If no duration info can be inferred, default 'duration' to 60 minutes.
    - Leave fields empty with the string 'None' if the information cannot be determined.

    ### EXISTING EVENTS ###
    {event_list}

    ### MESSAGE ###
    {event_details}

    ### CURRENT DATE/TIME ###
    {now}

    NEW EVENT DETAILS:
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

        # Make parsed datetimes timezone-aware if they're naive
        if parsed['start_dt'].tzinfo is None:
            parsed['start_dt'] = parsed['start_dt'].replace(tzinfo=local_tz)
        if parsed['end_dt'].tzinfo is None:
            parsed['end_dt'] = parsed['end_dt'].replace(tzinfo=local_tz)
        
        # Get current time with SAME TIMEZONE for comparison
        now = datetime.datetime.now(local_tz)

        # Check if start time is in the past - BOTH NOW TIMEZONE-AWARE
        if parsed['start_dt'] < now:
            print("Warning: Start time is in the past. Adjusting to future time.")
            # If editing to a time in the past, move it to tomorrow at the same time
            tomorrow = now.replace(hour=parsed['start_dt'].hour, minute=parsed['start_dt'].minute, second=0, microsecond=0) + datetime.timedelta(days=1)
            duration = parsed['end_dt'] - parsed['start_dt']
            parsed['start_dt'] = tomorrow
            parsed['end_dt'] = tomorrow + duration
            
        if parsed['end_dt'] <= parsed['start_dt']:
            print("Warning: End time is before or equal to start time. Setting end time to 1 hour after start time.")
            parsed['end_dt'] = parsed['start_dt'] + datetime.timedelta(hours=1)

        # Convert to UTC for Discord API - THIS WAS MISSING!
        parsed['start_dt'] = parsed['start_dt'].astimezone(datetime.timezone.utc)
        parsed['end_dt'] = parsed['end_dt'].astimezone(datetime.timezone.utc)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}")
        return None

def delete_event_details(event_details, event_list):
    tz = get_localzone()
    now = datetime.datetime.now(tz).isoformat()

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
    
    Rules:
    - All dates/times must be at least 1 hour AFTER the given 'Current Date/Time'. Use the 'Current Date/Time' when parsing relational phrases like 'today', 'tomorrow', 'next week', etc.
    - Use ISO 8601 format for 'start_dt', 'end_dt', 'search_start', and 'search_end' values.
    - The value for 'duration' should be the duration of the meeting in minutes as an integer.
    - If 'start_dt' and 'end_dt' are both missing, enter 'None' as a string for those fields and try to extract a search range from the message:
        - Look for date/time phrases indicating a search window (e.g. "after July 19th", "between July 9th and July 12th").
        - Populate 'search_start' and 'search_end' accordingly. If you find only one boundary, fill that one and leave the other as 'None' as a string.
        - Choose a date/time at least 1 hour AFTER the given 'Current Date/Time'.
    - If 'start_dt' is given and 'end_dt' is missing:
        - Look for a duration and calculate 'end_dt' by adding it to 'start_dt'.
        - If no duration is found, default 'duration' to 60 and set 'end_dt' as 'start_dt' + 60 minutes.
    - If both 'start_dt' and 'end_dt' are given, use them to calculate 'duration'.
    - If no duration info can be inferred, default 'duration' to 60 minutes.
    - Leave fields empty with the string 'None' if the information cannot be determined.

    ### EXISTING EVENTS ###
    {event_list}

    ### MESSAGE ###
    {event_details}

    ### CURRENT DATE/TIME ###
    {now}

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

        # Convert string timestamps to datetime objects
        parsed['start_dt'] = datetime.datetime.fromisoformat(parsed['start_dt'])
        parsed['end_dt'] = datetime.datetime.fromisoformat(parsed['end_dt'])

        # Make timezone-aware if needed
        if parsed['start_dt'].tzinfo is None:
            parsed['start_dt'] = parsed['start_dt'].replace(tzinfo=local_tz)
        if parsed['end_dt'].tzinfo is None:
            parsed['end_dt'] = parsed['end_dt'].replace(tzinfo=local_tz)

        # Convert to UTC for Discord API consistency
        parsed['start_dt'] = parsed['start_dt'].astimezone(datetime.timezone.utc)
        parsed['end_dt'] = parsed['end_dt'].astimezone(datetime.timezone.utc)

        return parsed

    except Exception as e:
        print(f"Error generating event details: {e}")
        return None

def create_gcal_event(event_dict, guests):
    service, creds = get_google_service('calendar', 'v3')

    event = {
        'summary': event_dict['title'],
        'start': {'dateTime': event_dict['start_dt'].isoformat()},
        'end': {'dateTime': event_dict['end_dt'].isoformat()},
        'attendees': guests,
        'conferenceData': {
            'createRequest': {
                'requestId': f"{event_dict['title']} - {event_dict['discord_id']}",
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        }
    }

    created_event = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1, sendUpdates='all').execute()
    event_id = created_event['id']

    return created_event.get('hangoutLink'), event_id

def edit_gcal_event(updated_event_dict, event_id):
    try:
        service, creds = get_google_service('calendar', 'v3')

        updated_event = {
            'summary': updated_event_dict['title'],
            'start': {'dateTime': updated_event_dict['start_dt'].isoformat()},
            'end': {'dateTime': updated_event_dict['end_dt'].isoformat()}
        }

        result = service.events().patch(calendarId='primary', eventId=event_id, body=updated_event, sendUpdates='all').execute()
        print(f"✅ Event '{result['summary']}' updated.")
        return result.get('htmlLink')
    
    except Exception as e:
        print(f"❌ Error editing Google Calendar event: {e}")
        return None

def delete_gcal_event(event_id):
    try:
        service, creds = get_google_service('calendar', 'v3')
        service.events().delete(calendarId='primary', eventId=event_id, sendUpdates='all').execute()
        print(f"🗑️ Google Calendar event {event_id} deleted successfully.")
        return True
    except Exception as e:
        print(f"❌ Error deleting Google Calendar event: {e}")
        return False

def find_earliest_slot(attendees, search_start, search_end, duration):
    service, creds = get_google_service('calendar', 'v3')
    
    attendee_emails = []
    for user in attendees:
        attendee_emails.append(user['email'])
    
    now = datetime.datetime.now(pytz.utc)

    if search_start == 'None':
        slot_start = now + datetime.timedelta(hours=1)
    else:
        slot_start = search_start

    if search_end == 'None':
        if search_start == 'None':
            time_max = now + datetime.timedelta(days=3)
        elif search_start != 'None':
            time_max = search_start + datetime.timedelta(days=3)
    else:
        time_max = search_end

    body = {
        'timeMin': now.isoformat(),
        'timeMax': time_max.isoformat(),
        'timeZone': 'UTC',
        'items': [{'id': email} for email in attendees]
    }

    freebusy = service.freebusy().query(body=body).execute()['calendars']
    busy = [slot for cal in freebusy.values() for slot in cal.get('busy', [])]
    busy.sort(key=lambda s: s['start'])

    work_start = 8
    work_end = 17
    local_tz = pytz.timezone("America/Chicago")

    while slot_start + datetime.timedelta(minutes=duration) < time_max:
        local_slot = slot_start.astimezone(local_tz)
        local_hour = local_slot.hour

        if local_hour < work_start or (local_hour + duration) / 60 > work_end:
            next_day = (local_slot + datetime.timedelta(days=1)).replace(
                hour=work_start, minute=0, second=0, microsecond=0
            )
            slot_start = next_day.astimezone(pytz.utc)
            continue

        slot_end = slot_start + datetime.timedelta(minutes=duration)

        if not any(slot_start < datetime.fromisoformat(b['end']) and slot_end > datetime.fromisoformat(b['start']) for b in busy):
            return slot_start, slot_end, now
        
        slot_start += datetime.timedelta(minutes=15)
    
    return None, None, None

def get_upcoming_events():
    service, creds = get_google_service('calendar', 'v3')
    local_tz = get_localzone()
    now = datetime.datetime.now(local_tz)
    end = now + datetime.timedelta(days=7)
    events = service.events().list(
        calendarId='primary',
        timeMin=now.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    return events.get('items', [])