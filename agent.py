import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import datetime
import concurrent.futures

from functions.query import query_data
from functions.summary import summarize_conversation, summarize_conversation_by_time
from functions.search import search_conversation, search_conversation_quick
from local_memory import LocalMemory

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

local_memory = LocalMemory()

class State(TypedDict):
    """
    State for the agent graph.
    """
    graph_state: str
    current_user = str
    current_channel: str

def conductor(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def query(user_id: str, user_query: str) -> list[str]:
    """
    Query the SQL database with the user's query.

    Args:
        user_query (str): The user's query to be processed.
        user_id (str): The ID of the user making the query.
    Returns:
        list[str]: A list of messages containing the query result.
    """
    
    return query_data(user_id, user_query, local_memory.get_user_query_session_history(user_id))

def summarize(channel_id: str) -> str:
    """
    Summarize the conversation history for a given channel.

    Args:
        channel_id (str): The ID of the channel to summarize.
    Returns:
        str: A list of messages containing the summary.
    """
    
    return summarize_conversation(local_memory.get_chat_history(channel_id))

def summarize_by_time(channel_id: str, rollback_time: int, time_unit: str) -> str:
    """
    Summarize the conversation history for a given channel within a time range.

    Args:
        channel_id (str): The ID of the channel to summarize.
        rollback_time (int): The amount of time to roll back.
        time_unit (str): The unit of time for the rollback (e.g., 'days', 'hours').
    Returns:
        str: A list of messages containing the summary.
    """

    now = datetime.datetime.now()
    delta_args = {f"{time_unit}s": rollback_time}
    since = now - datetime.timedelta(**delta_args)
    return summarize_conversation_by_time(local_memory.get_chat_history(channel_id), since, now)

def search(channel_id: str, query: str) -> str:
    """
    Search the conversation history for a given channel.

    Args:
        channel_id (str): The ID of the channel to search.
        query (str): The search query.
    Returns:
        str: A list of messages containing the search results.
    """
    
    quick_result = search_conversation_quick(local_memory.get_vectorstore(channel_id), query)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(search_conversation, query, local_memory.get_cached_history_documents(channel_id), quick_result)

        total_result = None
        try:
            total_result = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            print("Long search operation timed out, using only quick response instead.")
        except Exception as e:
            print(f"Error during search operation: {e}")

    local_memory.clear_cached_history(channel_id)

    if total_result:
        local_memory.add_message(channel_id, "Bot", total_result if total_result else quick_result)

    return total_result if total_result else quick_result

# TODO: Actually look at the search, and implement the rest of the methods and then make the agent runnable with nodes and conditional edges.