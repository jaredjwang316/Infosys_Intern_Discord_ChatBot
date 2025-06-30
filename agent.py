import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema import Document
from langgraph.prebuilt import ToolNode, tools_condition
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

    return total_result if total_result else quick_result

llm_with_tools = llm.bind_tools(
    [
        query,
        summarize,
        summarize_by_time,
        search
    ]
)

class State(TypedDict):
    """
    State for the agent graph.
    """
    messages: list[str]
    current_user = str
    current_channel: str

def conductor(state: State) -> dict:
    """
    Conductor function to manage the state of the agent graph.
    This function is responsible for invoking the LLM with the current state and updating the messages.
    Args:
        state (State): The current state of the conversation.
    Returns:
        dict: Updated state with the new messages.
    """
    if not state["messages"]:
        state["messages"] = []
    local_memory.add_message(state["current_channel"], state["current_user"], state["messages"][-1])

    full_prompt = f"""
    You are an intelligent assistant that decides how to respond to user queries. 
    Given the latest user message and the conversation context, determine whether you can answer directly or if you need to use a tool (such as querying a database, searching conversation history, or summarizing previous messages).
    If the user asks for information retrieval, data lookup, or summary, consider using the appropriate tool.
    If the query is general, conversational, or does not require external data, respond directly.
    Always explain your reasoning briefly before taking action.

    Current conversation context:

    """

    for role, msg, _ in local_memory.get_chat_history(state["current_channel"]):
        full_prompt += f"{role}: {msg}\n"

    msg = HumanMessage(
        content=full_prompt
    )

    response = llm.invoke([msg])

    return {
        "messages": state["messages"].append(response.content.strip()),
        "current_user": state["current_user"],
        "current_channel": state["current_channel"]
        }

def router(state: State) -> str:
    """
    Router function to determine the next action based on the current state of the conversation.
    
    Args:
        state (State): The current state of the conversation.
    Returns:
        str: The next action to take, which can be a tool name or a direct response.
    """
    
    if not state["messages"]:
        return "conductor"
    
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"
    else:
        return "generate_response"

def generate_response(state: State) -> str:
    """
    Generate a response based on the current state of the conversation.

    Args:
        state (State): The current state of the conversation.
    Returns:
        str: The generated response from the LLM.
    """

    full_prompt = f"""
    You have gathered all the necessary information from available tools and the previous conversation context. 
    Now, compose a clear and concise final response to the user, directly addressing their query using the information you have found. 
    Do not mention the use of tools or internal processes—just provide the answer in a helpful and friendly manner.

    Current conversation context:

    """

    full_prompt += "\n".join(state["messages"]) + "\n"
    full_prompt += "\n\nYour Response:\n"
    
    messages = [
        HumanMessage(content=full_prompt)
    ]
    
    response = llm.invoke(messages)
    local_memory.add_message(state["current_channel"], 'Bot', response.content.strip())

    return response.content.strip()

tools = ToolNode(
    name="tools",
    tools=[
        query,
        summarize,
        summarize_by_time,
        search
    ]
)

builder = StateGraph(State)

builder.add_node("conductor", conductor)
# builder.add_node("router", router)
builder.add_node("tools", tools)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "conductor")
# builder.add_edge("conductor", "router")
builder.add_conditional_edges(
    source="conductor",
    path=router,
    path_map={
        "conductor": "conductor",
        "tools": "tools",
        "generate_response": "generate_response",
    }
)
builder.add_edge("tools", "conductor")
builder.add_edge("generate_response", END)