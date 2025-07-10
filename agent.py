"""
agent.py

This module defines the agentic behavior of the Discord bot using LangGraph and LangChain.
It enables the bot to understand user intent, decide autonomously whether tools are needed,
execute those tools, and generate a final coherent response.

Key Features:
- Uses LangGraph to manage agent flow and decision-making.
- Integrates custom tools (e.g., query, summarize, search) into the LLM loop.
- Dynamically plans and invokes tool calls based on user requests.
- Maintains stateful interaction using LocalMemory and conversation context.
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.schema import Document
from langgraph.prebuilt import ToolNode, tools_condition
import logging
import datetime
import concurrent.futures

from functions.query import query_data
from functions.summary import summarize_conversation, summarize_conversation_by_time
from functions.search import search_conversation, search_conversation_quick
from memory_storage import memory_storage

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Generate a session-specific log filename
timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
log_filename = os.path.join("logs", f"agent_session_{timestamp}.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # File
        logging.StreamHandler()  # Terminal (optional)
    ]
)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

@tool
def query(user_id: str, user_query: str) -> list[str]:
    """
    Query the SQL database with the user's query.

    Args:
        user_query (str): The user's query to be processed.
        user_id (str): The ID of the user making the query.
    Returns:
        list[str]: A list of messages containing the query result.
    """
    print("QUERYING")
    
    user_id = int(user_id)
    return query_data(user_id, user_query, memory_storage.local_memory.get_user_query_session_history(user_id))

@tool
def summarize(channel_id: str) -> str:
    """
    Summarize the conversation history for a given channel.

    Args:
        channel_id (str): The ID of the channel to summarize.
    Returns:
        str: A list of messages containing the summary.
    """
    print("SUMMARIZING")

    channel_id = int(channel_id)
    result = summarize_conversation(memory_storage.local_memory.get_chat_history(channel_id))

    return result

@tool
def summarize_by_time(channel_id: str, rollback_time: float, time_unit: str) -> str:
    """
    Summarize the conversation history for a given channel within a time range.

    Args:
        channel_id (str): The ID of the channel to summarize.
        rollback_time (int): The amount of time to roll back.
        time_unit (str): The unit of time for the rollback (e.g., 'days', 'hours').
    Returns:
        str: A list of messages containing the summary.
    """
    print("SUMMARIZING BY TIME")

    channel_id = int(channel_id)
    # rollback_time = int(rollback_time)

    memory_storage.store_all_in_long_term_memory()

    now = datetime.datetime.now()
    delta_args = {f"{time_unit}": rollback_time}
    since = now - datetime.timedelta(**delta_args)
    result = summarize_conversation_by_time(channel_id, since, now)
    print(f"🔍 Summarize by time result: {result}")

    return result

@tool
def search(channel_id: str, query: str) -> str:
    """
    Search the conversation history for a given channel.

    Args:
        channel_id (str): The ID of the channel to search.
        query (str): The search query.
    Returns:
        str: A list of messages containing the search results.
    """
    print("SEARCHING")

    channel_id = int(channel_id)

    quick_result = search_conversation_quick(memory_storage.get_local_vectorstore(channel_id), query)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(search_conversation, channel_id, query, quick_result)

        total_result = None
        try:
            total_result = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            print("Long search operation timed out, using only quick response instead.")
        except Exception as e:
            print(f"Error during search operation: {e}")

        memory_storage.local_memory.clear_cached_history(channel_id)

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
    Defines the conversation state for the agent.

    Attributes:
        messages (list): List of chat messages (user, bot, tool).
        current_user (str): ID of the current user.
        current_channel (str): ID of the current channel.
    """
    messages: Annotated[list, add_messages]
    current_user: str
    current_channel: str

def conductor(state: State) -> dict:
    """
    Main agent loop to decide actions, use tools, and respond.

    Args:
        state (State): Current conversation state.
    
    Returns:
        dict: A dictionary with the updated message list.
    """
    logging.info(f"🧭 Agent started — User: {state['current_user']}, Channel: {state['current_channel']}")
    logging.info(f"🧾 Current messages: {[m.content for m in state['messages'] if hasattr(m, 'content')]}")

    # 1) bootstrap memory
    if not state["messages"]:
        state["messages"] = []
    last = state["messages"][-1]
    if hasattr(last, "content") and not hasattr(last, "tool_call_id"):
        memory_storage.add_message(
            state["current_channel"],
            state["current_user"],
            last.content
        )

    # 2) build system prompt & history
    system_prompt = SystemMessage(content=f"""
    You are an intelligent assistant with access to tools and never hallucinates.
    You must decide when to use tools based on the user's request and the conversation history.

    You have access to the following tools:
    - query: For querying the SQL database with user-specific queries.
    - summarize: For summarizing the entire conversation history of a channel.
    - summarize_by_time: For summarizing conversation history within a specific time range.
    - search: For searching the conversation history for specific information.
                                  
    If the user's single request implies more than one tool operation, you should generate ALL of the corresponding tool calls in one go, in the order they should run, without asking the user to choose.
 
    Format your plan as a JSON array under `tool_calls`, e.g.:
    [
        {{ "name": "summarize", "args": {{ "channel_id": "{state['current_channel']}" }} }},
        {{ "name": "search",    "args": {{ "channel_id": "{state['current_channel']}", "query": "UI" }} }}
    ]
    
    IMPORTANT: Only use tools when the user explicitly requests information that requires them.
    
    Current channel ID: {state["current_channel"]}
    Current user: {state["current_user"]}
    
    WHEN TO USE TOOLS:
    - "summarize conversation history for last X days/hours" → Use summarize_by_time tool
    - "search for something" or asking about something from the conversation → Use search tool  
    - "query database" or specific data requests → Use query tool
    - "general summary" → Use summarize tool
    
    WHEN NOT TO USE TOOLS:
    - Greetings like "hello", "good afternoon", "hi"
    - General conversation or questions unrelated to the conversation history or database
    - Simple responses that don't require data lookup
    
    For simple greetings and conversation, respond directly without using tools.
    Keep in mind, tool outputs will not be shown to the user directly. You must interpret the results and provide a clear, helpful response.
    When asked for summaries, only use information given by the tools.
    
    If this is a simple greeting or conversation, respond directly. 
    If this requires database/search/summary operations, use the appropriate tool.

    Feel free to ask for clarification if the user's request is ambiguous.
    DO NOT HALLUCINATE OR MAKE UP INFORMATION. If you don't know the answer, say so.
    
    IMPORTANT: If you have already called a tool and received results, provide a final answer to the user based on those results. Do NOT call the same tool again.
    """)
    messages = [system_prompt] + state["messages"]

    logging.info("🧠 Invoking LLM to generate plan with tools (if needed)...")
    plan = llm_with_tools.invoke(messages)

    logging.info(f"💬 LLM plan generated: {plan.content}")
    if plan.tool_calls:
        logging.info(f"📦 Tool calls planned: {[call['name'] for call in plan.tool_calls]}")
    else:
        logging.info("📦 No tool calls planned. Final response will be generated directly.")
        return {"messages": [plan]}

    # 3) Execute each requested tool, but append results as AIMessage
    tool_map = {t.name: t for t in (query, summarize, summarize_by_time, search)}
    for call in plan.tool_calls:
        name = call["name"]
        args = call.get("args", {})
        if name not in tool_map:
            raise ValueError(f"Unknown tool: {name}")

        # Run the tool
        result = tool_map[name](args)

        # Log the tool usage
        logging.info(
            f"""🛠️ Tool Called: {name}
            ⏰ Time: {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")}
            👤 User: {state['current_user']}
            💬 Channel: {state['current_channel']}
            🧾 Args: {args}
            """
        )

        # Append tool output as AI message
        messages.append(AIMessage(content=f"[{name} output]:\n{result}"))

    logging.info("🧠 Invoking LLM to synthesize final response from tool outputs...")
    # 4) Final synthesis—only System/Human/AI messages here
    final = llm.invoke(
        messages + [
            HumanMessage(
                content="Please combine the above results into one final clear answer for the user."
            )
        ]
    )
    logging.info("✅ Final response generated successfully.")

    return {"messages": [final]}

def router(state: State) -> str:
    """
    Router function to determine the next action based on the current state of the conversation.
    
    Args:
        state (State): The current state of the conversation.
    Returns:
        str: The next action to take, which can be a tool name or a direct response.
    """

    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"
    else:
        return "generate_response"

def generate_response(state: State) -> dict:
    """
    Synthesizes a final response from tool outputs and conversation history.

    Args:
        state (State): Current state.
    
    Returns:
        dict: A dictionary with the final response message.
    """
    user_query = ""
    tool_results = ""
    conversation_history = ""
    
#     for msg in state["messages"]:
#         if hasattr(msg, '__class__'):
#             if msg.__class__.__name__ == 'HumanMessage' and hasattr(msg, 'content'):
#                 user_query = msg.content
#                 conversation_history += f"User: {msg.content}\n"
#             elif msg.__class__.__name__ == 'AIMessage' and hasattr(msg, 'content'):
#                 if msg.content and "Tool Calls:" not in msg.content:
#                     conversation_history += f"Assistant: {msg.content}\n"
#             elif msg.__class__.__name__ == 'ToolMessage' and hasattr(msg, 'content'):
#                 tool_results += f"{msg.content}\n"

#     # Create a clean, simple prompt
#     final_prompt = f"""
#     User's original question: {user_query}
    
#     Tool results: {tool_results}
    
#     Previous conversation:
#     {conversation_history}
    
#     Please provide a clear, helpful final response to the user's question using the tool results:
#     """

#     try:
#         response = llm.invoke([HumanMessage(content=final_prompt)])
#         memory_storage.add_message(state["current_channel"], 'Bot', response.content.strip())
#         return {"messages": [response]}
#     except Exception as e:
#         print(f"❌ ERROR: {e}")
#         fallback_response = AIMessage(
#             content="I apologize, but I'm experiencing technical difficulties. Please try your request again."
#         )
#         return {"messages": [fallback_response]}

# ── Graph Construction ────────────────────────────────────────────────
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
builder.add_node("tools", tools)
# builder.add_node("generate_response", generate_response)

builder.set_entry_point("conductor")
builder.add_conditional_edges(
    "conductor", 
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
builder.add_edge("tools", "conductor")
# builder.add_edge("generate_response", END)

chat_memory, thread_id = memory_storage.local_memory.get_chat_memory()

agent_graph = builder.compile(checkpointer=chat_memory)

# TODO Add logging using python logging library
# TODO Add better local memory management (basically first test to see if memory is going through properly)
# TODO Add error handling for tool calls and responses for the logging
# TODO When logging, make a class to filter out sensitive information (api keys, passwords, etc.)
# TODO Calendar is now low priority, do it last. First focus on making data visualization and search work well, make a better summary tool that works 
# with the return from search and query tools.
