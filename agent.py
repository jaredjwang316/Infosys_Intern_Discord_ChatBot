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

class State(TypedDict):
    """
    State for the agent graph.
    """
    messages: Annotated[list, add_messages]
    current_user: str
    current_channel: str
    task_description: str

class Agent:
    """
    Agent class to handle the conversation and tool interactions.
    """

    def __init__(self, role_name='default_role', allowed_tools=['query', 'summarize', 'summarize_by_time', 'search']):
        self.role_name = role_name
        self.allowed_tools = allowed_tools
        self.tool_functions = [getattr(self, tool_name) for tool_name in allowed_tools if tool_name in allowed_tools]

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.llm_with_tools = self.llm.bind_tools(self.tool_functions)
        self.tools = ToolNode(
            name="tools",
            tools=self.tool_functions,
        )

        self.graph = None
        self.build_graph()

    def invoke(self, state: State, config: dict) -> dict:
        """
        Invoke the agent with the current state.

        Args:
            state (State): The current state of the conversation.
            config (dict): Configuration options for the invocation.
        Returns:
            dict: A dictionary containing the response messages from the agent.
        """
        if self.graph is None:
            self.build_graph()

        return self.graph.invoke(state, config)

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

    def conductor(self, state: State) -> dict:
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

        previous_messages = memory_storage.local_memory.get_chat_history(state["current_channel"])
        formatted_previous_messages = ""
        for message in previous_messages[:-1]:
            user = message[0]
            content = message[1]
            formatted_previous_messages += f"{user}: {content}\n"

        all_descriptions = {
            'query': '- query: For querying the SQL database with user-specific queries.\n',
            'summarize': '- summarize: For summarizing the entire conversation history of a channel.\n',
            'summarize_by_time': '- summarize_by_time: For summarizing conversation history within a specific time range.\n',
            'search': '- search: For searching the conversation history for specific information.\n',
        }
        descriptions = [all_descriptions[t] for t in self.allowed_tools]

        all_when_to_use = {
            'summarize_by_time': '- "summarize conversation history for last X days/hours" → Use summarize_by_time tool\n',
            'search': '- "search for something" or asking about something from the conversation → Use search tool\n',
            'query': '- "query database" or specific data requests → Use query tool\n',
            'summarize': '- "general summary" → Use summarize tool\n',
        }
        when_to_use = [all_when_to_use[t] for t in self.allowed_tools]

        # 2) build system prompt & history
        system_prompt = f"""
        You are an intelligent assistant with access to tools and never hallucinates.
        You must decide when to use tools based on the user's request and the conversation history.

        You have access to the following tools:
        {'\n'.join(descriptions)}
                                    
        If the user's single request implies more than one tool operation, you should generate ALL of the corresponding tool calls in one go, in the order they should run, without asking the user to choose.

        IMPORTANT: Only use tools when the user explicitly requests information that requires them.
        
        Current channel ID: {state["current_channel"]}
        Current user: {state["current_user"]}
        
        WHEN TO USE TOOLS:
        {'\n'.join(when_to_use)}
        
        WHEN NOT TO USE TOOLS:
        - Greetings like "hello", "good afternoon", "hi"
        - General conversation or questions unrelated to the conversation history or database
        - Simple responses that don't require data lookup
        
        For simple greetings and conversation, respond directly without using tools.
        
        If this requires database/search/summary operations, use the appropriate tool.

        ABOUT TASK DESCRIPTIONS:
        When creating a task description, be extremely specific and include ALL relevant details:
        - BAD: "Search for information based on previous queries"
        - GOOD: "Search for information about Python error handling that was discussed yesterday"
        
        Your task description must be self-contained with all necessary context because:
        1. The response generator will only see THIS task description, not the full conversation history
        2. Previous queries/messages are not automatically accessible
        3. All relevant details from user's current and previous messages must be included
        
        For example, if a user says "Can you find what John said about databases?", your task should be:
        "TASK: Search for messages from user John about databases in the conversation history"

        IMPORTANT INSTRUCTIONS:
        1. If you decide to use tools, first provide a clear task description that explains what the user is asking for, including any context from previous messages.
        2. The task description should be comprehensive enough that someone reading it would understand exactly what needs to be answered.
        3. Format your response as:
        
        "TASK: [Clear description of what the user wants, with full context of previous responses]"
        
        Then make your tool calls.

        Feel free to ask for clarification if the user's request is ambiguous.
        DO NOT HALLUCINATE OR MAKE UP INFORMATION. If you don't know the answer, say so.
        
        PREVIOUS MESSAGES:
        {formatted_previous_messages}

        CURRENT USER MESSAGE:
        """

        message = system_prompt + "\n" + state["messages"][-1].content

        system_message = HumanMessage(content=message)

        response = self.llm_with_tools.invoke([system_message])

        print(f"\n##### CONDUCTOR RESPONSE ##### \n\n{response}")

        task_description = ""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            content = response.content or ""
            if "TASK:" in content:
                task_start = content.find("TASK:") + len("TASK:")
                task_end = content.find("\n", task_start)
                if task_end == -1:
                    task_end = len(content)
                task_description = content[task_start:task_end].strip()
            else:
                task_description = f"User asked: {state['messages'][-1].content.strip()}"
        return {
            "messages": [response],
            "current_user": state["current_user"],
            "current_channel": state["current_channel"],
            "task_description": task_description
        }
    
    def generate_response(self, state: State) -> dict:
        """
        Generate a response based on the current state of the conversation.

        Args:
            state (State): The current state of the conversation.
        Returns:
            dict: A dictionary containing the response messages from the agent.
        """
        original_user_query = None
        tools_results = []

        for msg in state["messages"]:
            if hasattr(msg, "content") and not hasattr(msg, "tool_call_id") and not hasattr(msg, "tool_calls"):
                if not original_user_query:
                    original_user_query = msg
            elif hasattr(msg, "tool_call_id"):
                tools_results.append(msg)

        tool_results_text = ""
        if tools_results:
            tool_results_text = "\n".join([
                f"Tool '{msg.name}' results: {msg.content}" for msg in tools_results
            ])

        task_to_complete = state.get("task_description", "") or (original_user_query.content if original_user_query else "Unknown query")
        
        response_prompt = f"""
        You are an intelligent assistant. Based on the tool results below, provide a comprehensive and helpful answer to complete the specified task.

        TASK TO COMPLETE: {task_to_complete}

        Original User Message: "{original_user_query.content if original_user_query else 'Unknown query'}"

        Tool Results:
        {tool_results_text}

        Instructions:
        1. Focus on completing the task as described above using the information from the tool results.
        2. Provide a complete, helpful answer based on the tool results.
        3. Do not hallucinate or make up information not provided in the tool results.
        4. Be concise but thorough in your response.
        5. Do NOT request additional tools unless the current results are completely insufficient.

        Current channel ID: {state["current_channel"]}
        Current user: {state["current_user"]}

        Provide a final answer based on the tool results above or call additional tools only if absolutely necessary.
        """

        human_message = HumanMessage(content=response_prompt)

        response = self.llm_with_tools.invoke([human_message])

        print(f"\n##### RESPONSE ##### \n\n{response}")

        return {
            "messages": [response],
            "current_user": state["current_user"],
            "current_channel": state["current_channel"],
            "task_description": state.get("task_description", "")
        }

    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("conductor", self.conductor)
        builder.add_node("tools", self.tools)
        builder.add_node("generate_response", self.generate_response)
        builder.set_entry_point("conductor")

        builder.add_conditional_edges(
            "conductor", 
            tools_condition,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        builder.add_edge("tools", "generate_response")
        builder.add_conditional_edges(
            "generate_response",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END
            }
        )

        chat_memory = memory_storage.get_memory_saver()
        self.graph = builder.compile(checkpointer=chat_memory)
