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

# TODO: Add in gcal integration to allow the bot to create events in the user's calendar
# TODO: Add in a max queries and timeout system to prevent the bot trying indefinitely
# TODO: Add in a way to switch llms or to try again on queries that fail due to rate limits or other issues

import os
from typing import Annotated
from typing_extensions import TypedDict, Any
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
    images: list[tuple[str, bytes]] | None
    loop_count: int
    guild: Any

class Agent:
    """
    Agent class to handle the conversation and tool interactions.
    """

    def __init__(self, role_name='default_role', allowed_tools=['query', 'summarize', 'summarize_by_time', 'search']):
        self.role_name = role_name
        self.allowed_tools = allowed_tools
        self.tool_functions = [getattr(self, tool_name) for tool_name in allowed_tools if tool_name in allowed_tools]

        self._pending_images = []

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

        self.max_iterations = 3

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
    def query(user_query: str) -> str:
        """
        Query the SQL database with the user's query.

        Args:
            user_query (str): The user's query to be processed.
            user_id (str): The ID of the user making the query.
        Returns:
            list[str]: A list of messages containing the query result.
        """
        print("QUERYING")

        result = query_data(user_query)
        
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            if result.get("file"):
                filename = result.get("filename", "chart.png")
                file_data = result.get("file")

                if hasattr(file_data, 'getvalue'):
                    file_bytes = file_data.getvalue()
                else:
                    file_bytes = file_data

                if not hasattr(Agent, '_current_instance'):
                    Agent._current_instance = None
                
                if Agent._current_instance:
                    Agent._current_instance._pending_images.append((filename, file_bytes))

                return "Query returned a diagram or table representing the data and is ready to be displayed."
            else:
                return "Query failed to generate a diagram or table."

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
        Agent._current_instance = self
        self._pending_images = []

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
            'query': '- query: For querying the SQL database with natural language queries. No changes are allowed to be made to the database (no INSERT, DROP, etc.). It is also able to visualize results in a chart or diagram.\n',
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
        You are an intelligent assistant's planner with access to tools and never hallucinates.
        Decide when to use tools based on user requests.

        Available tools:
        {'\n'.join(descriptions)}
                        
        If one request needs multiple tools, generate ALL tool calls at once in the correct order.

        Current channel: {state["current_channel"]} | User: {state["current_user"]}
        
        WHEN TO USE TOOLS:
        {'\n'.join(when_to_use)}
        
        WHEN NOT TO USE TOOLS:
        - Greetings ("hello", "hi", etc.)
        - General conversation unrelated to conversation history/database
        - Simple responses not requiring data lookup
        
        TASK DESCRIPTIONS:
        Before making tool calls, you MUST plan and create a comprehensive task description:

        1. ANALYZE the user's request and identify what specific information they need
        2. CONSIDER any relevant context from previous messages
        3. PLAN what tools you'll need and in what order
        4. CREATE a detailed task description that includes:
            - The specific question/request
            - Relevant context from conversation
            - What type of information is needed
            - Any time constraints or specifics mentioned

        PLANNING PROCESS:
        - First, think through what the user is asking for
        - Identify which tools are needed
        - Consider the order of operations
        - Write a complete task description with all context
        - Then make your tool calls

        Example:
        User: "What did Sarah say about the project last week?"
        Your planning: The user wants to find messages from Sarah about a project from last week. I need to search the conversation history.
        TASK: Search for messages from user Sarah about project-related topics from the past week

        User: "Can you summarize recent discussions?"
        Your planning: The user wants a summary of recent conversations. I should use summarize_by_time for recent activity.
        TASK: Summarize conversation history from the past few days to capture recent discussions

        IMPORTANT: Your task description must be self-contained with all necessary context because:
        1. The response generator will only see THIS task description, not the full conversation history
        2. Previous queries/messages are not automatically accessible
        3. All relevant details from user's current and previous messages must be included

        Example of task descriptions:
        - BAD: "Search for information based on previous queries"
        - GOOD: "Search for information about Python error handling that was discussed yesterday"

        Feel free to ask for clarification if the user's request is ambiguous.
        DO NOT HALLUCINATE OR MAKE UP INFORMATION. If you don't know the answer, say so.
        
        Format: "TASK: [Complete description with full context]"
        Then make tool calls.
        
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

        logging.info(f"Task: {task_description}\nTool calls: {response.tool_calls}")

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
        current_loop_count = state.get("loop_count", 0) + 1

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

        response_prompt = ""
        if current_loop_count >= self.max_iterations:
            response_prompt = f"""
            You are an intelligent assistant. Based on the tool results below, provide a comprehensive and final answer to complete the specified task. This is your final response - no additional tools can be called.

            TASK TO COMPLETE: {task_to_complete}

            Original User Message: "{original_user_query.content if original_user_query else 'Unknown query'}"

            Tool Results:
            {tool_results_text}

            Instructions:
            1. Focus on completing the task as described above using the information from the tool results.
            2. Provide a complete, helpful answer based on the tool results.
            3. Do not hallucinate or make up information not provided in the tool results.
            4. Be concise but thorough in your response.
            5. This is your FINAL response - you CANNOT call any additional tools.
            6. If the tool results are insufficient, clearly state what information is missing and provide the best answer possible with available data.

            Current channel ID: {state["current_channel"]}
            Current user: {state["current_user"]}

            Provide your final answer based on the tool results above. No additional tools will be called.
            """
        else:
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

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
            logging.info(f"Response contains tool calls: {tool_calls}")
        else:
            logging.info("Response does not contain any tool calls.")

        return {
            "messages": [response],
            "current_user": state["current_user"],
            "current_channel": state["current_channel"],
            "task_description": state.get("task_description", ""),
            "images": state.get("images", []),
            "loop_count": current_loop_count
        }
    
    def tools_node(self, state: State) -> dict:

        self._pending_images = []
        result = self.tools.invoke(state)

        if self._pending_images:
            result["images"] = self._pending_images
            self._pending_images = []
        
        return result

    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("conductor", self.conductor)
        builder.add_node("tools", self.tools_node)
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

        self.graph = builder.compile()
