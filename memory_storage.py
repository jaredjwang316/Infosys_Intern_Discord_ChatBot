"""
memory_storage.py

This module defines the MemoryStorage class, which manages hybrid memory for a chat application
using both in-memory (short-term) and persistent PostgreSQL-based vector memory (long-term).

MemoryStorage intelligently combines fast local retrieval with long-term archival,
enabling real-time conversation handling, summarization, and semantic search.

Components:
- LocalMemory: Keeps recent conversations in RAM for quick access.
- RemoteMemory: Stores embedded documents in a PostgreSQL + PGVector vector store.
- MemoryStorage: Coordinates between LocalMemory and RemoteMemory, supporting:
    • Adding messages
    • Flushing to long-term storage
    • Time-range retrieval
    • Semantic search on past data
"""

from datetime import datetime
from langchain_core.vectorstores import InMemoryVectorStore
from memory.remote_memory import RemoteMemory
from memory.local_memory import LocalMemory
from langgraph.checkpoint.memory import MemorySaver

MAX_CACHED_MESSAGES = 20

class MemoryStorage:
    """
    MemoryStorage is the central manager for hybrid memory in the chat system.

    It handles:
    - Caching recent messages in memory
    - Flushing memory to long-term PostgreSQL vector store
    - Semantic search across both memory types
    - Time-based retrieval of messages

    Combines:
    - LocalMemory (InMemoryVectorStore) for short-term speed
    - RemoteMemory (PGVector/PostgreSQL) for persistence and scale
    """
    def __init__(self):
        self.local_memory = LocalMemory()
        self.remote_memory = RemoteMemory()
        self.memory_saver = MemorySaver()

    def add_message(self, channel_id: int, user_id: int, content: str):
        """
        Adds a message to the chat history for a given channel ID.
        The message is stored as a Document with metadata including the user and timestamp.
        Args:
            channel_id (int): The ID of the channel to add the message to.
            user_id (int): The ID of the user sending the message.
            content (str): The content of the message.
        """

        self.local_memory.add_message(channel_id, user_id, content)

        if len(self.local_memory.total_chat_history.get(channel_id).store.values()) > 20:
            self.store_in_long_term_memory(channel_id)

    def get_local_vectorstore(self, channel_id: int) -> InMemoryVectorStore:
        """
        Retrieves the local vector store for a given channel ID.
        This is used for similarity search and other vector operations.
        Returns an InMemoryVectorStore object.
        Args:
            channel_id (int): The ID of the channel to retrieve the vector store for.
        Returns:
            InMemoryVectorStore: The vector store containing the cached chat history for the specified channel.
        """

        return self.local_memory.get_vectorstore(channel_id)
    
    def get_local_messages(self, channel_id: int) -> list[dict]:
        """
        Retrieves the cached chat history for a given channel ID.
        This is used to access the recent messages stored in local memory.
        Returns a list of Document objects representing the chat history.
        Args:
            channel_id (int): The ID of the channel to retrieve the chat history for.
        Returns:
            list[dict]: A list of dictionaries representing the cached chat history for the specified channel.
        """

        return self.local_memory.get_chat_history(channel_id)

    def get_local_vectorstore(self, channel_id: int) -> InMemoryVectorStore:
        """
        Retrieves the local vector store for a given channel ID.
        This is used for similarity search and other vector operations.
        Returns an InMemoryVectorStore object.
        Args:
            channel_id (int): The ID of the channel to retrieve the vector store for.
        Returns:
            InMemoryVectorStore: The vector store containing the cached chat history for the specified channel.
        """

        return self.local_memory.get_vectorstore(channel_id)
    
    def get_memory_saver(self) -> MemorySaver:
        """
        Retrieves the MemorySaver instance used for checkpointing.
        This is used to save and restore the state of the memory storage.
        Returns:
            MemorySaver: The MemorySaver instance for checkpointing.
        """

        return self.memory_saver
    
    def search_long_term_memory(self, channel_id: int, search_query: str, k: int = 5, similarity_threshold: float = 0.7) -> list[dict]:
        """
        Searches the long-term memory (PostgreSQL vector store) for documents matching the search query.
        This is used to retrieve relevant information from the stored chat history.
        Returns a list of Document objects that match the search criteria.
        Args:
            channel_id (int): The ID of the channel to search in.
            search_query (str): The query string to search for.
            k (int): The number of top results to return. Default is 5.
            similarity_threshold (float): The minimum similarity score for results. Default is 0.7.
        Returns:
            list[dict]: A list of dictionaries representing the documents that match the search query.
        """

        return self.remote_memory.search_documents(channel_id, search_query, k, similarity_threshold)
    
    def search_by_time(self, channel_id: int, start_time: datetime, end_time: datetime = None) -> list[dict]:
        """
        Searches the cached chat history for messages within a specific time range.
        This is used to retrieve messages that were sent between the specified start and end times.
        Returns a list of Document objects representing the messages in the specified time range.
        Args:
            channel_id (int): The ID of the channel to search in.
            start_time (datetime): The start time of the range to search for messages.
            end_time (datetime, optional): The end time of the range to search for messages. Defaults to now.
        Returns:
            list[dict]: A list of dictionaries representing the messages in the specified time range.
                Format of dictionary:
                {
                    "sender": "user_id",
                    "timestamp": datetime,
                    "content": "Message content",
                    "bot_message": "Bot response content"
                }
        """

        if end_time is None:
            end_time = datetime.now()

        if start_time > end_time:
            raise ValueError("Start time must be before end time.")
        
        _, _, oldest_local_time = self.local_memory.get_oldest_message(channel_id)

        if oldest_local_time is None or start_time < oldest_local_time:
            self.store_in_long_term_memory(channel_id)
            return self.remote_memory.search_by_time(channel_id, start_time, end_time)
        else:
            formatted_messages = self._group_user_bot_messages(
                self.local_memory.get_messages_by_time(channel_id, start_time, end_time)
            )
            return formatted_messages

    def _group_user_bot_messages(self, messages: list[dict]) -> list[dict]:
        """
        Groups user and bot messages together based on their timestamps.
        This is used to create a more coherent chat history for summarization.
        Args:
            messages (list[dict]): A list of dictionaries representing the chat messages.
        Returns:
            list[dict]: A list of dictionaries where each dictionary contains a user's message and the corresponding bot messages.
        Each dictionary has the following structure:
        {
            'sender': 'user_id',
            'content': 'User message content',
            'timestamp': datetime,
            'bot_message': 'Bot response content'
        }
        """

        grouped_messages = []
        current_group = {}

        for message in messages:
            sender = message.get('sender', 'Unknown').lower()
            content = message.get('content', '')
            timestamp = message.get('timestamp', datetime.now())

            if sender != 'bot':
                if current_group:
                    grouped_messages.append(current_group)
                current_group = {
                    'sender': sender,
                    'content': content,
                    'timestamp': timestamp,
                    'bot_message': ''
                }
            elif current_group:
                current_group['bot_message'] += f"\n{content}"

        if current_group:
            grouped_messages.append(current_group)

        return grouped_messages


    def store_in_long_term_memory(self, channel_id: int):
        """
        Stores the cached chat history into long-term memory (PostgreSQL vector store).
        This is used to persist chat history for future retrieval.
        Args:
            channel_id (int): The ID of the channel to store the chat history for.
        """

        if channel_id not in self.local_memory.cached_chat_history:
            print(f"No cached history found for channel {channel_id}. Nothing to store.")
            return

        documents = self.local_memory.get_cached_history_documents(channel_id)

        if documents:
            self.remote_memory.add_documents(channel_id, documents)
            self.local_memory.clear_cached_history(channel_id)
            print(f"Stored {len(documents)} documents from channel {channel_id} into long-term memory.")
        else:
            print(f"No documents to store for channel {channel_id}.")

    def store_all_in_long_term_memory(self):
        """
        Stores all cached chat histories across all channels into long-term memory.
        This is used to persist all chat history for future retrieval.
        """

        channel_ids = list(self.local_memory.cached_chat_history.keys())

        for channel_id in channel_ids:
            self.store_in_long_term_memory(channel_id)
        
        print("Stored all cached histories into long-term memory.")

    def get_config(self, channel_id: int) -> dict:
        """
        Retrieves the configuration for a given channel ID.
        This is used to access settings or preferences associated with the channel.
        Args:
            channel_id (int): The ID of the channel to retrieve the configuration for.
        Returns:
            dict: A dictionary containing the configuration settings for the specified channel.
        """

        return {"configurable": {"thread_id": str(self.remote_memory.get_thread_id(channel_id))}}

memory_storage = MemoryStorage()