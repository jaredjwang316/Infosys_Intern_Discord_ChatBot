"""
local_memory.py

Purpose:
--------
This module defines the `LocalMemory` class, which manages short-term and long-term memory for an AI-powered
chatbot running on Discord. It supports both in-memory storage for fast session-specific operations and persistent
vector-based storage (PostgreSQL) for long-term semantic retrieval and historical context.

Key Responsibilities:
---------------------
- 🧠 Short-Term Memory: Keeps track of recent chat history per channel for context-aware conversations.
- 🗃️ Long-Term Memory: Saves embeddings and message metadata into PostgreSQL via `PGVector` for retrieval across sessions.
- 🧩 Embeddings: Uses Google's `text-embedding-004` model to embed messages for similarity search.
- 🧵 Session Memory: Tracks last command types, user-specific query history, and thread context.
- 🧽 Memory Management: Supports clearing, syncing, and retrieving chat history across channels and time ranges.

Use Cases:
----------
- Context-aware question answering based on message history.
- Generating summaries or extracting relevant information from prior interactions.
- Storing and retrieving messages using vector similarity for follow-up queries.

Key Technologies:
-----------------
- **LangChain + LangGraph** — Manages message flow and memory checkpoints.
- **Google Generative AI (Gemini)** — Embeds messages and powers AI replies.
- **InMemoryVectorStore** — Stores session messages temporarily in-memory with semantic search capability.
- **PGVector** — PostgreSQL-backed vector storage for long-term retention of embeddings.

"""
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
import datetime
from langchain_community.vectorstores import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# TODO: Memorysaver is not working correctly, something is happening with persistent memory again

class LocalMemory:
    """
    LocalMemory is a class that manages conversation history and embeddings for a chat application.
    It uses Google Generative AI for chat and embeddings.
    """
    def __init__(self):
        """
        Initializes the LocalMemory class with empty histories and Google Generative AI models.
        """

        self.total_chat_history = {}
        self.cached_chat_history = {}
        self.chat_memory_configs = {}
        self.user_query_session_history = {}
        self.last_command_type = {}

        self.model = ChatVertexAI(
            model="gemini",
            temperature=0.7,
            max_tokens=None,
            max_retries=2
        )
        self.embedding_model = VertexAIEmbeddings(
            model_name="models/text-embedding-004",
            task_type="SEMANTIC_SIMILARITY"
        )

    def _find_last_command_type(self, user_id, doc):
        """
        Determines the command type (e.g., 'ask', 'summary', 'query') based on message content.

        Args:
            user_id (str): The Discord user ID.
            doc (Document): The document containing message content and metadata.

        Returns:
            str: Detected command type or previously stored command for the user.
        """

        if str(doc.metadata.get("user")).lower() == "bot":
            return self.last_command_type[user_id]
        else:
            content = doc.page_content.strip().lower()
            if content.startswith("ask:"):
                return "ask"
            elif content.startswith("summary:"):
                return "summary_time_limited"
            elif content == "summary":
                return "summary"
            elif content.startswith("search:"):
                return "search"
            elif content.startswith("query:"):
                return "query"
            elif content == "clear":
                return "clear"
            elif content == "exit":
                return "exit"
            elif content == "gen_chat":
                return "gen_chat"
            elif content == "test":
                return "test"
            elif content == "show_history":
                return "show_history"
            elif content == "where_am_i":
                return "where_am_i"
            elif content == "help":
                return "help"
            else:
                return self.last_command_type[user_id]

    def _add_message_to_histories(self, channel_id, user_id, doc):
        """
        Adds a message to all relevant memory structures:
        - Total history
        - Cached history (for long-term sync)
        - Command history

        Args:
            channel_id (str): The Discord channel ID.
            user_id (str): The user who sent the message.
            doc (Document): The message content and metadata.
        """

        self.total_chat_history.setdefault(channel_id, InMemoryVectorStore(embedding=self.embedding_model))
        self.cached_chat_history.setdefault(channel_id, InMemoryVectorStore(embedding=self.embedding_model))
        self.user_query_session_history.setdefault(user_id, [])
        self.last_command_type.setdefault(user_id, None)

        self.total_chat_history[channel_id].add_documents([doc])
        self.cached_chat_history[channel_id].add_documents([doc])

        self.last_command_type[user_id] = self._find_last_command_type(user_id, doc)
        if self.last_command_type[user_id] == "query":
            self.user_query_session_history[user_id].append(doc)
        else:
            self.user_query_session_history[user_id] = []

    def add_message(self, channel_id, user_id, content):
        """
        Inserts a message into memory with current timestamp.

        Args:
            channel_id (str): Discord channel ID.
            user_id (str): User ID of sender.
            content (str): The message text.
        """

        timestamp = datetime.datetime.now().utcnow()
        doc = Document(page_content=content, metadata={"user": user_id, "timestamp": timestamp})
        self._add_message_to_histories(channel_id, user_id, doc)

    def add_messages(self, channel_id, messages):
        """
        Inserts multiple messages into memory for the channel.

        Args:
            channel_id (str): Discord channel ID.
            messages (List[Tuple[str, str, datetime]]): Tuples of (user_id, content, timestamp).
        """

        if channel_id not in self.total_chat_history:
            self.total_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
            self.cached_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
            self.user_query_session_history[channel_id] = []
            self.last_command_type[channel_id] = None
        
        for user_id, content, timestamp in messages:
            doc = Document(page_content=content, metadata={"user": user_id, "timestamp": timestamp})
            self._add_message_to_histories(channel_id, user_id, doc)

    def set_chat_history(self, channel_id, messages):
        """
        Overwrites memory history with new message list for a channel.

        Args:
            channel_id (str): Discord channel ID.
            messages (List[Tuple[str, str, datetime]]): Tuples of (user_id, content, timestamp).
        """

        self.total_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
        self.cached_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
        self.user_query_session_history[channel_id] = []
        self.last_command_type[channel_id] = None

        self.add_messages(channel_id, messages)

    def get_chat_history(self, channel_id):
        """
        Retrieves full chat history for a channel.

        Args:
            channel_id (str): Discord channel ID.

        Returns:
            List[Tuple[str, str, datetime]]: List of (user, content, timestamp).
        """

        if channel_id not in self.total_chat_history:
            return []
        
        chat_history = self.total_chat_history[channel_id]

        formatted_history = []
        for doc in chat_history.store.values():
            user = doc['metadata']['user']
            content = doc['text']
            timestamp = doc['metadata']['timestamp']
            formatted_history.append((user, content, timestamp))

        return formatted_history

    def get_vectorstore(self, channel_id):
        """
        Returns the current vector store for a channel.

        Args:
            channel_id (str): Discord channel ID.

        Returns:
            InMemoryVectorStore: Vector store for similarity search.
        """

        if channel_id not in self.total_chat_history:
            return InMemoryVectorStore(embedding=self.embedding_model)
        
        return self.total_chat_history[channel_id]
    
    def get_last_command_type(self, user_id):
        """
        Gets the last command type issued by a user.

        Args:
            user_id (str): Discord user ID.

        Returns:
            str or None: Last command type string.
        """

        return self.last_command_type.get(user_id, None)
    
    def get_cached_history_documents(self, channel_id):
        """
        Converts cached chat history into LangChain Document objects with metadata.

        Args:
            channel_id (str): Discord channel ID.

        Returns:
            List[Document]: Cached documents to be stored in long-term memory.
        """

        if channel_id not in self.cached_chat_history:
            return []
        
        cached_history = self.cached_chat_history[channel_id]
        documents = []
        for doc_id, doc in cached_history.store.items():
            metadata = doc['metadata'].copy()

            if 'timestamp' in metadata and isinstance(metadata['timestamp'], datetime.datetime):
                metadata['timestamp'] = metadata['timestamp'].isoformat()

            metadata['channel_id'] = str(channel_id)

            user = metadata.get('user', 'Unknown')
            if str(user).lower() == 'bot':
                metadata['role'] = 'bot'
            else:
                metadata['role'] = str(user)
            
            documents.append(Document(page_content=doc['text'], metadata=metadata))

        return documents
    
    def clear_cached_history(self, channel_id):
        """
        Clears short-term memory (cached) for a specific channel.

        Args:
            channel_id (str): Discord channel ID.
        """

        if channel_id in self.cached_chat_history:
            del self.cached_chat_history[channel_id]
            print(f"Cleared cached history for channel {channel_id}.")
        else:
            print(f"No cached history found for channel {channel_id}.")

    def clear_chat_history(self, channel_id):
        """
        Clears all memory (short and total) for a specific channel.

        Args:
            channel_id (str): Discord channel ID.
        """

        if channel_id in self.total_chat_history:
            del self.total_chat_history[channel_id]
        if channel_id in self.cached_chat_history:
            del self.cached_chat_history[channel_id]
        if channel_id in self.user_query_session_history:
            del self.user_query_session_history[channel_id]
        if channel_id in self.last_command_type:
            del self.last_command_type[channel_id]
            
        print(f"Cleared total history for channel {channel_id}.")

    def clear_all_cached_histories(self):
        """
        Clears all cached chat histories across all channels.
        This removes all cached messages but retains the total chat history.
        """

        self.cached_chat_history.clear()
        print("Cleared all cached histories.")

    def clear_all_total_histories(self):
        """
        Clears all total chat histories across all channels.
        This removes all messages from the total history and cached history.
        """

        self.total_chat_history.clear()
        self.cached_chat_history.clear()
        self.user_query_session_history.clear()
        self.last_command_type.clear()
        print("Cleared all total histories.")

    def get_user_query_session_history(self, channel_id):
        """
        Returns the user's query messages (i.e., recent search intent).

        Args:
            channel_id (str): Discord channel ID.

        Returns:
            List[Document]: List of user's query-related messages.
        """

        return self.user_query_session_history.get(channel_id, [])
    
    def get_messages_by_time(self, channel_id, start_time, end_time=datetime.datetime.now().utcnow()):
        """
        Returns messages from a specific time window.

        Args:
            channel_id (str): Discord channel ID.
            start_time (datetime): Start of time range.
            end_time (datetime): End of time range (default: now).

        Returns:
            List[Tuple[str, str, datetime]]: Filtered messages.
        """

        if channel_id not in self.total_chat_history:
            return []

        chat_history = self.total_chat_history[channel_id]

        filtered_history = []
        for _, doc in chat_history.store.items():
            timestamp = doc['metadata']['timestamp']
            if start_time <= timestamp <= end_time:
                user = doc['metadata']['user']
                content = doc['text']
                filtered_history.append((user, content, timestamp))

        formatted_history = []
        for user, content, timestamp in filtered_history:
            formatted_history.append({
                'sender': user,
                'content': content,
                'timestamp': timestamp
            })

        formatted_history.sort(key=lambda x: x['timestamp'])

        return formatted_history
    
    def get_chat_embeddings(self, channel_id):
        """
        Returns embedding vectors from chat history of a given channel.

        Args:
            channel_id (str): Discord channel ID.

        Returns:
            List[Tuple[str, str, List[float]]]: (doc_id, message text, embedding vector).
        """

        if channel_id not in self.total_chat_history:
            return []
        chat_history = self.total_chat_history[channel_id]
        paired_data = []

        for doc_id, document in chat_history.store.items():            
            paired_data.append((doc_id, document['text'], document['vector']))

        return paired_data
    
    def get_oldest_message(self, channel_id):
        """
        Stores cached messages into long-term vector storage (PostgreSQL) and clears them.
        Retrieves the oldest message from the chat history for a given channel ID.
        Returns a tuple containing the user, content, and timestamp of the oldest message.

        Args:
            channel_id (str): Discord channel ID.
        """

        if channel_id not in self.total_chat_history:
            return None
        
        chat_history = self.total_chat_history[channel_id]
        if not chat_history.store:
            return None
        
        oldest_doc = min(chat_history.store.values(), key=lambda doc: doc['metadata']['timestamp'])
        user = oldest_doc['metadata']['user']
        content = oldest_doc['text']
        timestamp = oldest_doc['metadata']['timestamp']

        return user, content, timestamp