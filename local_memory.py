from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
import datetime
from langchain_community.vectorstores import PGVector
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

long_vectorstore = PGVector(
    collection_name="chat_embeddings",
    connection_string=connection_string,
    embedding_function=embedding_model,
    distance_strategy="cosine"
)

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
        self.user_query_session_history = {}
        self.last_command_type = {}

        self.model = ChatGoogleGenerativeAI(
            model="gemini",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

    def _find_last_command_type(self, user_id, doc):
        """
        Determines the type of command based on the content of the document.
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
        Adds a message document to the total chat history and cached chat history for a given channel ID.
        Also updates the user query session history and last command type.
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
        Adds a message to the chat history for a given channel ID.
        The message is stored as a Document with metadata including the user and timestamp.
        """

        timestamp = datetime.datetime.now().utcnow()
        doc = Document(page_content=content, metadata={"user": user_id, "timestamp": timestamp})
        self._add_message_to_histories(channel_id, user_id, doc)

    def add_messages(self, channel_id, messages):
        """
        Adds multiple messages to the chat history.
        Each message should be a tuple containing (user_id, content, timestamp).
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
        Sets the chat history for a given channel ID.
        This replaces any existing history with the provided messages.
        Each message should be a tuple containing (user_id, content, timestamp).
        """

        self.total_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
        self.cached_chat_history[channel_id] = InMemoryVectorStore(embedding=self.embedding_model)
        self.user_query_session_history[channel_id] = []
        self.last_command_type[channel_id] = None

        self.add_messages(channel_id, messages)

    def get_chat_history(self, channel_id):
        """
        Retrieves the chat history for a given channel ID.
        Returns a list of tuples containing the user, content, and timestamp of each message.
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
        Retrieves the vector store for a given channel ID.
        This is used for similarity search and other vector operations.
        Returns an InMemoryVectorStore object.
        """

        if channel_id not in self.total_chat_history:
            return InMemoryVectorStore(embedding=self.embedding_model)
        
        return self.total_chat_history[channel_id]
    
    def get_last_command_type(self, user_id):
        """
        Retrieves the last command type for a given channel ID.
        The command type is determined based on the last message in the chat history.
        Returns None if no command type is found.
        """

        return self.last_command_type.get(user_id, None)
    
    def get_cached_history_documents(self, channel_id):
        """
        Retrieves the cached chat history documents for a given channel ID.
        Returns a list of Document objects representing the cached history.
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
        Clears the cached chat history for a given channel ID.
        This removes all cached messages but retains the total chat history.
        """

        if channel_id in self.cached_chat_history:
            del self.cached_chat_history[channel_id]
            print(f"Cleared cached history for channel {channel_id}.")
        else:
            print(f"No cached history found for channel {channel_id}.")

    def clear_chat_history(self, channel_id):
        """
        Clears the total chat history for a given channel ID.
        This removes all messages from the total history and cached history.
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
        Retrieves the user query session history for a given channel ID.
        Returns a list of Document objects representing the user queries in the session.
        """

        return self.user_query_session_history.get(channel_id, [])
    
    def get_messages_by_time(self, channel_id, start_time, end_time=datetime.datetime.now().utcnow()):
        """
        Retrieves messages from the chat history for a given channel ID within a specified time range.
        Returns a list of tuples containing the user, content, and timestamp of each message.
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

        return filtered_history
    
    def get_chat_embeddings(self, channel_id):
        """
        Retrieves the embeddings for the chat history of a given channel ID.
        Returns a list of Document objects with embeddings.
        """

        if channel_id not in self.total_chat_history:
            return []
        chat_history = self.total_chat_history[channel_id]
        paired_data = []

        for doc_id, document in chat_history.store.items():            
            paired_data.append((doc_id, document['text'], document['vector']))

        return paired_data

    def store_in_long_term_memory(self, channel_id):
        """
        Stores the cached chat history into long-term memory (PostgreSQL vector store).
        This is used to persist chat history for future retrieval.
        """

        if channel_id not in self.cached_chat_history:
            print(f"No cached history found for channel {channel_id}. Nothing to store.")
            return
        
        documents = self.get_cached_history_documents(channel_id)

        if documents:
            long_vectorstore.add_documents(documents)
            self.clear_cached_history(channel_id)
            print(f"Stored {len(documents)} documents from channel {channel_id} into long-term memory.")
        else:
            print(f"No documents to store for channel {channel_id}.")

    def store_all_in_long_term_memory(self):
        """
        Stores all cached chat histories across all channels into long-term memory.
        This is used to persist all chat history for future retrieval.
        """

        channel_ids = list(self.cached_chat_history.keys())
        
        for channel_id in channel_ids:
            self.store_in_long_term_memory(channel_id)
        
        print("Stored all cached histories into long-term memory.")