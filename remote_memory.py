import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError
import datetime
import numpy as np

from langchain.schema import Document
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

PG_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT", 5432),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname":   os.getenv("DB_NAME"),
}

# TODO: Make sure to either combine bot messages with user messages, but main thing is just 
# to keep the similarity searches based only on user messages, not bot messages.

class RemoteMemory:
    def __init__(self):
        print("Connecting to Postgres...")
        try:
            self.conn = psycopg2.connect(**PG_CONFIG)
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
        except OperationalError as e:
            print("Could not connect to Postgres:", e)
            raise
        print("Connected to Postgres successfully!")

        self.client = genai.Client(api_key=api_key)

    def _add_channel(self, channel_id: int) -> None:
        """
        Add a new channel to the remote memory (Postgres).
        Only stores the channel_id in the channels table.
        Creates a new table for the channel if it does not exist.
        Args:
            channel_id (int): The ID of the channel to add.
        """
        check_query = """
        SELECT channel_id
        FROM channels
        WHERE channel_id = %s;
        """
        self.cur.execute(check_query, (channel_id,))
        result = self.cur.fetchone()

        if not result:
            insert_query = """
            INSERT INTO channels (channel_id)
            VALUES (%s)
            """
            self.cur.execute(insert_query, (channel_id,))

            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {channel_id} (
                message_id SERIAL PRIMARY KEY,
                sender TEXT,
                timestamp TIMESTAMP,
                content TEXT,
                embedding vector(768)
            );
            """
            self.cur.execute(create_table_query)

            index_query = f"""
            CREATE INDEX ON {channel_id}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10)
            """

            self.cur.execute(index_query)

            print(f"Channel added: {channel_id}")

    # TODO: Implement this function to group user and bot messages.
    def _group_user_bot_messages(self, documents: list[Document]) -> list:
        """
        Group user and bot messages into a single Document object for each unique timestamp.
        This is to ensure that we only store user messages in the similarity search.
        Args:
            documents (list[Document]): A list of Document objects with metadata containing 'role' and 'timestamp'.
        Returns:
            list: A list of grouped dicts.
        """
        grouped_docs = []
        
        for i in range(len(documents)):
            doc = documents[i]
            role = doc.metadata.get('role', 'Unknown')
            timestamp = doc.metadata.get('timestamp', datetime.datetime.now(datetime.timezone.utc))

            if i < len(documents) - 1:
                pass

    def add_documents(self, channel_id, documents: list[Document]) -> None:
        """
        Add documents to the remote memory (Postgres).
        Each document should have metadata with 'role' and 'timestamp'.
        Args:
            channel_id (int): The ID of the channel to add documents to.
            documents (list[Document]): A list of Document objects to add.
        """

        texts = [doc.page_content for doc in documents]
        embeddings = self._get_embedding(texts)

        roles = [doc.metadata.get('role', 'Unknown') for doc in documents]
        timestamps = [
            doc.metadata.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            for doc in documents
        ]
        contents = texts

        data = np.array(list(zip(roles, timestamps, contents, embeddings)))

        self._add_channel(channel_id)

        insert_query = f"""
        INSERT INTO {channel_id} (sender, timestamp, content, embedding)
        VALUES (%s, %s, %s, %s)
        """
        for i in range(len(data)):
            self.cur.execute(insert_query, (
                data[i][0],  # sender
                data[i][1],  # timestamp
                data[i][2],  # content
                data[i][3]   # embedding
            ))
        print(f"Added {len(documents)} documents to channel {channel_id}.")

    def _get_embedding(self, texts: list[str]):
        """
        Generate an embedding for the given text using Google Generative AI.
        Args:
            texts (list[str]): A list of text strings to embed.
        """
        response = self.client.embed_content(
            model="models/text-embedding-004",
            contents=texts
        )
        return response.embeddings
    
    def search_documents(self, channel_id: int, query: str, k: int = 5, cutoff: float = 0.7) -> list[dict]:
        """
        Search for documents in the specified channel using a query.
        Returns the top k results with cosine similarity above the cutoff.
        Args:
            channel_id (int): The ID of the channel to search in.
            query (str): The search query.
            k (int): The number of top results to return.
            cutoff (float): The minimum cosine similarity to include a result.
        Returns:
            list[dict]: A list of dictionaries containing the search results (sender, timestamp, content, similarity).
        """
        self._add_channel(channel_id)

        query_embedding = self._get_embedding([query])[0]

        search_query = f"""
        SELECT sender, timestamp, content, embedding
        FROM {channel_id}
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        self.cur.execute(search_query, (query_embedding, k))
        results = self.cur.fetchall()

        filtered_results = []
        for result in results:
            sender, timestamp, content, embedding = result
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            if similarity >= cutoff:
                filtered_results.append({
                    "sender": sender,
                    "timestamp": timestamp,
                    "content": content,
                    "similarity": similarity
                })

        return filtered_results
    
    def search_by_time(self, channel_id: int, start_time: datetime.datetime, end_time: datetime.datetime = datetime.datetime.now()) -> list[dict]:
        """
        Search for documents in the specified channel within a time range.
        Returns documents with timestamps between start_time and end_time.
        Args:
            channel_id (int): The ID of the channel to search in.
            start_time (datetime.datetime): The start of the time range.
            end_time (datetime.datetime): The end of the time range.
        Returns:
            list[dict]: A list of dictionaries containing the search results (sender, timestamp, content).
        """
        self._add_channel(channel_id)

        search_query = f"""
        SELECT sender, timestamp, content
        FROM {channel_id}
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC;
        """
        
        self.cur.execute(search_query, (start_time, end_time))
        results = self.cur.fetchall()

        if not results:
            return "No documents found in the specified time range."

        formatted_results = []
        for sender, timestamp, content in results:
            formatted_results.append({
                "sender": sender,
                "timestamp": timestamp,
                "content": content
            })

        return formatted_results
    
    def reindex_channel(self, channel_id: int, num_lists: int) -> None:
        """
        Reindexes the specified channel with the specified number of lists.
        Args:
            channel_id (int): The ID of the channel to reindex.
            num_lists (int): The number of lists to reindex to.
        """

        index_query = f"""
        CREATE INDEX ON {channel_id}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {num_lists});
        """
        self.cur.execute(index_query)