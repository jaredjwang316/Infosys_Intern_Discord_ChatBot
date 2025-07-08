import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError
import datetime
import numpy as np

from langchain.schema import Document
from google import genai
from google.genai import types

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
                sender TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                bot_message TEXT NULL,
                embedding vector(768) NOT NULL
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

    def _group_user_bot_messages(self, documents: list[Document]) -> list[dict]:
        """
        Group user and bot messages into a single Document object for each unique timestamp.
        This is to ensure that we only store user messages in the similarity search.
        Args:
            documents (list[Document]): A list of Document objects with metadata containing 'role' and 'timestamp'.
        Returns:
            list[dict]: A list of dictionaries, each containing 'user', 'timestamp', 'content', and 'bot_message'.
        """
        grouped_docs = []
        
        current_docs = {}
        for i in range(len(documents)):
            doc = documents[i]
            role = doc.metadata.get('role', 'Unknown').lower()
            timestamp = doc.metadata.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            content = doc.page_content

            # If the role is user, we need to finalize the previous document.
            # If the role is bot, we can continue adding to the current document.
            if role != 'bot':
                if current_docs:
                    grouped_docs.append(current_docs)

                current_docs = {
                    'user': role,
                    'timestamp': timestamp,
                    'content': content,
                    'bot_message': ''
                }

            elif current_docs:
                bot_message = current_docs.get('bot_message', '') + content + '\n'
                current_docs['bot_message'] = bot_message

        if current_docs:
            grouped_docs.append(current_docs)

        return grouped_docs

    def add_documents(self, channel_id, documents: list[Document]) -> None:
        """
        Add documents to the remote memory (Postgres).
        Each document should have metadata with 'role' and 'timestamp'.
        Args:
            channel_id (int): The ID of the channel to add documents to.
            documents (list[Document]): A list of Document objects to add.
        """

        if not documents:
            print("No documents to add.")
            return
        
        grouped_docs = self._group_user_bot_messages(documents)
        
        filtered_docs = [
            doc for doc in grouped_docs
            if doc.get('user') and doc.get('timestamp') and doc.get('content')
        ]

        if not filtered_docs:
            print("No valid user messages found after filtering.")
            return
        
        contents_to_embed = [doc['content'] for doc in filtered_docs]

        embeddings = self._get_embedding(contents_to_embed)

        for i, doc in enumerate(filtered_docs):
            doc['embedding'] = embeddings[i]

        self._add_channel(channel_id)

        insert_query = f"""
        INSERT INTO {channel_id} (sender, timestamp, content, embedding)
        VALUES (%s, %s, %s, %s)
        """
        for doc in filtered_docs:
            self.cur.execute(insert_query, (
                doc['user'],
                doc['timestamp'],
                doc['content'],
                doc['embedding']
            ))
        print(f"Added {len(filtered_docs)} documents to channel {channel_id}.")

    def _get_embedding(self, texts: list[str]):
        """
        Generate an embedding for the given text using Google Generative AI.
        Args:
            texts (list[str]): A list of text strings to embed.
        """
        if not texts:
            return []
        
        max_batch_size = 250
        max_tokens_per_request = 18000
        max_tokens_per_text = 1900

        def estimate_tokens(text):
            return len(text) // 4 + 1

        all_embeddings = []
        current_batch = []
        current_token_count = 0

        for text in texts:
            if estimate_tokens(text) > max_tokens_per_text:
                text = text[:max_tokens_per_text * 4]
            
            text_tokens = estimate_tokens(text)
            
            if (len(current_batch) >= max_batch_size or 
                current_token_count + text_tokens > max_tokens_per_request):

                if current_batch:
                    response = self.client.embed_content(
                        model="models/text-embedding-004",
                        contents=current_batch,
                        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                    )
                    all_embeddings.extend(response.embeddings)
                    current_batch = []
                    current_token_count = 0
            
            current_batch.append(text)
            current_token_count += text_tokens

        if current_batch:
            response = self.client.embed_content(
                model="models/text-embedding-004",
                contents=current_batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings
    
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