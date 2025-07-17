"""
search.py

Purpose:
--------
This module implements search capabilities over both short-term (in-memory) and long-term (vector database) 
chat history using semantic similarity. It leverages Google's Gemini model via LangChain to search relevant 
information from past conversations.

Key Technologies:
-----------------
- 🔍 **GoogleGenerativeAIEmbeddings** — Generates semantic embeddings for stored chat messages.
- 🧠 **PGVector** — Manages long-term memory using a PostgreSQL + vector database backend.

This file enables intelligent, context-aware recall of prior discussion points to answer follow-up questions 
or regenerate summary insights.
"""

import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from memory_storage import memory_storage

load_dotenv()
model_name = os.getenv("MODEL_NAME")

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# gemini
model = ChatVertexAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    max_retries=2
)

embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004"
)

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def search_conversation_quick(short_vectorstore, search_query):
    """
    Searches recent conversation history (short-term memory) for messages semantically similar to the user's query.

    Parameters:
        short_vectorstore (VectorStore): A temporary in-memory vectorstore containing recent chat messages.
        search_query (str): The user's current query to search for in prior messages.

    Returns:
        str: A Gemini-generated summary of relevant recent messages, or fallback raw excerpts if generation fails.
    """
    print("Searching short-term memory...")
    short_results = short_vectorstore.similarity_search(search_query, k=5)

    if not short_results:
        return "No recent information found in conversation history."
    
    quick_context = ""
    for i, doc in enumerate(short_results):
        role = doc.metadata.get('role', 'Unknown')
        timestamp = doc.metadata.get('timestamp', 'Unknown time')
        quick_context += f"[{i+1}] {role} ({timestamp}): {doc.page_content}\n"

    quick_prompt = f"""
    Based on recent conversation messages, provide a brief summary of information related to "{search_query}":
    
    Recent messages:
    {quick_context}
    
    Keep the response concise and focused on the most relevant recent information.
    """

    try:
        response = model.invoke(quick_prompt)
        return response.content.strip()
    
    except Exception as e:
        print(f"Error generating quick response: {e}")
        
        fallback_response = f"Search results for '{search_query}':\n\n"
        for i, doc in enumerate(short_results[:10]):
            role = doc.metadata.get('role', 'Unknown')
            timestamp = doc.metadata.get('timestamp', 'Unknown time')
            fallback_response += f"• {role} ({timestamp}): {doc.page_content}\n"
        return fallback_response

def search_conversation(channel_id, search_query, quick_result):
    """
    Performs a full search over both short-term and long-term memory to retrieve and synthesize relevant information.

    Parameters:
        search_query (str): The user’s search intent or topic of interest.
        cached_chat_history (List[Document]): List of recent chat history entries to store in long-term memory.
        quick_result (str): The summary previously generated from short-term memory to avoid redundancy.

    Returns:
        str: A synthesized, bullet-point summary of all information related to the query, drawn from long-term memory.
        Falls back to raw results if searching fails.
    """
    print("Searching using both short-term and long-term memory...")

    memory_storage.store_in_long_term_memory(channel_id)
    
    print("Finished adding cached chat history to long-term memory.")

    print("🔍 Searching long-term memory...")
    long_results = memory_storage.search_long_term_memory(channel_id, search_query, 5, 0.7)
    
    # Remove duplicates based on content
    unique_results = []
    seen_content = set()
    for doc in long_results:
        if doc.get('content') not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.get('content'))
    
    print(f"📊 Found {len(unique_results)} in long-term")
    print(unique_results)
    
    if not unique_results:
        return None
    
    # Create a combined context from both short and long term results
    combined_context = ""
    for i, doc in enumerate(unique_results):
        role = doc.get('sender', 'Unknown')
        timestamp = doc.get('timestamp', 'Unknown time')
        combined_context += f"[{i+1}] {role} ({timestamp}): {doc.get('content')}\n"

    # Use the model to synthesize information from both sources
    synthesis_prompt = f"""
    Based on the following conversation excerpts from historical messages, 
    please gather *all* the information related to "{search_query}" and present it as concise bullet points.
    If the information has already been mentioned in recent summarys, do not repeat it.
    If the information is new, include it in the summary.
    
    Recent search result from short-term memory:
    {quick_result}

    Conversation excerpts:
    {combined_context}
    
    Please organize the information chronologically where possible and indicate if information comes from recent or older conversations.
    Please follow the format of the previous summaries, using bullet points for each piece of information.
    """
    
    try:
        response = model.invoke(synthesis_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        # Fallback: return raw results if model fails
        fallback_response = f"Search results for '{search_query}':\n\n"
        for i, doc in enumerate(unique_results[:10]):  # Limit to top 10 results
            role = doc.metadata.get('role', 'Unknown')
            timestamp = doc.metadata.get('timestamp', 'Unknown time')
            fallback_response += f"• {role} ({timestamp}): {doc.page_content}\n"
        return fallback_response
