import os
from dotenv import load_dotenv
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Annoy, InMemoryVectorStore, PGVector
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

db_host = os.getenv("PG_DB_HOST")
db_port = os.getenv("PG_DB_PORT")
db_name = os.getenv("PG_DB_NAME")
db_user = os.getenv("PG_DB_USER")
db_password = os.getenv("PG_DB_PASSWORD")

# gemini
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

long_vectorstore = PGVector(
    collection_name="chat_embeddings",
    connection_string=connection_string,
    embedding_function=embedding_model,
    distance_strategy="cosine"
)

def search_conversation_quick(short_vectorstore, search_query):
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

def search_conversation(search_query, cached_chat_history, channel_id, quick_result):
    print("Searching using both short-term and long-term memory...")

    if cached_chat_history:
        long_vectorstore.add_documents(cached_chat_history)

    print("🔍 Searching long-term memory...")
    long_results = long_vectorstore.similarity_search(search_query, k=5)
    
    # Remove duplicates based on content
    unique_results = []
    seen_content = set()
    for doc in long_results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    
    print(f"📊 Found {len(long_results)} in long-term")
    
    if not unique_results:
        return None
    
    # Create a combined context from both short and long term results
    combined_context = ""
    for i, doc in enumerate(unique_results):
        role = doc.metadata.get('role', 'Unknown')
        timestamp = doc.metadata.get('timestamp', 'Unknown time')
        combined_context += f"[{i+1}] {role} ({timestamp}): {doc.page_content}\n"
    
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
