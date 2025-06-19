import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# gemini
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# takes text (like "Hi how are you") and returns a high-dimensional vector that represents the semantic meaning.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def search_conversation(history, search_query):
    # turn history into Documents
    docs = [
        Document(page_content=message, metadata={"role": role})
        for role, message in history
    ]

    # chunk into ~1000-char slices with 200-char overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3. Store in pgvector 
    # The vectors and associated text data will be stored in the chat_embeddings table created in my schema
    vectorstore = PGVector(
        collection_name="chat_embeddings",  #specifies the table name in your PostgreSQL database where the vectors and associated text data will be stored.
        connection_string=connection_string,
        embedding_function=embedding_model,     #convert your input text into semantic vector embeddings 
        distance_strategy="cosine"  #cosine measures how similar vectors are directionally, not in terms of magnitude 
    )

    # add_documents only once (or check if already in DB)
    vectorstore.add_documents(chunks)

    # build a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="map_reduce", # robust for aggregation
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # top-5 chunks
        return_source_documents=False
    )

    prompt = (
        f"Please gather *all* the information related to “{search_query}” "
        "from the conversation, and present it as concise bullet points."
    )

    # return qa.run(prompt) # deprecated
    #This runs the chain and returns just the final Gemini response
    return qa.invoke(prompt)['result']