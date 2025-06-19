import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Annoy, InMemoryVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME")

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

# uses Chroma, FAISS is hard to install on macos - rochan
def search_conversation(vectorstore, search_query):
    # build a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # run
    prompt = (
        f"Please gather *all* the information related to “{search_query}” "
        "from the conversation, and present it as concise bullet points."
    )

    return qa.invoke(prompt)['result']