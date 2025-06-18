import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma, Annoy
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings

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
def search_conversation(history, search_query):
    # turn history into Documents
    docs = [
        Document(page_content=message, metadata={"role": role})
        for role, message in history
    ]

    # chunk into ~1000-char slices with 200-char overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Annoy.from_documents(chunks, embedding_model, index_params={"n_trees": 10})

    # build a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="map_reduce", # robust for aggregation
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # top-5 chunks
        return_source_documents=False
    )

    # run
    prompt = (
        f"Please gather *all* the information related to “{search_query}” "
        "from the conversation, and present it as concise bullet points."
    )

    # return qa.run(prompt) # deprecated
    return qa.invoke(prompt)['result']