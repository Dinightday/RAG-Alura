from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

store = LocalFileStore("./cache_embed/")

pdf_load = PyPDFLoader("documento/regras_futebol.pdf")
arquivo = pdf_load.load()

recorte = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100

).split_documents(arquivo)


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key
)

cache = CacheBackedEmbeddings.from_bytes_store(
    embed,
    store,
    namespace="Cache_embed"
)

chroma = Chroma.from_documents(
    documents=recorte, 
    embedding=cache, 
    persist_directory="./chroma_cache/"
)

response = chroma.as_retriever(
    search_kwargs={"k": 3}
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=chroma,
    return_source_documents=True
)

