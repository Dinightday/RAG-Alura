from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
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


cliente = ChatOpenAI(
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

response_chroma = chroma.as_retriever(
    search_kwargs={"k": 3}
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente especialista em esportes. Crie um resumo com {context}."),
        ("human", "{question}")
    ]
)

qa = RetrievalQA.from_chain_type(
    llm=cliente,
    chain_type="stuff",
    retriever=response_chroma,
    return_source_documents=True,
    chain_type_kwargs={"prompt": template}
)

pergunta = "Como funciona o impedimento?"

try:
    response = qa.invoke(
        {
            "query": pergunta
        }
    )

except Exception as e:
    print(f"Erro: {e}")

print(f"Pergunta: {pergunta}")
print(f"IA:\n{response["result"]}")


print("-"*80)
for i, doc in enumerate(response["source_documents"]):
    print(f"Trecho: {i+1}\nPage: {doc.metadata.get("page")}")