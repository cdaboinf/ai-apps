import pymongo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param
import nltk

client = pymongo.MongoClient(key_param.MONGO_URI)
dbName = "rag_demo"
collectionName = "vector_logs"
collection = client[dbName][collectionName]

loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection )

def load_text(text:str):
    return true

def load_text_files():
    loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
    data = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
    vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection )
    return true

def load_pdf_files(file_name:str):
    loader = PyPDFLoader("./sample_files", glob="SPEC_GEN1014_SR417944_Web Automated Kronos New Hire Import_V16.0.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    docs = text_splitter.split_documents(data)

    x = MongoDBAtlasVectorSearch.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(disallowed_special=()), 
        collection=mdbcollection, 
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)
    return true