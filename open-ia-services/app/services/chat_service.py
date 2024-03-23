from openai import OpenAI
from models.chat_message import chat_message
from models.chat_request import chat_request
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi import UploadFile
from clients.mongodb_client import mongodb_client
from clients.openai_client import openai_client

import json
import os

class chat_service():
    instance = None
    
    def __new__(cls):
        if not cls.instance:                    
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL")
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_port = os.getenv("REDIS_PORT")
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.upload_directory = os.getenv("UPLOAD_DIRECTORY")

        self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.mongo_client = mongodb_client()
        self.openai = openai_client()
    
    def process_rag_chat(self, request: chat_request):
        chat_messages = []

        chat_query = self.mongo_client.find_chat(request)
        if(chat_query is None):
            # vector context search
            search_result = self.mongo_client.chat_context_search(request)
            chat_messages=[
                chat_message(role="system",content="You are a helpful assistant."),
                chat_message(role="user",content="Answer this user query: " + request.query + " with the following context: " + search_result)
            ]
            
        else:
            chat_messages.extend(chat_query["messages"])
            chat_messages.append(chat_message(role="user", content=request.query))

        # openAI request
        response = self.openai.process_chat(chat_messages)

        chat_model = json.loads(response.model_dump_json())
        chat_messages.append(chat_message(role="assistant", content=chat_model["choices"][0]["message"]["content"]))
    
        self.mongo_client.upsert_chat(request, chat_query, chat_messages)

        return chat_messages
    
    def upload_rag_document_context(self, file: UploadFile):
        try:
            file_type = file.content_type
            file_path = f"{self.upload_directory}\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        db = self.mongo_client.CustomsServices
        mdbcollection = db.ContextData

        if file_type == "text/plain":
            loader = DirectoryLoader(self.upload_directory, glob=file.filename, show_progress=True)
            data = loader.load()

            x = MongoDBAtlasVectorSearch.from_documents(
                documents=data, 
                embedding=self.embeddings, 
                collection=mdbcollection, 
                index_name=self.mongo_db_vector)

        if file_type == "application/pdf":
            loader = PyPDFLoader(f"{self.upload_directory}{file.filename}")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
            docs = text_splitter.split_documents(data)

            # insert the documents in MongoDB Atlas Vector Search
            x = MongoDBAtlasVectorSearch.from_documents(
                documents=docs, 
                embedding=self.embeddings, 
                collection=mdbcollection, 
                index_name=self.mongo_db_vector)

        return file.filename
    
    def clear_document_collection(self):
        db = self.mongo_client.CustomsVectorSearch
        db.SpecDocuments.delete_many({})