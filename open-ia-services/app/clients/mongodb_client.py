from models.chat_document import chat_document
from models.chat_request import chat_request
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from clients.openai_client import openai_client

import os
import pymongo

class mongodb_client():
    instance = None

    def __new__(cls):
        if not cls.instance:                    
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.mongo_client =pymongo.MongoClient(os.getenv("MONGO_URI"))
        self.mongo_db_vector = os.getenv("MONGO_CONTEXT_VECTOR_SEARCH")
        self.embeddings = openai_client.get_emmbeddings()

    def find_chat(self, request: chat_request):
        db = self.mongo_client.CustomsServices
        chatcollection = db.ChatMessages
            
        query = chatcollection.find_one({'key': request.cache_key})

        return query
    
    def chat_context_search(self, request: chat_request):
        db = self.mongo_client.CustomsServices
        mdbcollection = db.ContextData
        
         # vector store object
        vectorStore = MongoDBAtlasVectorSearch(mdbcollection, self.embeddings, index_name=self.mongo_db_vector)

        # Convert question to vector using OpenAI embeddings
        # Perform Atlas Vector Search using Langchain's vectorStore
        # similarity_search returns MongoDB documents most similar to the query
        search_result = ''  
        docs = vectorStore.similarity_search(request.query, K=1)
        for doc in docs:
            search_result += f"{doc.page_content}\\n"
            
        return search_result
    
    def upsert_chat(self, request: chat_request, chat_query: any, chat_messages: any):
        db = self.mongo_client.CustomsServices
        chatcollection = db.ChatMessages
        
        if(chat_query is None):
            chat =  chat_document(key=request.cache_key, messages=chat_messages)
            document = chat.model_dump()
            chatcollection.insert_one(document)
        else:
            chat =  chat_document(key=request.cache_key, messages=chat_messages)
            document = chat.model_dump()
            id = chat_query.get('_id')
            chatcollection.update_one({'_id': id},{"$set": { "messages" : document["messages"]}},False)
            
        return chat_messages
    
    def save_chat_context(self, request: chat_request, file: UploadFile):
        db = self.mongo_client.CustomsServices
        mdbcollection = db.ContextData
        file_type = file.content_type

        if file_type == "text/plain":
            loader = DirectoryLoader(self.upload_directory, glob=file.filename, show_progress=True)
            data = loader.load()

            x = MongoDBAtlasVectorSearch.from_documents(
                documents=data, 
                embedding=openai_client.get_emmbeddings(), 
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
