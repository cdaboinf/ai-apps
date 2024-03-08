from openai import OpenAI
from models.chat_document import chat_document
from models.chat_message import chat_message
from models.chat_request import chat_request
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi import UploadFile

import json
import os
import redis
import pymongo

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
        self.mongo_client =pymongo.MongoClient(os.getenv("MONGO_URI"))
        self.mongo_db_vector = os.getenv("MONGO_CONTEXT_VECTOR_SEARCH")
    
    def process_rag_chat(self, request: chat_request):
        db = self.mongo_client.CustomsServices
        mdbcollection = db.ContextData
        chatcollection = db.ChatMessages
           
        #get cached chat
        r = redis.Redis(host=self.redis_host, port=self.redis_port, password=self.redis_password)

        chat_messages = []

        #if null - create cache for new chat
        #if(not r.exists(request.cache_key)): # redis cache check
        chat_query = chatcollection.find_one({'key': request.cache_key})
        if(chat_query is None):
            # vector store object
            vectorStore = MongoDBAtlasVectorSearch(mdbcollection, self.embeddings, index_name=self.mongo_db_vector)

            # Convert question to vector using OpenAI embeddings
            # Perform Atlas Vector Search using Langchain's vectorStore
            # similarity_search returns MongoDB documents most similar to the query
            search_result = ''  
            docs = vectorStore.similarity_search(request.query, K=1)
            for doc in docs:
                search_result += f"{doc.page_content}\\n"
                
            chat_messages=[
                chat_message(role="system",content="You are a helpful assistant."),
                chat_message(role="user",content="Answer this user query: " + request.query + " with the following context: " + search_result)
            ]
            
        #if exists - add new role user query to messages
        else:
            # redis cache retrieval 
            #chat_messages.extend(json.loads(r.get(request.cache_key)))
            # memory cache retrieval
            chat_messages.extend(chat_query["messages"])
            chat_messages.append(chat_message(role="user", content=request.query))

        # OpenAI request
        response = self.open_ai_client.chat.completions.create(
            model=self.model,
            messages= chat_messages,
            temperature=0,
        )

        # append last open-ai response
        chat_model = json.loads(response.model_dump_json())
        chat_messages.append(chat_message(role="assistant", content=chat_model["choices"][0]["message"]["content"]))
        
        # cache updated chat messages/expires in 2 mins after last update
        # redis
        #r.set(request.cache_key, json.dumps(chat_messages))
        #r.expire(request.cache_key, 300)

        # mongodb
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