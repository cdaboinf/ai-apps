from openai import OpenAI
from models.chat_request import chat_request
from fastapi import UploadFile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import json
import os
import pymongo

class rag_service():
    def __init__(self, api_key):
        self.OPENAI_API_KEY = api_key
        #self.api_key = api_key

    def upload_document(self, file: UploadFile):
        OPENAI_API_KEY = ""
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "doc_vector_search"
        MONGO_URI = ""

        try:
            file_type = file.content_type
            file_path = f"C:\\Users\\carlos.daboin\\Documents\\Dev\\Test-Apps\\python-ai\\ai-apps\\open-ia-services\\app\\routers\\files\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        EMBEDDING_FIELD_NAME = "embedding"
        client = pymongo.MongoClient(MONGO_URI)
        db = client.CustomsVectorSearch
        mdbcollection = db.SpecDocuments

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        if file_type == "text/plain":
            loader = DirectoryLoader("C:\\Users\\carlos.daboin\\Documents\\Dev\\Test-Apps\\python-ai\\ai-apps\\open-ia-services\\app\\routers\\files", glob=file.filename, show_progress=True)
            data = loader.load()

            x = MongoDBAtlasVectorSearch.from_documents(
                documents=data, 
                embedding=embeddings, 
                collection=mdbcollection, 
                index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

        if file_type == "application/pdf":
            loader = PyPDFLoader(f"C:\\Users\\carlos.daboin\\Documents\\Dev\\Test-Apps\\python-ai\\ai-apps\\open-ia-services\\app\\routers\\files\\{file.filename}")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
            docs = text_splitter.split_documents(data)

            # insert the documents in MongoDB Atlas Vector Search
            x = MongoDBAtlasVectorSearch.from_documents(
                documents=docs, 
                embedding=embeddings, 
                collection=mdbcollection, 
                index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

        return file.filename

    def search(self, request: chat_request):
        OPENAI_API_KEY = ""
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "doc_vector_search"
        MONGO_URI = ""

        # vector store object
        client = pymongo.MongoClient(MONGO_URI)
        db = client.CustomsVectorSearch
        mdbcollection = db.SpecDocuments

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorStore = MongoDBAtlasVectorSearch(mdbcollection, embeddings, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

        # Convert question to vector using OpenAI embeddings
        # Perform Atlas Vector Search using Langchain's vectorStore
        # similarity_search returns MongoDB documents most similar to the query    
        docs = vectorStore.similarity_search(request.query, K=1)
        as_output = docs[0].page_content

        # Leveraging Atlas Vector Search paired with Langchain's QARetriever

        # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
        # If it's not specified (for example like in the code below),
        # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023
        llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

        # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
        # Implements _get_relevant_documents which retrieves documents relevant to a query.
        retriever = vectorStore.as_retriever()

        # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
        # inserts them all into a prompt and passes that prompt to an LLM.
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

        # Execute the chain
        retriever_output = qa.run(request.query)

        return { "vector": as_output, "llm": retriever_output }

    def clear_document(self, request: chat_request):
        MONGO_URI = ""
        client = pymongo.MongoClient(MONGO_URI)
        db = client.CustomsVectorSearch
        db.SpecDocuments.delete_many({})

    def __get_specs_collection():
        connection_string = "mongodb+srv://carlosdaboin:tg2yFWrEazgqve1V@clusterukgrapidsearch.v9zv1e4.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(connection_string)
        db = client.CustomsVectorSearch
        collection = db.SpecDocuments
    
        return collection