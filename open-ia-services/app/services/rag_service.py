from openai import OpenAI
from models.chat_request import chat_request
from fastapi import UploadFile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import pymongo
import os

class rag_service():
    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.upload_directory = os.getenv("UPLOAD_DIRECTORY")
        self.mongo_db_vector = os.getenv("MONGO_VECTOR_SEARCH")
        self.mongo_uri = os.getenv("MONGO_URI")

        self.mongo_client =pymongo.MongoClient(os.getenv("MONGO_URI"))
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


    def upload_document(self, file: UploadFile):
        try:
            file_type = file.content_type
            file_path = f"{self.upload_directory}{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        db = self.mongo_client.CustomsVectorSearch
        mdbcollection = db.SpecDocuments

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

    def search(self, request: chat_request):
        # vector store object
        db = self.mongo_client.CustomsVectorSearch
        mdbcollection = db.SpecDocuments

        vectorStore = MongoDBAtlasVectorSearch(mdbcollection, self.embeddings, index_name=self.mongo_db_vector)

        # Convert question to vector using OpenAI embeddings
        # Perform Atlas Vector Search using Langchain's vectorStore
        # similarity_search returns MongoDB documents most similar to the query    
        docs = vectorStore.similarity_search(request.query, K=1)
        as_output = docs[0].page_content

        # Leveraging Atlas Vector Search paired with Langchain's QARetriever

        # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
        # If it's not specified (for example like in the code below),
        # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023
        # self.llm

        # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
        # Implements _get_relevant_documents which retrieves documents relevant to a query.
        retriever = vectorStore.as_retriever()

        # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
        # inserts them all into a prompt and passes that prompt to an LLM.
        qa = RetrievalQA.from_chain_type(self.llm, chain_type="stuff", retriever=retriever)

        # Execute the chain
        retriever_output = qa.run(request.query)

        return { "vector": as_output, "llm": retriever_output }

    def clear_document(self, request: chat_request):
        db = self.mongo_client.CustomsVectorSearch
        db.SpecDocuments.delete_many({})