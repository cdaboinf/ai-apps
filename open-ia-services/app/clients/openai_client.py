from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

import os

class openai_client():
    instance = None

    def __new__(cls):
        if not cls.instance:                    
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL")

        self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def process_chat(self, chat_messages: any):
        response = self.open_ai_client.chat.completions.create(
            model=self.model,
            messages= chat_messages,
            temperature=0,
        )

        return response
    
    def get_emmbeddings():
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        return embeddings
