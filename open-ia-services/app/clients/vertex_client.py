import vertexai
from vertexai.language_models import TextGenerationModel
from langchain.embeddings.openai import ChatGoogleGenerativeAI

import os

class vertex_client():
    instance = None

    def __new__(cls):
        if not cls.instance:                    
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.project_id = os.getenv("GENESIS_MODEL")
        self.location = os.getenv("GOOGLE_API_KEY")


    def process_chat(self, chat_messages: any):
        vertexai.init(project=project_id, location=location)
        # TODO developer - override these parameters as needed:
        parameters = {
            "temperature": 0.2,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        model = TextGenerationModel.from_pretrained("text-bison@002")
        response = model.predict(
            "Give me ten interview questions for the role of program manager.",
            **parameters,
        )
        print(f"Response from Model: {response.text}")

        return response.text
    
    def get_emmbeddings():
        embeddings = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.api_key)
        return embeddings
