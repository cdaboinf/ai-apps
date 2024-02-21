from openai import OpenAI
from models.chat_request import chat_request
from services import IChatService

import json
import os
import redis

class chat_service():
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def process_chat(self, request: chat_request):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", self.api_key))

        #get cached chat
        r = redis.Redis(host='redis-10140.c282.east-us-mz.azure.cloud.redislabs.com', port=10140, password='e3dVGa5lAaJ9DOfFWF3gGMvJQQn1GCkd')

        chat_messages = []

        #if null - create cache for new chat
        if(not r.exists(request.cache_key)):
            chat_messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.query}
            ] 
            
        #if exists - add new role user query to messages
        else:
            chat_messages.extend(json.loads(r.get(request.cache_key)))
            chat_messages.append({"role": "user", "content": request.query})

        # OpenAI request
        response = client.chat.completions.create(
            model=self.model,
            messages= chat_messages,
            temperature=0,
        )

        # append last open-ai response
        chat_model = json.loads(response.model_dump_json())
        chat_messages.append({"role": "assistant", "content": chat_model["choices"][0]["message"]["content"]})
        
        # cache updated chat messages
        r.set(request.cache_key, json.dumps(chat_messages))

        return chat_messages