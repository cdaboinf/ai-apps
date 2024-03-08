from pydantic import BaseModel

class chat_message(BaseModel):
    role: str = ''
    content: str = ''