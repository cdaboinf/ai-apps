from typing import List
from pydantic import BaseModel

from models.chat_message import chat_message

class chat_document(BaseModel):
    key: str = ''
    messages: List[chat_message]