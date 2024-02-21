from pydantic import BaseModel

class chat_request(BaseModel):
    query: str = ''
    session: str = ''
    cache_key: str = ''